"""
LLM Prompt Shield — FastAPI entry point & pipeline orchestrator.

Request flow for POST /v1/chat/completions:

  auth --> rate limit --> Layer 1 (deterministic) --> Layer 2 (classifier)
        --> canary injection --> Layer 3 (upstream proxy)
        --> Layer 4 (output DLP / canary check) --> client

See README.md for the full architecture explanation and SETUP.md for
deployment instructions.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.core.audit import AuditEvent, AuditLogger, fingerprint_key, make_prompt_preview
from app.core.canary import Canary, generate_canary, inject_into_messages
from app.core.proxy import (
    UpstreamError,
    build_http_client,
    forward_non_stream,
    forward_stream,
    get_adapter,
)
from app.core.rate_limit import TokenBucketLimiter
from app.layers.layer1_deterministic import Finding, run_layer1
from app.layers.layer2_classifier import get_classifier
from app.layers.layer4_output_dlp import StreamingDLPBuffer, scan_complete_output
from app.schemas import ChatCompletionRequest
from app.core.message_content import extract_text_parts, sanitize_content

logger = logging.getLogger("prompt_shield")

settings: Settings = get_settings()
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

rate_limiter = TokenBucketLimiter(settings.RATE_LIMIT_REQUESTS_PER_MINUTE, settings.RATE_LIMIT_BURST)
audit_logger = AuditLogger(settings.AUDIT_LOG_PATH)

METRICS: dict[str, int] = {
    "requests_total": 0,
    "requests_blocked_injection": 0,
    "requests_blocked_secret_or_pii": 0,
    "requests_blocked_canary_leak": 0,
    "requests_blocked_rate_limit": 0,
    "requests_blocked_auth": 0,
    "requests_allowed": 0,
    "upstream_errors": 0,
}

http_client = None  # set in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = build_http_client(settings)
    try:
        classifier = get_classifier(settings)
        logger.info("Layer 2 classifier backend: %s", classifier.backend_name)
    except Exception:
        logger.exception("Layer 2 classifier failed to initialize at startup")
    yield
    await http_client.aclose()


app = FastAPI(
    title="LLM Prompt Shield",
    description="Inline security reverse proxy for LLM APIs — injection defense, PII/secret DLP, canary leak detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _extract_client_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "")


def _check_proxy_auth(request: Request) -> Optional[str]:
    if not settings.REQUIRE_PROXY_AUTH:
        return None
    key = _extract_client_key(request)
    if not key or key not in settings.proxy_api_key_set:
        return "Missing or invalid proxy API key. Send `Authorization: Bearer <key>`."
    return None


def _rate_limit_key(request: Request) -> str:
    key = _extract_client_key(request)
    if key:
        return key
    return request.client.host if request.client else "unknown"


def _error_response(request_id: str, status_code: int, code: str, message: str, **extra) -> JSONResponse:
    body = {"error": {"message": message, "type": "prompt_shield_blocked", "code": code, **extra}}
    return JSONResponse(status_code=status_code, content=body, headers={"X-Shield-Request-Id": request_id})


def _audit(
    request_id: str,
    client_fp: str,
    decision: str,
    started: float,
    score: Optional[float],
    injection_hit: bool,
    findings: list[Finding],
    evasion_signals: list[str],
    prompt_text: str,
    provider: str,
    reason: Optional[str] = None,
) -> None:
    latency_ms = (time.perf_counter() - started) * 1000
    event = AuditEvent(
        request_id=request_id,
        timestamp=time.time(),
        client_key_fingerprint=client_fp,
        decision=decision,
        layer2_score=round(score, 4) if score is not None else None,
        injection_heuristic_hit=injection_hit,
        secret_findings=[f.label for f in findings if f.kind == "secret"],
        pii_findings=[f.label for f in findings if f.kind == "pii"],
        evasion_signals=evasion_signals,
        prompt_preview=make_prompt_preview(prompt_text, settings.AUDIT_LOG_PROMPT_PREVIEW_CHARS),
        latency_ms=round(latency_ms, 2),
        upstream_provider=provider,
        reason=reason,
    )
    audit_logger.log(event)


# --------------------------------------------------------------------- #
# Operational endpoints
# --------------------------------------------------------------------- #
@app.get("/healthz")
async def healthz():
    classifier_backend = "unloaded"
    try:
        classifier_backend = get_classifier(settings).backend_name
    except Exception:
        classifier_backend = "error"
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "environment": settings.ENVIRONMENT,
        "upstream_provider": settings.UPSTREAM_PROVIDER,
        "classifier_backend": classifier_backend,
    }


@app.get("/metrics")
async def metrics():
    lines = [f"prompt_shield_{k} {v}" for k, v in METRICS.items()]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


# --------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------- #
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    METRICS["requests_total"] += 1

    auth_error = _check_proxy_auth(request)
    if auth_error:
        METRICS["requests_blocked_auth"] += 1
        return _error_response(request_id, 401, "authentication_error", auth_error)

    client_fp = fingerprint_key(_extract_client_key(request))

    if settings.ENABLE_RATE_LIMIT:
        allowed, retry_after = rate_limiter.allow(_rate_limit_key(request))
        if not allowed:
            METRICS["requests_blocked_rate_limit"] += 1
            resp = _error_response(request_id, 429, "rate_limited", "Rate limit exceeded. Slow down and retry.")
            resp.headers["Retry-After"] = str(int(retry_after) + 1)
            return resp

    try:
        raw_body = await request.json()
        chat_request = ChatCompletionRequest.model_validate(raw_body)
    except (ValidationError, ValueError) as exc:
        return _error_response(request_id, 400, "invalid_request", f"Invalid request body: {exc}")

    # ------------------------------------------------------------ #
    # Layer 1 — deterministic scanner (secrets / PII / injection heuristics)
    # ------------------------------------------------------------ #
    sanitized_messages: list[dict] = []
    all_findings: list[Finding] = []
    injection_heuristic_hit = False
    evasion_signals: list[str] = []
    scan_text_parts: list[str] = []

    for msg in chat_request.messages:
        msg_dict = msg.model_dump(exclude_none=True)
        content = msg.content
        text_parts = extract_text_parts(content)

        if not text_parts:
            sanitized_messages.append(msg_dict)
            continue

        per_part_results = [
            run_layer1(text, decode_evasions=settings.ENABLE_ANTI_EVASION_DECODING)
            for text in text_parts
        ]
        all_findings.extend(f for result in per_part_results for f in result.findings)
        injection_heuristic_hit = injection_heuristic_hit or any(
            result.injection_heuristic_hit for result in per_part_results
        )
        evasion_signals.extend(
            signal for result in per_part_results for signal in result.evasion_signals
        )

        def sanitize_text(text: str) -> str:
            result = run_layer1(text, decode_evasions=settings.ENABLE_ANTI_EVASION_DECODING)
            return result.sanitized_text if (
                settings.ENABLE_SECRET_REDACTION or settings.ENABLE_PII_REDACTION
            ) else text

        sanitized_content, sanitized_parts = sanitize_content(content, sanitize_text)
        msg_dict["content"] = sanitized_content
        sanitized_messages.append(msg_dict)

        if msg.role in ("user", "system"):
            scan_text_parts.extend(sanitized_parts)

    combined_scan_text = "\n".join(scan_text_parts)

    has_secret = any(f.kind == "secret" for f in all_findings)
    has_pii = any(f.kind == "pii" for f in all_findings)
    if (settings.BLOCK_ON_SECRET and has_secret) or (settings.BLOCK_ON_PII and has_pii):
        METRICS["requests_blocked_secret_or_pii"] += 1
        _audit(request_id, client_fp, "blocked_secret_or_pii", started, None, injection_heuristic_hit,
               all_findings, evasion_signals, combined_scan_text, settings.UPSTREAM_PROVIDER,
               reason="secret_or_pii_policy")
        return _error_response(request_id, 400, "secret_or_pii_detected",
                                "Request blocked: the prompt contains a hardcoded secret or PII and this "
                                "deployment is configured to block such requests.")

    # ------------------------------------------------------------ #
    # Layer 2 — ML / heuristic injection classifier
    # ------------------------------------------------------------ #
    classifier = get_classifier(settings)
    score = classifier.score(combined_scan_text)

    if score >= settings.INJECTION_SCORE_THRESHOLD:
        METRICS["requests_blocked_injection"] += 1
        _audit(request_id, client_fp, "blocked_injection", started, score, injection_heuristic_hit,
               all_findings, evasion_signals, combined_scan_text, settings.UPSTREAM_PROVIDER,
               reason=f"layer2_score={score:.3f} >= threshold={settings.INJECTION_SCORE_THRESHOLD}")
        return _error_response(
            request_id, 422, "injection_detected",
            "Request blocked: prompt injection detected.", score=round(score, 3),
        )

    # ------------------------------------------------------------ #
    # Canary injection (Layer 4 setup)
    # ------------------------------------------------------------ #
    canary: Optional[Canary] = generate_canary() if settings.ENABLE_CANARY else None
    final_messages = inject_into_messages(sanitized_messages, canary) if canary else sanitized_messages

    upstream_payload = chat_request.model_dump(exclude_none=True)
    upstream_payload["messages"] = final_messages

    adapter = get_adapter(settings)

    # ------------------------------------------------------------ #
    # Streaming path
    # ------------------------------------------------------------ #
    if chat_request.stream:
        return await _stream_response(
            request_id, client_fp, started, score, injection_heuristic_hit, all_findings,
            evasion_signals, combined_scan_text, adapter, upstream_payload, canary,
        )

    # ------------------------------------------------------------ #
    # Layer 3 — forward to upstream (non-streaming)
    # ------------------------------------------------------------ #
    try:
        response_json = await forward_non_stream(http_client, adapter, upstream_payload)
    except UpstreamError as exc:
        METRICS["upstream_errors"] += 1
        _audit(request_id, client_fp, "error", started, score, injection_heuristic_hit, all_findings,
               evasion_signals, combined_scan_text, settings.UPSTREAM_PROVIDER, reason=str(exc))
        status = exc.status_code if exc.status_code >= 400 else 502
        return _error_response(request_id, status, "upstream_error", exc.detail)

    # ------------------------------------------------------------ #
    # Layer 4 — output DLP + canary leak check
    # ------------------------------------------------------------ #
    output_text = adapter.extract_output_text(response_json)
    scan_result = scan_complete_output(output_text, canary, enable_secret_scan=settings.ENABLE_OUTPUT_SECRET_SCAN)

    if scan_result.blocked:
        METRICS["requests_blocked_canary_leak"] += 1
        _audit(request_id, client_fp, "blocked_canary_leak", started, score, injection_heuristic_hit,
               all_findings, evasion_signals, combined_scan_text, settings.UPSTREAM_PROVIDER,
               reason=scan_result.block_reason)
        return _error_response(
            request_id, 502, scan_result.block_reason or "output_blocked",
            "Response blocked: potential system-prompt or secret leak detected in model output.",
        )

    if settings.UPSTREAM_PROVIDER == "anthropic":
        final_body = adapter.to_openai_response(response_json, text_override=scan_result.sanitized_text)
    else:
        final_body = (
            adapter.apply_redacted_text(response_json, scan_result.sanitized_text)
            if scan_result.findings else response_json
        )

    METRICS["requests_allowed"] += 1
    _audit(request_id, client_fp, "allowed", started, score, injection_heuristic_hit, all_findings,
           evasion_signals, combined_scan_text, settings.UPSTREAM_PROVIDER)

    return JSONResponse(
        status_code=200,
        content=final_body,
        headers={"X-Shield-Request-Id": request_id, "X-Shield-Layer2-Score": f"{score:.3f}"},
    )


async def _stream_response(
    request_id: str,
    client_fp: str,
    started: float,
    score: float,
    injection_heuristic_hit: bool,
    all_findings: list[Finding],
    evasion_signals: list[str],
    combined_scan_text: str,
    adapter,
    upstream_payload: dict,
    canary: Optional[Canary],
) -> StreamingResponse:
    async def event_generator():
        buffer = StreamingDLPBuffer(
            canary,
            buffer_chars=settings.STREAMING_DLP_BUFFER_CHARS,
            enable_secret_scan=settings.ENABLE_OUTPUT_SECRET_SCAN,
            hold_back=settings.STRICT_STREAMING_DLP,
        )
        blocked = False
        block_reason: Optional[str] = None
        finish_sent = False

        try:
            async for chunk in forward_stream(http_client, adapter, upstream_payload):
                if chunk.is_done:
                    release, is_blocked, reason = buffer.flush()
                    if is_blocked:
                        blocked, block_reason = True, reason
                        yield adapter.encode_error(reason)
                        break
                    if release:
                        yield adapter.encode_stream_chunk(release)
                    if not finish_sent:
                        yield adapter.encode_stream_chunk("", finish_reason=chunk.finish_reason or "stop")
                        finish_sent = True
                    yield adapter.encode_done()
                    break

                if chunk.is_control:
                    # Force-release whatever is safely bufferable before
                    # forwarding a control frame, to preserve ordering.
                    release, is_blocked, reason = buffer.flush()
                    if is_blocked:
                        blocked, block_reason = True, reason
                        yield adapter.encode_error(reason)
                        break
                    if release:
                        yield adapter.encode_stream_chunk(release)
                    if chunk.finish_reason and not finish_sent:
                        yield adapter.encode_stream_chunk("", finish_reason=chunk.finish_reason)
                        finish_sent = True
                    continue

                release, is_blocked, reason = buffer.feed(chunk.text_delta)
                if is_blocked:
                    blocked, block_reason = True, reason
                    yield adapter.encode_error(reason)
                    break
                if release:
                    yield adapter.encode_stream_chunk(release)
        except UpstreamError as exc:
            METRICS["upstream_errors"] += 1
            yield adapter.encode_error(f"upstream_error: {exc.detail}")

        if blocked:
            METRICS["requests_blocked_canary_leak"] += 1
            _audit(request_id, client_fp, "blocked_canary_leak", started, score, injection_heuristic_hit,
                   all_findings, evasion_signals, combined_scan_text, settings.UPSTREAM_PROVIDER,
                   reason=block_reason)
        else:
            METRICS["requests_allowed"] += 1
            _audit(request_id, client_fp, "allowed", started, score, injection_heuristic_hit, all_findings,
                   evasion_signals, combined_scan_text, settings.UPSTREAM_PROVIDER)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Shield-Request-Id": request_id, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
