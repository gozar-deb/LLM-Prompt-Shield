"""
Layer 3 — Upstream proxy & routing engine (target: < 5ms overhead).

The shield presents a single OpenAI-compatible surface
(`/v1/chat/completions`) to clients regardless of which upstream actually
serves the request. A `ProviderAdapter` translates between that surface and
whatever the configured upstream speaks:

  * `OpenAICompatibleAdapter` — passthrough, works for OpenAI itself and for
    any OpenAI-compatible server (vLLM's `--api-key` OpenAI server mode,
    Ollama's `/v1/chat/completions` compatibility endpoint, Azure OpenAI
    with a small header tweak, etc).
  * `AnthropicAdapter` — translates request/response shapes to/from
    Anthropic's native Messages API (`/v1/messages`), since it is not
    wire-compatible with the OpenAI schema.

Both adapters expose the same interface so `main.py`'s pipeline code never
needs to know which upstream is in play.
"""
from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx

from app.config import Settings


class UpstreamError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@dataclass
class StreamChunk:
    """One normalized unit from an upstream SSE stream."""
    text_delta: str            # "" if this chunk carries no visible text
    is_control: bool           # True for role/finish_reason/ping frames with no text
    is_done: bool               # True for the terminal event ([DONE] / message_stop)
    finish_reason: Optional[str] = None


class ProviderAdapter(ABC):
    name: str

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    def build_request(self, payload: dict) -> tuple[str, dict, dict]:
        """Return (url, headers, json_body) for the upstream call."""

    @abstractmethod
    def extract_output_text(self, response_json: dict) -> str:
        """Full assistant text from a non-streaming response, for Layer 4 scanning."""

    @abstractmethod
    def apply_redacted_text(self, response_json: dict, redacted_text: str) -> dict:
        """Return a copy of response_json with the assistant text replaced."""

    @abstractmethod
    def parse_stream_line(self, line: str) -> Optional[StreamChunk]:
        """Parse one raw SSE line (already stripped of the `data:` prefix
        where applicable) into a StreamChunk, or None if the line carries
        no data (blank lines, comments)."""

    @abstractmethod
    def encode_stream_chunk(self, text: str, *, finish_reason: Optional[str] = None) -> bytes:
        """Encode a (possibly redacted/rebuffered) text delta as an
        OpenAI-style `data: {...}\\n\\n` SSE frame for the client."""

    def encode_done(self) -> bytes:
        return b"data: [DONE]\n\n"

    def encode_error(self, reason: str) -> bytes:
        payload = {"error": {"message": reason, "type": "prompt_shield_blocked", "code": "output_blocked"}}
        return f"data: {json.dumps(payload)}\n\n".encode() + b"data: [DONE]\n\n"


# --------------------------------------------------------------------- #
# OpenAI-compatible adapter (OpenAI, vLLM, Ollama, Azure-ish)
# --------------------------------------------------------------------- #
class OpenAICompatibleAdapter(ProviderAdapter):
    name = "openai_compatible"

    def build_request(self, payload: dict) -> tuple[str, dict, dict]:
        url = f"{self.settings.UPSTREAM_BASE_URL}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.UPSTREAM_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.UPSTREAM_API_KEY}"
        return url, headers, payload

    def extract_output_text(self, response_json: dict) -> str:
        try:
            return response_json["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def apply_redacted_text(self, response_json: dict, redacted_text: str) -> dict:
        out = json.loads(json.dumps(response_json))  # cheap deep copy
        try:
            out["choices"][0]["message"]["content"] = redacted_text
        except (KeyError, IndexError, TypeError):
            pass
        return out

    def parse_stream_line(self, line: str) -> Optional[StreamChunk]:
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return StreamChunk(text_delta="", is_control=True, is_done=True)
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None
        try:
            choice = obj["choices"][0]
            delta_text = choice.get("delta", {}).get("content") or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError):
            delta_text, finish_reason = "", None
        return StreamChunk(
            text_delta=delta_text,
            is_control=not delta_text,
            is_done=False,
            finish_reason=finish_reason,
        )

    def encode_stream_chunk(self, text: str, *, finish_reason: Optional[str] = None) -> bytes:
        obj = {
            "id": f"shield-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(obj)}\n\n".encode()


# --------------------------------------------------------------------- #
# Anthropic adapter — translates OpenAI-shaped requests/responses to/from
# Anthropic's native Messages API.
# --------------------------------------------------------------------- #
class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"

    def build_request(self, payload: dict) -> tuple[str, dict, dict]:
        url = f"{self.settings.UPSTREAM_BASE_URL}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.settings.UPSTREAM_API_KEY,
            "anthropic-version": self.settings.UPSTREAM_ANTHROPIC_VERSION,
        }

        system_parts = []
        messages = []
        for m in payload.get("messages", []):
            if m.get("role") == "system":
                system_parts.append(m.get("content", ""))
            else:
                role = m.get("role", "user")
                # Anthropic only knows "user" / "assistant".
                role = "assistant" if role == "assistant" else "user"
                messages.append({"role": role, "content": m.get("content", "")})

        body: dict[str, Any] = {
            "model": payload.get("model", "claude-sonnet-4-6"),
            "messages": messages,
            "max_tokens": payload.get("max_tokens") or 1024,
            "stream": bool(payload.get("stream", False)),
        }
        if system_parts:
            body["system"] = "\n\n".join(p for p in system_parts if p)
        if "temperature" in payload:
            body["temperature"] = payload["temperature"]
        if "top_p" in payload:
            body["top_p"] = payload["top_p"]
        if "stop" in payload:
            body["stop_sequences"] = payload["stop"] if isinstance(payload["stop"], list) else [payload["stop"]]

        return url, headers, body

    def extract_output_text(self, response_json: dict) -> str:
        blocks = response_json.get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    def apply_redacted_text(self, response_json: dict, redacted_text: str) -> dict:
        out = json.loads(json.dumps(response_json))
        out["content"] = [{"type": "text", "text": redacted_text}]
        return out

    def to_openai_response(self, response_json: dict, text_override: Optional[str] = None) -> dict:
        """Convert a completed Anthropic response into the OpenAI chat
        completion shape so clients get a consistent envelope regardless of
        upstream provider."""
        text = text_override if text_override is not None else self.extract_output_text(response_json)
        stop_map = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}
        usage = response_json.get("usage", {})
        return {
            "id": response_json.get("id", f"shield-{uuid.uuid4().hex[:12]}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response_json.get("model", ""),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": stop_map.get(response_json.get("stop_reason"), "stop"),
            }],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        }

    def parse_stream_line(self, line: str) -> Optional[StreamChunk]:
        line = line.strip()
        if not line.startswith("data:"):
            return None
        data = line[len("data:"):].strip()
        if not data:
            return None
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None

        event_type = obj.get("type")
        if event_type == "content_block_delta":
            delta = obj.get("delta", {})
            text = delta.get("text", "") if delta.get("type") == "text_delta" else ""
            return StreamChunk(text_delta=text, is_control=not text, is_done=False)
        if event_type == "message_delta":
            stop_reason = obj.get("delta", {}).get("stop_reason")
            return StreamChunk(text_delta="", is_control=True, is_done=False, finish_reason=stop_reason)
        if event_type == "message_stop":
            return StreamChunk(text_delta="", is_control=True, is_done=True)
        # message_start / content_block_start / content_block_stop / ping — no visible text.
        return StreamChunk(text_delta="", is_control=True, is_done=False)

    def encode_stream_chunk(self, text: str, *, finish_reason: Optional[str] = None) -> bytes:
        stop_map = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}
        obj = {
            "id": f"shield-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "delta": {"content": text} if text else {},
                "finish_reason": stop_map.get(finish_reason) if finish_reason else None,
            }],
        }
        return f"data: {json.dumps(obj)}\n\n".encode()


def get_adapter(settings: Settings) -> ProviderAdapter:
    if settings.UPSTREAM_PROVIDER == "anthropic":
        return AnthropicAdapter(settings)
    return OpenAICompatibleAdapter(settings)


# --------------------------------------------------------------------- #
# HTTP client factory
# --------------------------------------------------------------------- #
def build_http_client(settings: Settings) -> httpx.AsyncClient:
    timeout = httpx.Timeout(
        settings.UPSTREAM_TIMEOUT_SECONDS, connect=settings.CONNECT_TIMEOUT_SECONDS
    )
    return httpx.AsyncClient(timeout=timeout)


async def forward_non_stream(client: httpx.AsyncClient, adapter: ProviderAdapter, payload: dict) -> dict:
    url, headers, body = adapter.build_request(payload)
    try:
        resp = await client.post(url, headers=headers, json=body)
    except httpx.RequestError as exc:
        raise UpstreamError(502, f"Upstream request failed: {exc}") from exc

    if resp.status_code >= 400:
        raise UpstreamError(resp.status_code, f"Upstream returned {resp.status_code}: {resp.text[:500]}")

    return resp.json()


async def forward_stream(
    client: httpx.AsyncClient, adapter: ProviderAdapter, payload: dict
) -> AsyncIterator[StreamChunk]:
    url, headers, body = adapter.build_request(payload)
    try:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                raise UpstreamError(resp.status_code, f"Upstream returned {resp.status_code}: {error_body[:500]}")
            async for raw_line in resp.aiter_lines():
                chunk = adapter.parse_stream_line(raw_line)
                if chunk is not None:
                    yield chunk
    except httpx.RequestError as exc:
        raise UpstreamError(502, f"Upstream streaming request failed: {exc}") from exc
