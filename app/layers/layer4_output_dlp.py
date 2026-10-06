"""
Layer 4 — Output guardrails: canary leak verification + secret/PII DLP on
model responses (target: < 5ms per chunk).

Two consumption modes:

  * `scan_complete_output()` — for non-streaming responses, run once against
    the full assistant message before it is returned to the client.
  * `StreamingDLPBuffer` — for SSE streaming, feed each text delta in as it
    arrives. The buffer holds back a small trailing window of characters so
    a canary token or secret that straddles a chunk boundary is still
    caught (see `Settings.STREAMING_DLP_BUFFER_CHARS`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.core.canary import Canary, check_leak
from app.layers.layer1_deterministic import Finding, redact_pii, redact_secrets


@dataclass
class OutputScanResult:
    blocked: bool
    canary_leak: bool
    sanitized_text: str
    findings: List[Finding] = field(default_factory=list)
    block_reason: Optional[str] = None


def scan_complete_output(text: str, canary: Optional[Canary], *, enable_secret_scan: bool = True) -> OutputScanResult:
    if canary and check_leak(text, canary):
        return OutputScanResult(
            blocked=True,
            canary_leak=True,
            sanitized_text="",
            block_reason="canary_token_leak_detected",
        )

    findings: List[Finding] = []
    sanitized = text
    if enable_secret_scan:
        sanitized, secret_findings = redact_secrets(sanitized)
        sanitized, pii_findings = redact_pii(sanitized)
        findings = secret_findings + pii_findings

    return OutputScanResult(blocked=False, canary_leak=False, sanitized_text=sanitized, findings=findings)


class StreamingDLPBuffer:
    """Incremental scanner for SSE token/text deltas.

    Two modes, controlled by `hold_back` (wired from
    `Settings.STRICT_STREAMING_DLP`):

      * `hold_back=True` (strict, default) — text is held in a small
        trailing buffer and only released once we're confident a
        canary/secret match isn't straddling the boundary with the next
        chunk. Guarantees a match is never forwarded to the client, at the
        cost of `buffer_chars` worth of added latency on the tail of the
        response.
      * `hold_back=False` (fast) — text is forwarded immediately as it
        arrives, with a rolling window kept *only* for detection. If a
        match is found, the stream is aborted immediately, but any text
        already flushed before the match completed cannot be recalled.

    Usage:
        buf = StreamingDLPBuffer(canary=canary, buffer_chars=64)
        for delta in upstream_deltas:
            release, blocked, reason = buf.feed(delta)
            if blocked:
                break  # abort the stream, do not forward `release`
            if release:
                yield release  # safe to forward to the client
        tail, blocked, reason = buf.flush()
    """

    def __init__(
        self,
        canary: Optional[Canary],
        *,
        buffer_chars: int = 64,
        enable_secret_scan: bool = True,
        hold_back: bool = True,
    ):
        self._canary = canary
        self._buffer_chars = max(buffer_chars, 32)
        self._enable_secret_scan = enable_secret_scan
        self._hold_back = hold_back
        self._pending = ""   # used when hold_back=True
        self._window = ""    # rolling detection-only window when hold_back=False
        self.blocked = False
        self.block_reason: Optional[str] = None

    def _scan(self, text: str) -> Optional[str]:
        if self._canary and check_leak(text, self._canary):
            return "canary_token_leak_detected"
        if self._enable_secret_scan:
            # A cheap presence check is enough here; full redaction happens
            # on whatever we decide to release below.
            _, findings = redact_secrets(text)
            if findings:
                return "secret_detected_in_output_stream"
        return None

    def _redact(self, text: str) -> str:
        if not self._enable_secret_scan or not text:
            return text
        text, _ = redact_secrets(text)
        text, _ = redact_pii(text)
        return text

    def feed(self, delta: str) -> tuple[str, bool, Optional[str]]:
        """Feed one chunk of streamed text. Returns (safe_text_to_release, blocked, reason)."""
        if self.blocked:
            return "", True, self.block_reason

        if not self._hold_back:
            self._window = (self._window + delta)[-(self._buffer_chars * 2):]
            reason = self._scan(self._window)
            if reason:
                self.blocked = True
                self.block_reason = reason
                return "", True, reason
            return self._redact(delta), False, None

        self._pending += delta
        reason = self._scan(self._pending)
        if reason:
            self.blocked = True
            self.block_reason = reason
            return "", True, reason

        # Keep the trailing `buffer_chars` in reserve so a match split
        # across the *next* chunk boundary is still caught, and release
        # the rest.
        if len(self._pending) <= self._buffer_chars:
            return "", False, None

        release_len = len(self._pending) - self._buffer_chars
        to_release, self._pending = self._pending[:release_len], self._pending[release_len:]
        return self._redact(to_release), False, None

    def flush(self) -> tuple[str, bool, Optional[str]]:
        """Force-release the current pending buffer after a final scan.
        Call this whenever you need to guarantee ordering against a
        non-text control frame (e.g. a finish_reason event) as well as
        once the upstream stream has fully ended."""
        if self.blocked:
            return "", True, self.block_reason

        if not self._hold_back:
            # Nothing is held back in fast mode — everything was already
            # released (and scanned) as it arrived.
            return "", False, None

        reason = self._scan(self._pending)
        if reason:
            self.blocked = True
            self.block_reason = reason
            return "", True, reason

        remainder = self._pending
        self._pending = ""
        return self._redact(remainder), False, None
