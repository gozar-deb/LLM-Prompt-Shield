"""
Structured audit logging.

Every request produces one JSON line summarizing what the pipeline decided
and why — enough to investigate an incident or tune thresholds, without
ever persisting raw secrets, PII, or full prompt/response bodies. Only a
short, length-capped preview of the (already-redacted) prompt is stored.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger("prompt_shield.audit")


@dataclass
class AuditEvent:
    request_id: str
    timestamp: float
    client_key_fingerprint: Optional[str]
    decision: str  # "allowed" | "blocked_injection" | "blocked_canary_leak" | "blocked_rate_limit" | "error"
    layer2_score: Optional[float] = None
    injection_heuristic_hit: bool = False
    secret_findings: list = field(default_factory=list)
    pii_findings: list = field(default_factory=list)
    evasion_signals: list = field(default_factory=list)
    prompt_preview: str = ""
    latency_ms: float = 0.0
    upstream_provider: str = ""
    reason: Optional[str] = None


class AuditLogger:
    def __init__(self, path: Optional[str]):
        self._path = path
        if self._path:
            directory = os.path.dirname(self._path)
            if directory:
                os.makedirs(directory, exist_ok=True)

    def log(self, event: AuditEvent) -> None:
        line = json.dumps(asdict(event), default=str)
        logger.info(line)
        if self._path:
            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                logger.exception("Failed to write audit log to %s", self._path)


def fingerprint_key(key: str) -> str:
    """A short, non-reversible identifier for a proxy API key, safe to log."""
    import hashlib

    if not key:
        return "anonymous"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def make_prompt_preview(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
