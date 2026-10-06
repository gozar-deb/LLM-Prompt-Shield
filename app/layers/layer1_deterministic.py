"""
Layer 1 — Deterministic scanner (target: < 2ms).

Three independent jobs, all regex/rule based so latency stays predictable:
  1. Secret detection & redaction (API keys, AWS credentials, private keys...)
  2. PII detection & redaction (SSN, credit card w/ Luhn check, email)
  3. Prompt-injection heuristics (keyword/phrase patterns)

This layer runs on the anti-evasion-normalized text (see core/normalizer.py)
so obfuscated variants are caught too, but redaction is applied to the
*original* text before it is forwarded upstream, so legitimate content isn't
mangled by normalization.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from app.core.normalizer import normalize

# --------------------------------------------------------------------- #
# Secret patterns
# --------------------------------------------------------------------- #
SECRET_PATTERNS: dict[str, re.Pattern] = {
    "aws_access_key_id": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "aws_secret_access_key": re.compile(
        r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"
    ),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}(?:T3BlbkFJ[A-Za-z0-9]{20,})?\b"),
    "openai_project_key": re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}\b"),
    "anthropic_api_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,48}\b"),
    "stripe_key": re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
    "private_key_block": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        re.DOTALL,
    ),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
    "generic_api_key_assignment": re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|access[_-]?key)\b\s*[:=]\s*"
        r"['\"]?([A-Za-z0-9_\-/+=]{16,})['\"]?"
    ),
}

# --------------------------------------------------------------------- #
# PII patterns
# --------------------------------------------------------------------- #
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_CREDIT_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")


def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# --------------------------------------------------------------------- #
# Prompt-injection heuristic phrases. These are intentionally broad; Layer 2
# (ML classifier) is the primary injection defense — this list exists as a
# fast, zero-latency first pass and as the signal source for the heuristic
# classifier fallback when no ONNX model is loaded.
# --------------------------------------------------------------------- #
INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)ignore (?:all |any )?(?:previous|prior|above|the) instructions?"),
    re.compile(r"(?i)disregard (?:all |any )?(?:previous|prior|above) (?:instructions?|rules?)"),
    re.compile(r"(?i)forget (?:all |any )?(?:previous|prior|your) instructions?"),
    re.compile(r"(?i)system\s*(?:prompt\s*)?override"),
    re.compile(r"(?i)\bDAN\s*mode\b"),
    re.compile(r"(?i)\bdeveloper\s*mode\b"),
    re.compile(r"(?i)\bjailbreak(?:s|ing|ed)?\b"),
    re.compile(r"(?i)do anything now"),
    re.compile(r"(?i)you are now (?:in |acting as )?(?:a |an )?(?:unrestricted|unfiltered|uncensored)"),
    re.compile(r"(?i)pretend (?:you|to) (?:have|be) no (?:restrictions|filters|rules|limits)"),
    re.compile(r"(?i)act as if you (?:have no|had no|are not)"),
    re.compile(r"(?i)(?:reveal|print|show|output|repeat|leak) (?:the |your )?system prompt"),
    re.compile(r"(?i)(?:reveal|print|show|output) (?:your )?(?:hidden |secret )?instructions"),
    re.compile(r"(?i)bypass (?:your |the )?(?:safety|content|security) (?:guidelines?|policy|policies|filters?)"),
    re.compile(r"(?i)\bno (?:ethical|moral) (?:guidelines|constraints|restrictions)\b"),
    re.compile(r"(?i)respond only with|from now on you (?:will|must) (?:always )?respond"),
    re.compile(r"(?i)this is a hypothetical scenario with no rules"),
]


@dataclass
class Finding:
    kind: str  # "secret" | "pii" | "injection_heuristic"
    label: str
    preview: str  # short, safe-to-log preview (never the raw secret)


@dataclass
class Layer1Result:
    sanitized_text: str
    findings: List[Finding] = field(default_factory=list)
    injection_heuristic_hit: bool = False
    evasion_signals: List[str] = field(default_factory=list)

    @property
    def has_secret(self) -> bool:
        return any(f.kind == "secret" for f in self.findings)

    @property
    def has_pii(self) -> bool:
        return any(f.kind == "pii" for f in self.findings)


def redact_secrets(text: str) -> tuple[str, List[Finding]]:
    findings: List[Finding] = []
    for label, pattern in SECRET_PATTERNS.items():
        def _sub(m: re.Match, label=label) -> str:
            findings.append(Finding(kind="secret", label=label, preview=f"{m.group(0)[:6]}…(redacted)"))
            return "[REDACTED_SECRET]"
        text = pattern.sub(_sub, text)
    return text, findings


def redact_pii(text: str) -> tuple[str, List[Finding]]:
    findings: List[Finding] = []

    def _sub_ssn(m: re.Match) -> str:
        findings.append(Finding(kind="pii", label="ssn", preview="***-**-****"))
        return "[REDACTED_SSN]"

    text = _SSN_RE.sub(_sub_ssn, text)

    def _sub_email(m: re.Match) -> str:
        findings.append(Finding(kind="pii", label="email", preview="[REDACTED_EMAIL]"))
        return "[REDACTED_EMAIL]"

    text = _EMAIL_RE.sub(_sub_email, text)

    def _sub_cc(m: re.Match) -> str:
        if _luhn_valid(m.group(0)):
            findings.append(Finding(kind="pii", label="credit_card", preview="[REDACTED_CC]"))
            return "[REDACTED_CREDIT_CARD]"
        return m.group(0)

    text = _CREDIT_CARD_CANDIDATE_RE.sub(_sub_cc, text)
    return text, findings


def detect_injection_heuristics(text: str) -> tuple[bool, List[Finding]]:
    findings: List[Finding] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            findings.append(Finding(kind="injection_heuristic", label=pattern.pattern[:40], preview="matched"))
    return (len(findings) > 0), findings


def run_layer1(text: str, *, decode_evasions: bool = True) -> Layer1Result:
    """Run the full Layer 1 pipeline on one piece of text (typically a
    single message's content). Redaction is applied to the original text;
    injection heuristics run against the anti-evasion-normalized text plus
    any decoded Base64/URL-encoded sub-payloads found inside it."""
    norm = normalize(text, decode_evasions=decode_evasions)

    sanitized, secret_findings = redact_secrets(text)
    sanitized, pii_findings = redact_pii(sanitized)

    scan_targets = [norm.normalized_text, *norm.decoded_variants]
    injection_hit = False
    injection_findings: List[Finding] = []
    for target in scan_targets:
        hit, findings = detect_injection_heuristics(target)
        injection_hit = injection_hit or hit
        injection_findings.extend(findings)

    all_findings = secret_findings + pii_findings + injection_findings
    return Layer1Result(
        sanitized_text=sanitized,
        findings=all_findings,
        injection_heuristic_hit=injection_hit,
        evasion_signals=norm.evasion_signals,
    )
