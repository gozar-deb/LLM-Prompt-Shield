"""
Anti-evasion normalization.

Attackers evade naive keyword/regex filters by hiding instructions inside
HTML comments, using look-alike Unicode characters (homoglyphs), or
Base64/URL-encoding the payload. This module produces a *normalized* view of
the text for scanning purposes. The normalized text is never sent upstream
verbatim — it exists only so Layer 1 / Layer 2 can see through common
obfuscation tricks; the original, unmodified user text (with secrets/PII
redacted) is what actually gets forwarded to the LLM.
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

import ftfy

# --------------------------------------------------------------------- #
# Homoglyph table: common Cyrillic / Greek / fullwidth look-alikes mapped
# to their ASCII Latin equivalents. Not exhaustive — extend as new evasion
# patterns are observed in production traffic.
# --------------------------------------------------------------------- #
_HOMOGLYPH_MAP = {
    # Cyrillic -> Latin
    "а": "a", "А": "A",
    "е": "e", "Е": "E",
    "о": "o", "О": "O",
    "р": "p", "Р": "P",
    "с": "c", "С": "C",
    "х": "x", "Х": "X",
    "у": "y", "У": "Y",
    "і": "i", "І": "I",
    "ѕ": "s", "Ѕ": "S",
    "ј": "j", "Ј": "J",
    "ԁ": "d",
    "ⅰ": "i",
    # Greek -> Latin
    "α": "a", "Α": "A",
    "ο": "o", "Ο": "O",
    "ρ": "p", "Ρ": "P",
    "ν": "v", "Ν": "N",
    "ι": "i", "Ι": "I",
    # Fullwidth ASCII variants (U+FF01 - U+FF5E) collapse via NFKC below,
    # but a few common ones are listed explicitly for clarity/coverage.
    "ｉ": "i", "ｇ": "g", "ｎ": "n", "ｏ": "o", "ｒ": "r", "ｅ": "e",
}
_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_MAP)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\uFEFF]")

# Base64 candidate: 20+ chars of base64 alphabet, optionally padded.
_BASE64_CANDIDATE_RE = re.compile(r"(?:[A-Za-z0-9+/]{20,}={0,2})")
_URLENC_CANDIDATE_RE = re.compile(r"(?:%[0-9A-Fa-f]{2}){4,}")


@dataclass
class NormalizationResult:
    normalized_text: str
    decoded_variants: List[str] = field(default_factory=list)
    evasion_signals: List[str] = field(default_factory=list)


def strip_html(text: str) -> str:
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    return text


def strip_zero_width(text: str) -> str:
    return _ZERO_WIDTH_RE.sub("", text)


def normalize_homoglyphs(text: str) -> str:
    # NFKC first collapses fullwidth / compatibility forms; the explicit
    # table then catches common Cyrillic/Greek look-alikes NFKC won't touch.
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_HOMOGLYPH_TABLE)


def _try_base64_decode(candidate: str) -> str | None:
    if len(candidate) < 20:
        return None
    try:
        decoded = base64.b64decode(candidate, validate=True)
        text = decoded.decode("utf-8")
        # Reject decodes that are mostly non-printable garbage — a real
        # base64-smuggled instruction should look like readable text.
        printable_ratio = sum(c.isprintable() for c in text) / max(len(text), 1)
        if printable_ratio > 0.85 and len(text) >= 4:
            return text
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    return None


def _try_url_decode(text: str) -> str | None:
    from urllib.parse import unquote

    decoded = unquote(text)
    return decoded if decoded != text else None


def decode_embedded_encodings(text: str) -> tuple[List[str], List[str]]:
    """Find Base64/URL-encoded substrings and decode them so their content
    can be re-scanned by Layer 1 / Layer 2. Returns (decoded_variants, signals)."""
    variants: List[str] = []
    signals: List[str] = []

    for match in _BASE64_CANDIDATE_RE.finditer(text):
        decoded = _try_base64_decode(match.group(0))
        if decoded:
            variants.append(decoded)
            signals.append(f"base64_payload:{match.group(0)[:16]}...")

    for match in _URLENC_CANDIDATE_RE.finditer(text):
        decoded = _try_url_decode(match.group(0))
        if decoded:
            variants.append(decoded)
            signals.append(f"url_encoded_payload:{match.group(0)[:16]}...")

    return variants, signals


def normalize(text: str, *, decode_evasions: bool = True) -> NormalizationResult:
    """Full anti-evasion normalization pipeline.

    Returns the cleaned text plus any decoded sub-payloads that should also
    be scanned, and a list of human-readable evasion signals for audit logs.
    """
    fixed = ftfy.fix_text(text)
    fixed = strip_zero_width(fixed)
    no_html = strip_html(fixed)
    ascii_normalized = normalize_homoglyphs(no_html)
    collapsed_ws = re.sub(r"[ \t]{2,}", " ", ascii_normalized)

    decoded_variants: List[str] = []
    signals: List[str] = []
    if decode_evasions:
        decoded_variants, signals = decode_embedded_encodings(text)

    return NormalizationResult(
        normalized_text=collapsed_ws,
        decoded_variants=decoded_variants,
        evasion_signals=signals,
    )
