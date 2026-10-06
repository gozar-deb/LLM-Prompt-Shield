"""
Layer 2 — Real-time prompt-injection classifier (target: < 15ms).

Two interchangeable backends, selected by `Settings.CLASSIFIER_BACKEND`:

  * "onnx"      — loads a quantized ONNX sequence-classification model
                  (e.g. Prompt-Guard-2 or a DeBERTa-v3 injection classifier,
                  see scripts/export_onnx_model.py) and runs it on CPU/GPU.
  * "heuristic" — a dependency-free weighted-signal scorer that reuses the
                  Layer 1 injection phrase list plus a handful of structural
                  signals (imperative density, role-play framing, unusual
                  punctuation/formatting). It is deliberately conservative
                  and is NOT a substitute for a trained classifier — it
                  exists so the shield is fully functional out of the box,
                  before anyone has exported a model.

  * "auto" (default) tries ONNX first and falls back to the heuristic
    scorer if the model file, tokenizer, or `onnxruntime` package aren't
    available, logging a warning once at startup either way.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Protocol

from app.config import Settings
from app.layers.layer1_deterministic import INJECTION_PATTERNS

logger = logging.getLogger("prompt_shield.layer2")


class Classifier(Protocol):
    backend_name: str

    def score(self, text: str) -> float:
        """Return a probability in [0.0, 1.0] that `text` is a prompt-injection attempt."""
        ...


# --------------------------------------------------------------------- #
# Heuristic fallback backend
# --------------------------------------------------------------------- #
_ROLEPLAY_RE = re.compile(
    r"(?i)\b(?:you are now|from now on|pretend (?:you|to)|act as (?:if|a)|roleplay as)\b"
)
_IMPERATIVE_RE = re.compile(
    r"(?i)^\s*(?:ignore|disregard|forget|override|reveal|print|show|output|bypass|disable|stop)\b",
    re.MULTILINE,
)
_SUSPICIOUS_PUNCT_RE = re.compile(r"[#*_]{3,}|={3,}|-{5,}")


class HeuristicClassifier:
    backend_name = "heuristic"

    def score(self, text: str) -> float:
        if not text:
            return 0.0

        signal = 0.0
        phrase_hits = sum(1 for p in INJECTION_PATTERNS if p.search(text))
        signal += min(phrase_hits, 3) * 0.35

        if _ROLEPLAY_RE.search(text):
            signal += 0.15
        if _IMPERATIVE_RE.search(text):
            signal += 0.15
        if _SUSPICIOUS_PUNCT_RE.search(text):
            signal += 0.05

        # Long, all-caps shouting is mildly correlated with coercive injection attempts.
        letters = [c for c in text if c.isalpha()]
        if len(letters) > 20 and sum(1 for c in letters if c.isupper()) / len(letters) > 0.6:
            signal += 0.08

        return max(0.0, min(signal, 1.0))


# --------------------------------------------------------------------- #
# ONNX backend
# --------------------------------------------------------------------- #
class OnnxClassifier:
    backend_name = "onnx"

    def __init__(self, model_path: str, tokenizer_path: str):
        import onnxruntime as ort  # local import: optional dependency
        from transformers import AutoTokenizer

        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._input_names = {i.name for i in self._session.get_inputs()}

    def score(self, text: str) -> float:
        import numpy as np

        encoded = self._tokenizer(
            text, return_tensors="np", truncation=True, max_length=512, padding="max_length"
        )
        feed = {k: v for k, v in encoded.items() if k in self._input_names}
        outputs = self._session.run(None, feed)
        logits = outputs[0][0]
        # Softmax over the final dimension; convention: index 1 = "injection" class.
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        return float(probs[1]) if probs.shape[-1] > 1 else float(probs[0])


def _try_load_onnx(settings: Settings) -> Classifier | None:
    model_exists = os.path.isfile(settings.ONNX_MODEL_PATH)
    tokenizer_exists = os.path.isdir(settings.ONNX_TOKENIZER_PATH)
    if not (model_exists and tokenizer_exists):
        logger.warning(
            "ONNX model/tokenizer not found at %s / %s — falling back to heuristic classifier. "
            "Run scripts/export_onnx_model.py to enable the ML backend.",
            settings.ONNX_MODEL_PATH,
            settings.ONNX_TOKENIZER_PATH,
        )
        return None
    try:
        classifier = OnnxClassifier(settings.ONNX_MODEL_PATH, settings.ONNX_TOKENIZER_PATH)
        logger.info("Layer 2: loaded ONNX classifier from %s", settings.ONNX_MODEL_PATH)
        return classifier
    except ImportError as exc:
        logger.warning(
            "onnxruntime/transformers not installed (%s) — falling back to heuristic classifier. "
            "Install requirements-ml.txt to enable the ML backend.",
            exc,
        )
        return None
    except Exception:
        logger.exception("Failed to load ONNX classifier — falling back to heuristic classifier.")
        return None


# Module-level singleton cache. Not keyed via functools.lru_cache because
# pydantic Settings instances aren't hashable by default; `get_settings()`
# already guarantees a single shared Settings instance per process, so a
# plain module-level cache is equivalent and avoids that pitfall entirely.
_classifier_singleton: Classifier | None = None


def get_classifier(settings: Settings) -> Classifier:
    global _classifier_singleton
    if _classifier_singleton is not None:
        return _classifier_singleton

    if settings.CLASSIFIER_BACKEND == "heuristic":
        _classifier_singleton = HeuristicClassifier()
    elif settings.CLASSIFIER_BACKEND == "onnx":
        onnx = _try_load_onnx(settings)
        if onnx is None:
            raise RuntimeError(
                "CLASSIFIER_BACKEND=onnx but the ONNX model could not be loaded. "
                "Check ONNX_MODEL_PATH / ONNX_TOKENIZER_PATH or set CLASSIFIER_BACKEND=auto."
            )
        _classifier_singleton = onnx
    else:  # "auto"
        _classifier_singleton = _try_load_onnx(settings) or HeuristicClassifier()

    return _classifier_singleton


def reset_classifier_cache() -> None:
    """Test helper: force the next get_classifier() call to reload."""
    global _classifier_singleton
    _classifier_singleton = None
