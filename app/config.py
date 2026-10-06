"""
Central configuration for the LLM Prompt Shield.

All values are overridable via environment variables or a `.env` file in the
project root (see `.env.example`). Nothing here should be hardcoded at
deploy time — this module is the single source of truth for tunables so the
rest of the codebase never reads `os.environ` directly.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- #
    # Service identity
    # ---------------------------------------------------------------- #
    SERVICE_NAME: str = "llm-prompt-shield"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    # ---------------------------------------------------------------- #
    # Client-facing auth (protects the proxy itself)
    # ---------------------------------------------------------------- #
    # Comma separated list of API keys that CLIENTS must present to the
    # shield via `Authorization: Bearer <key>`. Keep this separate from the
    # upstream provider key, which clients never see.
    PROXY_API_KEYS: str = ""
    REQUIRE_PROXY_AUTH: bool = True

    @property
    def proxy_api_key_set(self) -> set:
        return {k.strip() for k in self.PROXY_API_KEYS.split(",") if k.strip()}

    # ---------------------------------------------------------------- #
    # Upstream provider
    # ---------------------------------------------------------------- #
    UPSTREAM_PROVIDER: Literal["openai", "anthropic", "openai_compatible"] = "openai"
    UPSTREAM_BASE_URL: str = "https://api.openai.com"
    UPSTREAM_API_KEY: str = ""
    UPSTREAM_TIMEOUT_SECONDS: float = 60.0
    UPSTREAM_ANTHROPIC_VERSION: str = "2023-06-01"  # required header for Anthropic API

    # ---------------------------------------------------------------- #
    # Layer 1 — deterministic scanner
    # ---------------------------------------------------------------- #
    ENABLE_SECRET_REDACTION: bool = True
    ENABLE_PII_REDACTION: bool = True
    ENABLE_ANTI_EVASION_DECODING: bool = True
    BLOCK_ON_PII: bool = False  # if False, PII is masked but the request proceeds
    BLOCK_ON_SECRET: bool = False  # if False, secrets are redacted but request proceeds

    # ---------------------------------------------------------------- #
    # Layer 2 — ML / heuristic injection classifier
    # ---------------------------------------------------------------- #
    INJECTION_SCORE_THRESHOLD: float = Field(default=0.70, ge=0.0, le=1.0)
    ONNX_MODEL_PATH: str = "app/models/prompt_guard_quant.onnx"
    ONNX_TOKENIZER_PATH: str = "app/models/tokenizer"
    CLASSIFIER_BACKEND: Literal["auto", "onnx", "heuristic"] = "auto"
    # "auto" uses ONNX if the model file + onnxruntime + tokenizer are all
    # available, and transparently falls back to the heuristic scorer
    # otherwise, so the proxy is usable out of the box.

    # ---------------------------------------------------------------- #
    # Layer 3 — routing / resilience
    # ---------------------------------------------------------------- #
    MAX_RETRIES: int = 1
    CONNECT_TIMEOUT_SECONDS: float = 5.0

    # ---------------------------------------------------------------- #
    # Layer 4 — output DLP / canary
    # ---------------------------------------------------------------- #
    ENABLE_CANARY: bool = True
    ENABLE_OUTPUT_SECRET_SCAN: bool = True
    # Buffered streaming trades a small amount of added latency for a hard
    # guarantee that canary/secret matches are never sent to the client,
    # even if the match straddles a chunk boundary. When disabled, the
    # shield still aborts the stream the moment a match is found, but text
    # already flushed to the client cannot be recalled.
    STRICT_STREAMING_DLP: bool = True
    STREAMING_DLP_BUFFER_CHARS: int = 64

    # ---------------------------------------------------------------- #
    # Rate limiting (in-memory token bucket; swap for Redis at scale — see SETUP.md)
    # ---------------------------------------------------------------- #
    ENABLE_RATE_LIMIT: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 20

    # ---------------------------------------------------------------- #
    # Audit logging
    # ---------------------------------------------------------------- #
    AUDIT_LOG_PATH: Optional[str] = "logs/audit.jsonl"
    AUDIT_LOG_PROMPT_PREVIEW_CHARS: int = 120

    # ---------------------------------------------------------------- #
    # CORS
    # ---------------------------------------------------------------- #
    CORS_ALLOW_ORIGINS: str = "*"

    @field_validator("UPSTREAM_BASE_URL")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Settings are read once and cached; restart the process to pick up
    changed environment variables (standard 12-factor behaviour)."""
    return Settings()
