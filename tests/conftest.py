"""
Shared pytest fixtures.

Environment variables are set *before* `app.main` (and therefore
`app.config.get_settings()`) is ever imported, since `get_settings()` is
process-cached. Every test module should import `app.*` lazily (inside test
functions or via fixtures) rather than at module scope, so this setup always
wins the race.
"""
import os

os.environ.setdefault("PROXY_API_KEYS", "test-key")
os.environ.setdefault("REQUIRE_PROXY_AUTH", "true")
os.environ.setdefault("UPSTREAM_PROVIDER", "openai")
os.environ.setdefault("UPSTREAM_BASE_URL", "https://api.openai.com")
os.environ.setdefault("UPSTREAM_API_KEY", "test-upstream-key")
os.environ.setdefault("ENABLE_RATE_LIMIT", "false")
os.environ.setdefault("ENABLE_CANARY", "true")
os.environ.setdefault("CLASSIFIER_BACKEND", "heuristic")
os.environ.setdefault("INJECTION_SCORE_THRESHOLD", "0.70")
os.environ.setdefault("AUDIT_LOG_PATH", "")

import pytest


@pytest.fixture
def auth_headers() -> dict:
    return {"Authorization": "Bearer test-key"}


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c
