"""
In-memory token-bucket rate limiter, keyed per client (proxy API key, or
source IP when auth is disabled).

This is intentionally simple and process-local — fine for a single
instance or for smoke-testing the pipeline. Once you run more than one
replica behind a load balancer, buckets won't be shared across processes
and effective limits become `N_replicas * configured_limit`. Swap this
module for a Redis-backed bucket (e.g. `redis` + a Lua script, or a package
like `limits`) before scaling horizontally — see SETUP.md.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    def __init__(self, requests_per_minute: int, burst: int):
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self.capacity = max(burst, 1)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity - 1, last_refill=now)
                self._buckets[key] = bucket
                return True, 0.0

            elapsed = now - bucket.last_refill
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_rate)
            bucket.last_refill = now

            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return True, 0.0

            retry_after = (1 - bucket.tokens) / self.refill_rate
            return False, retry_after

    def reset(self) -> None:
        """Test helper."""
        with self._lock:
            self._buckets.clear()
