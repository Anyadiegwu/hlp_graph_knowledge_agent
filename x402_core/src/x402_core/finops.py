"""
Algorithmic FinOps Governance — a pre-flight budget interceptor shared by
both the MCP server's tool-execution loop and the agent client's
planning/sampling loop.

Design: every LLM-touching step (a ToT branch, a Reflection iteration, a
planning step) calls `governor.preflight(session_id, projected_usd)`
*before* the LLM call fires. The governor looks up the session's running
spend, and if `spent + projected > ceiling` it raises
`FinOpsBudgetExceededException` — the call site is expected to catch this
and fall back to a cached or structurally-safe response instead of
letting the exception surface as a generic failure.

Spend is tracked in Redis when `REDIS_URL` is configured (so the server
and client processes share one ledger per session), and falls back to a
process-local in-memory dict otherwise (each process then governs only
its own calls — acceptable for local dev, not for the real multi-process
deployment).
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Deque

from x402_core.constants import DEFAULT_MAX_USD_PER_CHAIN
from x402_core.exceptions import FinOpsBudgetExceededException
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env"))

logger = logging.getLogger("x402_core.finops")

# Rough cost model: $ per 1K tokens, used to translate token counts into a
# USD projection for the budget check. This is intentionally simple/linear
# — swap in real per-model pricing if you need tighter accuracy.
DEFAULT_USD_PER_1K_TOKENS = 0.0006


class FinOpsGovernor:
    def __init__(
        self,
        max_usd_per_chain: float | None = None,
        redis_url: str | None = None,
        velocity_window_seconds: float = 60.0,
    ) -> None:
        self.max_usd_per_chain = max_usd_per_chain or float(
            os.getenv("FINOPS_MAX_USD_PER_CHAIN", DEFAULT_MAX_USD_PER_CHAIN)
        )
        self._redis = None
        redis_url = redis_url or os.getenv("REDIS_URL", "")
        if redis_url:
            try:
                import redis as redis_lib
                self._redis = redis_lib.from_url(redis_url)
            except Exception as exc:
                logger.warning("FinOps governor could not connect to Redis (%s) — using in-memory ledger.", exc)
                self._redis = None

        # in-memory fallback: session_id -> accumulated USD
        self._local_ledger: dict[str, float] = {}
        # session_id -> deque[(timestamp, usd)] for velocity curves
        self._velocity_window_seconds = velocity_window_seconds
        self._velocity: dict[str, Deque[tuple[float, float]]] = {}

    # ---- spend bookkeeping -------------------------------------------------

    def _ledger_key(self, session_id: str) -> str:
        return f"hlp:finops:ledger:{session_id}"

    def get_spent(self, session_id: str) -> float:
        if self._redis is not None:
            raw = self._redis.get(self._ledger_key(session_id))
            return float(raw) if raw is not None else 0.0
        return self._local_ledger.get(session_id, 0.0)

    def record_spend(self, session_id: str, usd: float) -> float:
        """Adds `usd` to the session's running total and returns the new total."""
        if self._redis is not None:
            new_total = self._redis.incrbyfloat(self._ledger_key(session_id), usd)
            self._redis.expire(self._ledger_key(session_id), 60 * 60 * 6)
            new_total = float(new_total)
        else:
            new_total = self._local_ledger.get(session_id, 0.0) + usd
            self._local_ledger[session_id] = new_total

        window = self._velocity.setdefault(session_id, deque())
        now = time.time()
        window.append((now, usd))
        cutoff = now - self._velocity_window_seconds
        while window and window[0][0] < cutoff:
            window.popleft()

        return new_total

    def tokens_to_usd(self, token_count: int, usd_per_1k: float = DEFAULT_USD_PER_1K_TOKENS) -> float:
        return (token_count / 1000.0) * usd_per_1k

    def velocity_usd_per_hour(self, session_id: str) -> float:
        window = self._velocity.get(session_id)
        if not window:
            return 0.0
        total = sum(usd for _, usd in window)
        elapsed = max(time.time() - window[0][0], 1e-6)
        return (total / elapsed) * 3600.0

    # ---- the actual governance gate ---------------------------------------

    def preflight(
        self,
        session_id: str,
        projected_usd: float,
        step_label: str = "llm_step",
    ) -> None:
        """Raises FinOpsBudgetExceededException if this step would push the
        session over its ceiling. Callers MUST call this before, not after,
        firing the LLM call it's budgeting for."""
        spent = self.get_spent(session_id)
        if spent + projected_usd > self.max_usd_per_chain:
            raise FinOpsBudgetExceededException(
                session_id=session_id,
                spent_usd=spent,
                ceiling_usd=self.max_usd_per_chain,
                detail={"step": step_label, "projected_usd": projected_usd},
            )

    def preflight_tokens(
        self,
        session_id: str,
        projected_tokens: int,
        step_label: str = "llm_step",
    ) -> float:
        """Convenience wrapper: converts a token estimate to USD, runs the
        preflight check, and returns the projected USD (so the caller can
        pass the same number to record_spend once the call actually
        completes)."""
        projected_usd = self.tokens_to_usd(projected_tokens)
        self.preflight(session_id, projected_usd, step_label)
        return projected_usd


_default_governor: FinOpsGovernor | None = None


def get_governor() -> FinOpsGovernor:
    global _default_governor
    if _default_governor is None:
        _default_governor = FinOpsGovernor()
    return _default_governor
