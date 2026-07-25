from __future__ import annotations

from typing import Any


class X402PaymentRequiredError(Exception):
    """Raised (server-side) or on the reciprocal client-side sampling wall
    when a caller has not presented a valid EIP-3009 payment authorization.

    Carries the structured challenge so callers up the stack can surface it
    verbatim rather than re-deriving pricing information.
    """

    def __init__(self, challenge_json: str, message: str = "Payment required") -> None:
        super().__init__(message)
        self.challenge_json = challenge_json


class X402VerificationError(Exception):
    """Raised when a presented payment payload fails facilitator
    verification (bad signature, expired window, wrong asset/network)."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"x402 payment verification failed: {reason}")
        self.reason = reason


class X402SettlementError(Exception):
    """Raised when verification passed but on-chain settlement relay
    through the Xpay facilitator failed or timed out."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"x402 settlement failed: {reason}")
        self.reason = reason


class FinOpsBudgetExceededException(Exception):
    """Raised by the FinOps governor when a session's accumulated or
    projected spend crosses its configured ceiling. Callers should catch
    this specifically and fall back to a cached/structurally-safe response
    rather than letting the exception propagate as a generic failure —
    this is a governance stop, not an error state.
    """

    def __init__(self, session_id: str, spent_usd: float, ceiling_usd: float, detail: dict[str, Any] | None = None) -> None:
        super().__init__(
            f"FinOps budget exceeded for session {session_id}: "
            f"${spent_usd:.4f} spent/projected against ${ceiling_usd:.4f} ceiling."
        )
        self.session_id = session_id
        self.spent_usd = spent_usd
        self.ceiling_usd = ceiling_usd
        self.detail = detail or {}
