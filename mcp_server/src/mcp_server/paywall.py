"""
Server-Side MCP x402 Paywall.

MCP tool calls don't carry raw HTTP headers, so there's no literal
`X-PAYMENT` header to intercept the way a REST framework would. Instead,
`@require_payment` adds a `x_payment: str = ""` keyword argument to the
wrapped tool — the client populates it with a JSON-encoded
`X402PaymentPayload` (see x402_core.schemas) once it has signed one. This
is the direct MCP-transport analogue of the HTTP 402 flow: same
challenge/verify/settle lifecycle, just carried as a tool argument instead
of a header.

Flow per call:
  1. No `x_payment` presented  -> return a structured X402Challenge (JSON)
     in place of the tool's real result. This IS the 402.
  2. `x_payment` presented     -> verify with the Xpay facilitator first
     (cheap, no chain write). Invalid -> raise X402VerificationError.
  3. Verified                  -> run the wrapped tool for real.
  4. Tool succeeded             -> settle (relay on-chain via Xpay,
     gas-sponsored) and append a structured entry to the FinOps audit
     trail (Redis list `hlp:finops:audit`, read by the dashboard and by
     `generate_finops_audit.py`).
"""
from __future__ import annotations

import functools
import json
import logging
import os
import time
from typing import Any, Callable
import inspect

from x402_core.audit import append_audit_entry
from x402_core.constants import usd_to_atomic
from x402_core.exceptions import X402VerificationError
from x402_core.facilitator import get_facilitator_client
from x402_core.schemas import PaymentRequirements, X402Challenge, X402PaymentPayload

logger = logging.getLogger("mcp_server.paywall")


def _build_requirements(resource: str, price_usd: float) -> PaymentRequirements:
    pay_to = os.getenv("SERVER_WALLET_ADDRESS", "")
    if not pay_to:
        logger.warning(
            "SERVER_WALLET_ADDRESS not set — paywalled tool '%s' will issue "
            "challenges but can never be settled. Set it in .env.", resource,
        )
    return PaymentRequirements(
        max_amount_required=usd_to_atomic(price_usd),
        pay_to=pay_to,
        resource=resource,
        description=f"Access to '{resource}' — {price_usd:.4f} USDC per call.",
    )


def require_payment(resource: str, price_usd: float) -> Callable:
    """Decorator for an MCP `@mcp.tool()` async function. Must be applied
    *below* `@mcp.tool()` (i.e. closer to the function) so FastMCP still
    sees the final signature including `x_payment`."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, x_payment: str = "", **kwargs) -> str:
            requirements = _build_requirements(resource, price_usd)

            if not x_payment:
                challenge = X402Challenge(accepts=[requirements])
                append_audit_entry(
                    "challenge_issued",
                    resource=resource,
                    price_usd=price_usd,
                    challenge=challenge.model_dump(),
                )
                return challenge.to_json()

            try:
                payment = X402PaymentPayload.model_validate_json(x_payment)
            except Exception as exc:
                raise X402VerificationError(f"malformed x_payment payload: {exc}") from exc

            facilitator = get_facilitator_client()
            verify_result = await facilitator.verify(payment, requirements)
            append_audit_entry(
                "verify_attempted",
                resource=resource,
                payer=payment.payload.authorization.from_address,
                is_valid=verify_result.is_valid,
                invalid_reason=verify_result.invalid_reason,
            )
            if not verify_result.is_valid:
                raise X402VerificationError(verify_result.invalid_reason or "signature/window invalid")

            # Payment verified — run the actual gated logic.
            result = await func(*args, **kwargs)

            settlement = await facilitator.settle(payment, requirements)
            append_audit_entry(
                "settlement_recorded",
                resource=resource,
                payer=payment.payload.authorization.from_address,
                amount_atomic=requirements.max_amount_required,
                success=settlement.success,
                transaction_hash=settlement.transaction_hash,
                error=settlement.error,
                direction="spent",
            )
            if not settlement.success:
                logger.error(
                    "Settlement failed for resource=%s payer=%s: %s — "
                    "result was already computed and IS returned; a failed "
                    "settlement here is a billing gap, not a correctness bug, "
                    "and should be reconciled out-of-band.",
                    resource, payment.payload.authorization.from_address, settlement.error,
                )

            return result

        original_sig = inspect.signature(func)
        new_params = list(original_sig.parameters.values()) + [
            inspect.Parameter(
                "x_payment",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default="",
                annotation=str,
            )
        ]
        wrapper.__signature__ = original_sig.replace(parameters=new_params)
        wrapper.__annotations__ = {**func.__annotations__, "x_payment": str}
        return wrapper

    return decorator
