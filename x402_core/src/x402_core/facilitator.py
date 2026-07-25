"""
Async client for the Xpay Staging Facilitator.

The facilitator brokers two calls per payment:
  POST /verify  — checks the EIP-3009 signature + authorization window are
                  valid, WITHOUT broadcasting anything on-chain yet.
  POST /settle  — relays the authorization on-chain (gas-sponsored) and
                  returns a transaction hash once the transfer lands.

Splitting verify/settle lets the server reject obviously-bad payloads
(expired window, wrong asset) cheaply, before paying for the slower
settlement relay — this matters because `/settle` waits on a
non-deterministic block-finalization window (see REFLECTION_STAGE6.md).
"""
from __future__ import annotations

import logging
import os
import json
import httpx

from x402_core.constants import XPAY_FACILITATOR_URL
from x402_core.schemas import (
    PaymentRequirements,
    SettlementReceipt,
    VerifyResult,
    X402PaymentPayload,
)
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env"))
logger = logging.getLogger("x402_core.facilitator")


class XpayFacilitatorClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: float = 15.0) -> None:
        self.base_url = (base_url or os.getenv("XPAY_FACILITATOR_URL") or XPAY_FACILITATOR_URL).rstrip("/")
        self._timeout = timeout_seconds

    async def verify(
        self,
        payment: X402PaymentPayload,
        requirements: PaymentRequirements,
    ) -> VerifyResult:
        body = {
            "x402Version": payment.x402_version,
            "paymentPayload": payment.model_dump(by_alias=True),
            "paymentRequirements": requirements.model_dump(by_alias=True),
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self.base_url}/verify", json=body)
                print("REQUEST BODY:", json.dumps(body, indent=2))
                print("RESPONSE:", resp.status_code, resp.text)
                resp.raise_for_status()
                data = resp.json()
            return VerifyResult(
                is_valid=bool(data.get("isValid", False)),
                invalid_reason=data.get("invalidReason"),
            )
        except httpx.HTTPError as exc:
            logger.error("Xpay /verify call failed: %s", exc)
            return VerifyResult(is_valid=False, invalid_reason=f"facilitator_unreachable: {exc}")
        
    async def settle(
        self,
        payment: X402PaymentPayload,
        requirements: PaymentRequirements,
    ) -> SettlementReceipt:
        body = {
            "x402Version": payment.x402_version,
            "paymentPayload": payment.model_dump(by_alias=True),
            "paymentRequirements": requirements.model_dump(by_alias=True),
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self.base_url}/settle", json=body)
                print("SETTLE REQUEST BODY:", json.dumps(body, indent=2))
                print("SETTLE RESPONSE:", resp.status_code, resp.text)
                resp.raise_for_status()
                data = resp.json()
                success = bool(data.get("success", False))
            return SettlementReceipt(
                success=success,
                transaction_hash=data.get("transaction") or None,   
                payer=payment.payload.authorization.from_address,
                amount_atomic=requirements.max_amount_required,
                error=None if success else (data.get("errorReason") or f"raw_response: {data}"),
            )
        except httpx.HTTPError as exc:
            logger.error("Xpay /settle call failed: %s", exc)
            return SettlementReceipt(
                success=False,
                payer=payment.payload.authorization.from_address,
                amount_atomic=requirements.max_amount_required,
                error=f"facilitator_unreachable: {exc}",
            )
        
_default_client: XpayFacilitatorClient | None = None


def get_facilitator_client() -> XpayFacilitatorClient:
    global _default_client
    if _default_client is None:
        _default_client = XpayFacilitatorClient()
    return _default_client
