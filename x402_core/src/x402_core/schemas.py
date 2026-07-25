"""
Pydantic models for the x402 challenge / payment / settlement lifecycle.

These mirror the shapes used by the x402 spec's HTTP 402 flow
(https://x402.org — `PaymentRequirements`, `X-PAYMENT` payload, and the
facilitator `/verify` + `/settle` responses), adapted to travel as plain
JSON strings over MCP tool arguments and MCP sampling metadata instead of
raw HTTP headers, since MCP transports don't expose header injection to
tool authors the way a REST framework would.
"""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from x402_core.constants import (
    CAIP2_CHAIN_ID,
    USDC_CONTRACT_ADDRESS,
    X402_NETWORK_ID,
    X402_SCHEME_EXACT,
)
from .constants import EIP712_DOMAIN_NAME, EIP712_DOMAIN_VERSION

class PaymentRequirements(BaseModel):
    scheme: str = X402_SCHEME_EXACT
    network: str = X402_NETWORK_ID
    caip2: str = CAIP2_CHAIN_ID
    max_amount_required: str = Field(..., alias="maxAmountRequired")
    asset: str = USDC_CONTRACT_ADDRESS
    pay_to: str = Field(..., alias="payTo")
    resource: str
    description: str = ""
    mime_type: str = Field(default="application/json", alias="mimeType")
    max_timeout_seconds: int = Field(default=300, alias="maxTimeoutSeconds")
    extra: dict = Field(
        default_factory=lambda: {
            "name": EIP712_DOMAIN_NAME,
            "version": EIP712_DOMAIN_VERSION,
        }
    )

    model_config = {"populate_by_name": True}

class X402Challenge(BaseModel):
    x402_version: int = Field(default=1, alias="x402Version")
    error: str = "payment_required"
    accepts: list[PaymentRequirements]
    issued_at: float = Field(default_factory=time.time)

    model_config = {"populate_by_name": True}

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)

class EIP3009Authorization(BaseModel):
    """The signed message fields for `transferWithAuthorization`."""

    from_address: str = Field(alias="from")
    to_address: str = Field(alias="to")
    value: str
    valid_after: int = Field(alias="validAfter")
    valid_before: int = Field(alias="validBefore")
    nonce: str  # 32-byte hex, unique per authorization

    model_config = {"populate_by_name": True}


class ExactSchemePayload(BaseModel):
    """The x402 'exact' scheme payload shape: a combined 65-byte signature
    alongside the EIP-3009 message it signs."""

    signature: str
    authorization: EIP3009Authorization

    model_config = {"populate_by_name": True}


class X402PaymentPayload(BaseModel):
    x402_version: int = Field(default=1, alias="x402Version")
    scheme: str = X402_SCHEME_EXACT
    network: str = X402_NETWORK_ID
    payload: ExactSchemePayload
    resource: str

    model_config = {"populate_by_name": True}

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)
    
class SettlementReceipt(BaseModel):
    """Result of a verified + settled payment, as returned by the Xpay
    facilitator's `/settle` staging endpoint (or a locally-shaped
    equivalent when running in offline/demo mode)."""

    success: bool
    transaction_hash: str | None = None
    network: str = X402_NETWORK_ID
    payer: str | None = None
    amount_atomic: str | None = None
    settled_at: float = Field(default_factory=time.time)
    error: str | None = None


class VerifyResult(BaseModel):
    """Result of a `/verify` call — checks the signature and authorization
    window are valid *before* the caller wastes an LLM call, without yet
    broadcasting/settling on-chain."""

    is_valid: bool
    invalid_reason: str | None = None
