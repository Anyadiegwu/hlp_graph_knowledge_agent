"""
Client-side x402 payment helper.

When a remote MCP tool call comes back as a structured X402Challenge
(instead of its normal result — see mcp_server/paywall.py), this module
signs an EIP-3009 authorization with the agent's own wallet and retries
the call with `x_payment` populated. This is the agent's own
"wallet authorization-retry" sequence.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from x402_core.exceptions import X402PaymentRequiredError
from x402_core.schemas import X402Challenge, X402PaymentPayload
from x402_core.wallet import get_default_wallet

logger = logging.getLogger("agent_client.x402_client")


def parse_challenge(result: Any) -> X402Challenge | None:
    """Returns a parsed X402Challenge if `result` looks like one, else None.
    Tool results are otherwise free-form text/markdown, so we only treat a
    string as a challenge if it validates against the exact schema —
    that avoids false-positiving on a normal answer that merely happens
    to contain the word 'payment'."""
    if not isinstance(result, str):
        return None
    stripped = result.strip()
    if not stripped.startswith("{"):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if data.get("error") != "payment_required" or "accepts" not in data:
        return None
    try:
        return X402Challenge.model_validate(data)
    except Exception:
        return None


async def pay_and_build_argument(challenge: X402Challenge) -> str:
    """Signs an EIP-3009 authorization satisfying the first acceptable
    requirement in the challenge and returns the JSON string to pass as
    the tool's `x_payment` argument on retry.

    Raises X402PaymentRequiredError (re-wrapped, carrying the original
    challenge) if no wallet is configured, so the caller can surface a
    clean "payment unavailable" message instead of a wallet stack trace.
    """
    if not challenge.accepts:
        raise X402PaymentRequiredError(challenge.to_json(), "Challenge carried no acceptable payment methods.")

    requirements = challenge.accepts[0]
    wallet = get_default_wallet()
    if wallet is None:
        raise X402PaymentRequiredError(
            challenge.to_json(),
            "No AGENT_WALLET_PRIVATE_KEY configured — cannot pay for "
            f"'{requirements.resource}'. Fund a Base Sepolia testnet wallet "
            "via https://faucet.circle.com/ and set the env var.",
        )

    auth = wallet.sign_transfer_authorization(
        to_address=requirements.pay_to,
        value_atomic=requirements.max_amount_required,
        contract_address=requirements.asset,
    )
    payment = X402PaymentPayload(payload=auth, resource=requirements.resource)
    logger.info(
        "Signed EIP-3009 authorization: payer=%s resource=%s amount_atomic=%s",
        auth.authorization.from_address, requirements.resource, requirements.max_amount_required,
    )
    return payment.to_json()


def _extract_text(result: Any) -> str:
    """MCP tool results commonly come back as a list of content blocks
    rather than a plain string — normalize before challenge-detection."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
    return str(result)


async def invoke_with_payment_retry(remote_tool: Any, arguments: dict[str, Any]) -> str:
    result = await remote_tool.ainvoke(arguments)
    print("DEBUG initial result:", repr(result)[:500])
    challenge = parse_challenge(_extract_text(result))
    if challenge is None:
        print("DEBUG: no challenge detected, returning initial result as-is")
        return result

    print("DEBUG: challenge detected, signing payment...")
    x_payment = await pay_and_build_argument(challenge)
    retried = await remote_tool.ainvoke({**arguments, "x_payment": x_payment})
    print("DEBUG retried result:", repr(retried)[:500])

    if parse_challenge(_extract_text(retried)) is not None:
        print("DEBUG: still a challenge after payment — declined")
        return (
            "Error: payment was presented but the server still declined the "
            "request (verification or settlement issue). Raw response: " + str(retried)[:300]
        )
    print("DEBUG: payment accepted, returning real result")
    return retried