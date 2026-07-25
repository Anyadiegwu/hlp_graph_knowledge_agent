"""
Generates `finops_compliance_audit.json` — a structural trace of one
paywalled operation end to end: the initial 402 challenge, the signed
payment's metadata, and the final verified settlement return.

Run this against a REAL, funded testnet wallet + a running Xpay staging
facilitator for a genuine deliverable. This script lives inside the
x402_core/ workspace member, so run it from there directly:

    cd x402_core
    uv run python generate_finops_audit.py --resource query_knowledge --price-usd 0.01

Or from the repo root, without cd'ing:

    uv run --package x402_core python x402_core/generate_finops_audit.py \
        --resource query_knowledge --price-usd 0.01

Without network access to the facilitator (e.g. offline dev), pass
--offline to produce a schema-accurate structural example instead —
useful for reviewing the JSON shape, but NOT a substitute for the real
run required by the deliverable checklist.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from x402_core.constants import usd_to_atomic
from x402_core.facilitator import get_facilitator_client
from x402_core.schemas import PaymentRequirements, X402Challenge, X402PaymentPayload
from x402_core.wallet import get_default_wallet


async def run_real(resource: str, price_usd: float, pay_to: str) -> dict:
    wallet = get_default_wallet()
    if wallet is None:
        print(
            "AGENT_WALLET_PRIVATE_KEY not set — cannot produce a REAL trace. "
            "Fund a testnet wallet via https://faucet.circle.com/, set the env "
            "var, and re-run. Or pass --offline for a structural example.",
            file=sys.stderr,
        )
        sys.exit(1)

    requirements = PaymentRequirements(
        max_amount_required=usd_to_atomic(price_usd),
        pay_to=pay_to,
        resource=resource,
        description=f"Access to '{resource}'.",
    )
    challenge = X402Challenge(accepts=[requirements])
    trace: dict = {"resource": resource, "generated_at": time.time(), "mode": "real"}
    trace["initial_402_challenge"] = challenge.model_dump()

    auth = wallet.sign_transfer_authorization(
        to_address=requirements.pay_to,
        value_atomic=requirements.max_amount_required,
        contract_address=requirements.asset,
    )
    payment = X402PaymentPayload(payload=auth, resource=resource)
    trace["signature_metadata"] = payment.model_dump(by_alias=True)

    facilitator = get_facilitator_client()
    verify_result = await facilitator.verify(payment, requirements)
    trace["verify_result"] = verify_result.model_dump()

    if verify_result.is_valid:
        settlement = await facilitator.settle(payment, requirements)
        trace["settlement_result"] = settlement.model_dump()
    else:
        trace["settlement_result"] = None
        trace["note"] = "Verification failed — settlement was not attempted."

    return trace


def build_offline_example(resource: str, price_usd: float, pay_to: str) -> dict:
    """Schema-accurate but NOT cryptographically real — every signature/hash
    field below is a placeholder, clearly marked, for reviewing the JSON
    shape without a funded wallet or live facilitator."""
    requirements = PaymentRequirements(
        max_amount_required=usd_to_atomic(price_usd),
        pay_to=pay_to or "0x0000000000000000000000000000000000dEaD",
        resource=resource,
        description=f"Access to '{resource}'.",
    )
    challenge = X402Challenge(accepts=[requirements])
    return {
        "resource": resource,
        "generated_at": time.time(),
        "mode": "offline_structural_example",
        "warning": "This trace uses placeholder signature/settlement fields — not a real testnet run.",
        "initial_402_challenge": challenge.model_dump(),
        "signature_metadata": {
            "x402_version": 1,
            "scheme": "exact",
            "network": "base-sepolia",
            "resource": resource,
            "payload": {
                "from": "0xPLACEHOLDER_PAYER_ADDRESS",
                "to": requirements.pay_to,
                "value": requirements.max_amount_required,
                "validAfter": 0,
                "validBefore": int(time.time()) + 300,
                "nonce": "0x" + "ab" * 32,
                "v": 27,
                "r": "0x" + "cd" * 32,
                "s": "0x" + "ef" * 32,
            },
        },
        "verify_result": {"is_valid": True, "invalid_reason": None},
        "settlement_result": {
            "success": True,
            "transaction_hash": "0x" + "11" * 32,
            "network": "base-sepolia",
            "payer": "0xPLACEHOLDER_PAYER_ADDRESS",
            "amount_atomic": requirements.max_amount_required,
            "settled_at": time.time(),
            "error": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", default="query_knowledge")
    parser.add_argument("--price-usd", type=float, default=0.01)
    parser.add_argument("--pay-to", default="")
    parser.add_argument("--offline", action="store_true", help="Produce a structural example instead of a real run.")
    parser.add_argument("--out", default="finops_compliance_audit.json")
    args = parser.parse_args()

    import os
    pay_to = args.pay_to or os.getenv("SERVER_WALLET_ADDRESS", "")

    if args.offline:
        trace = build_offline_example(args.resource, args.price_usd, pay_to)
    else:
        trace = asyncio.run(run_real(args.resource, args.price_usd, pay_to))

    with open(args.out, "w") as f:
        json.dump(trace, f, indent=2, default=str)
    print(f"Wrote {args.out} (mode={trace['mode']})")


if __name__ == "__main__":
    main()
