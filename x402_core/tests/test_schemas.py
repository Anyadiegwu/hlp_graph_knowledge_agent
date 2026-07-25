"""
Run: uv run --package x402_core pytest x402_core/tests/test_schemas.py -v
"""
from x402_core.constants import usd_to_atomic, atomic_to_usd
from x402_core.schemas import (
    EIP3009Authorization,
    PaymentRequirements,
    X402Challenge,
    X402PaymentPayload,
)


def test_usd_atomic_roundtrip():
    assert usd_to_atomic(0.05) == "50000"
    assert usd_to_atomic(0.01) == "10000"
    assert atomic_to_usd("50000") == 0.05
    assert atomic_to_usd(50000) == 0.05


def test_challenge_json_roundtrip():
    requirements = PaymentRequirements(
        max_amount_required=usd_to_atomic(0.01),
        pay_to="0x000000000000000000000000000000000000dEaD",
        resource="query_knowledge",
    )
    challenge = X402Challenge(accepts=[requirements])
    raw = challenge.to_json()

    parsed = X402Challenge.model_validate_json(raw)
    assert parsed.error == "payment_required"
    assert parsed.accepts[0].resource == "query_knowledge"
    assert parsed.accepts[0].max_amount_required == "10000"


def test_payment_payload_uses_aliases_for_authorization_fields():
    # EIP3009Authorization aliases "from"/"to" (reserved words in Python),
    # so by_alias=True must be used on dump or the JSON won't match what
    # a real facilitator expects.
    auth = EIP3009Authorization(
        **{
            "from": "0xPayerAddress",
            "to": "0xPayeeAddress",
            "value": "10000",
            "valid_after": 0,
            "valid_before": 9999999999,
            "nonce": "0x" + "ab" * 32,
            "v": 27,
            "r": "0x" + "cd" * 32,
            "s": "0x" + "ef" * 32,
        }
    )
    payload = X402PaymentPayload(payload=auth, resource="query_knowledge")
    raw = payload.to_json()

    assert '"from":"0xPayerAddress"' in raw.replace(" ", "")
    assert '"to":"0xPayeeAddress"' in raw.replace(" ", "")

    reparsed = X402PaymentPayload.model_validate_json(raw)
    assert reparsed.payload.authorization.from_address == "0xPayerAddress"
    assert reparsed.payload.authorization.to_address == "0xPayeeAddress"
