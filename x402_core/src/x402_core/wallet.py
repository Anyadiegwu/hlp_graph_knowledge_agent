"""
Non-custodial EVM wallet helper for signing EIP-3009
`transferWithAuthorization` payloads off-chain.

No native Base Sepolia ETH ever leaves this wallet and no gas is spent by
it directly — the signed authorization is handed to the Xpay smart proxy,
which relays it on-chain as the gas sponsor. This module only ever
produces a signature; it never itself calls a JSON-RPC `sendTransaction`.
"""
from __future__ import annotations

import os
import secrets
import time

from eth_account import Account
from eth_account.messages import encode_typed_data

from x402_core.constants import (
    EIP712_DOMAIN_NAME,
    EIP712_DOMAIN_VERSION,
    EVM_CHAIN_ID,
    USDC_CONTRACT_ADDRESS,
)
from x402_core.schemas import EIP3009Authorization, ExactSchemePayload
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env"))

_TRANSFER_WITH_AUTHORIZATION_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "TransferWithAuthorization": [
        {"name": "from", "type": "address"},
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "validAfter", "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce", "type": "bytes32"},
    ],
}


class AgentWallet:
    """Wraps a single programmatic EOA used to authorize micropayments.

    The private key is read from an environment variable rather than a
    file or CLI flag so it never lands in shell history or a repo — see
    `.env.example` for `AGENT_WALLET_PRIVATE_KEY`.
    """

    def __init__(self, private_key: str | None = None) -> None:
        pk = private_key or os.getenv("AGENT_WALLET_PRIVATE_KEY", "")
        if not pk:
            raise ValueError(
                "AGENT_WALLET_PRIVATE_KEY not set — cannot sign x402 "
                "authorizations without a funded testnet wallet. Fund one "
                "via https://faucet.circle.com/ and set the env var."
            )
        self._account = Account.from_key(pk)

    @property
    def address(self) -> str:
        return self._account.address

    def sign_transfer_authorization(
        self,
        to_address: str,
        value_atomic: str,
        contract_address: str = USDC_CONTRACT_ADDRESS,
        valid_for_seconds: int = 300,
    ) -> ExactSchemePayload:
        now = int(time.time())
        valid_after = 0
        valid_before = now + valid_for_seconds
        nonce = "0x" + secrets.token_hex(32)

        message = {
            "from": self._account.address,
            "to": to_address,
            "value": int(value_atomic),
            "validAfter": valid_after,
            "validBefore": valid_before,
            "nonce": nonce,
        }
        domain = {
            "name": EIP712_DOMAIN_NAME,
            "version": EIP712_DOMAIN_VERSION,
            "chainId": EVM_CHAIN_ID,
            "verifyingContract": contract_address,
        }
        typed_data = {
            "types": _TRANSFER_WITH_AUTHORIZATION_TYPES,
            "primaryType": "TransferWithAuthorization",
            "domain": domain,
            "message": message,
        }
        signable = encode_typed_data(full_message=typed_data)
        signed = self._account.sign_message(signable)

        # Combine r (32 bytes) + s (32 bytes) + v (1 byte) into one 65-byte hex signature
        r_hex = signed.r.to_bytes(32, "big").hex()
        s_hex = signed.s.to_bytes(32, "big").hex()
        v_hex = signed.v.to_bytes(1, "big").hex()
        signature = "0x" + r_hex + s_hex + v_hex

        from eth_account import Account 
        # simpler, version-agnostic approach:
        recovered_addr = Account.recover_message(signable, signature=signed.signature)
        if recovered_addr.lower() != self._account.address.lower():
            raise RuntimeError(
                f"SELF-CHECK FAILED: signature recovers to {recovered_addr}, "
                f"expected {self._account.address}. Signing/combination logic is broken."
            )
        print(f"SELF-CHECK OK: signature correctly recovers to {recovered_addr}")
        return ExactSchemePayload(
            signature=signature,
            authorization=EIP3009Authorization(
                **{
                    "from": self._account.address,
                    "to": to_address,
                    "value": value_atomic,
                    "valid_after": valid_after,
                    "valid_before": valid_before,
                    "nonce": nonce,
                }
            ),
        )

_default_wallet: AgentWallet | None = None
_wallet_init_tried = False


def get_default_wallet() -> AgentWallet | None:
    """Lazy singleton — returns None (rather than raising) when no key is
    configured, so callers can degrade to 'payment unavailable' instead of
    crashing at import time."""
    global _default_wallet, _wallet_init_tried
    if _wallet_init_tried:
        return _default_wallet
    _wallet_init_tried = True
    try:
        _default_wallet = AgentWallet()
    except ValueError:
        _default_wallet = None
    return _default_wallet
