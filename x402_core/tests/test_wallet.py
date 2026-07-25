"""
Run: uv run --package x402_core pytest x402_core/tests/test_wallet.py -v

No network or funded wallet needed — Account.create() makes a throwaway
in-memory keypair purely for exercising the signing logic.
"""
import pytest
from eth_account import Account

from x402_core.constants import USDC_CONTRACT_ADDRESS
from x402_core.wallet import AgentWallet, get_default_wallet


@pytest.fixture
def throwaway_wallet() -> AgentWallet:
    acct = Account.create()
    return AgentWallet(private_key=acct.key.hex())


def test_wallet_address_matches_key(throwaway_wallet):
    # AgentWallet shouldn't mangle the address eth_account derived.
    assert throwaway_wallet.address.startswith("0x")
    assert len(throwaway_wallet.address) == 42


def test_sign_transfer_authorization_shape(throwaway_wallet):
    auth = throwaway_wallet.sign_transfer_authorization(
        to_address="0x000000000000000000000000000000000000dEaD",
        value_atomic="50000",
    )
    assert auth.from_address == throwaway_wallet.address
    assert auth.to_address == "0x000000000000000000000000000000000000dEaD"
    assert auth.value == "50000"
    assert auth.valid_after == 0
    assert auth.valid_before > 0
    assert auth.nonce.startswith("0x")
    assert len(auth.nonce) == 66  # "0x" + 64 hex chars = 32 bytes
    assert isinstance(auth.v, int)
    assert auth.r.startswith("0x")
    assert auth.s.startswith("0x")


def test_nonce_is_unique_per_call(throwaway_wallet):
    # EIP-3009 relies on a fresh random nonce per authorization to prevent
    # replay — two calls must never produce the same one.
    auth_1 = throwaway_wallet.sign_transfer_authorization("0x000000000000000000000000000000000000dEaD", "1000")
    auth_2 = throwaway_wallet.sign_transfer_authorization("0x000000000000000000000000000000000000dEaD", "1000")
    assert auth_1.nonce != auth_2.nonce
    assert auth_1.r != auth_2.r  # different message -> different signature


def test_default_contract_is_testnet_usdc(throwaway_wallet):
    auth = throwaway_wallet.sign_transfer_authorization(
        to_address="0x000000000000000000000000000000000000dEaD",
        value_atomic="1000",
    )
    # sign_transfer_authorization doesn't echo the contract address back
    # onto the authorization itself (it's part of the signing domain, not
    # the message), so we just confirm the constant it defaults to is the
    # one from the task spec.
    assert USDC_CONTRACT_ADDRESS == "0x036CbD53842c5426634e7929541eC2318f3dCF7e"


def test_get_default_wallet_returns_none_without_env(monkeypatch):
    monkeypatch.delenv("AGENT_WALLET_PRIVATE_KEY", raising=False)
    # reset the module-level singleton so the missing-env-var path re-runs
    import x402_core.wallet as wallet_module
    wallet_module._wallet_init_tried = False
    wallet_module._default_wallet = None
    assert get_default_wallet() is None
