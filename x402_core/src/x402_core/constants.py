"""
Shared on-chain / protocol constants for the x402 stablecoin billing layer.

Every number here is testnet-only (Base Sepolia). Nothing in this module
should ever be pointed at mainnet without a full security review — the
EIP-3009 authorizations this package signs move real value once the
underlying asset is a real USDC contract.
"""
from __future__ import annotations

# CAIP-2 chain identifier for Base Sepolia.
CAIP2_CHAIN_ID = "eip155:84532"
EVM_CHAIN_ID = 84532

# Testnet USDC on Base Sepolia (per task spec).
USDC_CONTRACT_ADDRESS = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
USDC_DECIMALS = 6

# Xpay staging facilitator — brokers signature verification + settlement
# relay so agent wallets never need native Base Sepolia ETH for gas.
XPAY_FACILITATOR_URL = "https://facilitator.xpay.sh"

# EIP-3009 domain fields for the USDC contract. `name`/`version` follow the
# canonical Circle USDC EIP-712 domain; verify against the deployed staging
# contract's `name()`/`version()` before relying on this in anger.
EIP712_DOMAIN_NAME = "USDC"
EIP712_DOMAIN_VERSION = "2"

# x402 scheme identifiers, per the x402 spec's `accepts[].scheme` field.
X402_SCHEME_EXACT = "exact"
X402_NETWORK_ID = "base-sepolia"

# Default single-shot ceiling for one full agent execution chain — this is
# a *safety* ceiling, not a pricing decision. Tool prices are independent
# and set per-endpoint (see mcp_server.paywall).
DEFAULT_MAX_USD_PER_CHAIN = 0.05


def usd_to_atomic(usd: float, decimals: int = USDC_DECIMALS) -> str:
    """Convert a USD float amount to a 6-decimal atomic unit string.

    x402 requires atomic amounts as decimal strings (not floats) to avoid
    floating point drift in on-chain-adjacent arithmetic — e.g.
    0.05 USDC -> "50000".
    """
    atomic = round(usd * (10 ** decimals))
    return str(atomic)


def atomic_to_usd(atomic: str | int, decimals: int = USDC_DECIMALS) -> float:
    return int(atomic) / (10 ** decimals)
