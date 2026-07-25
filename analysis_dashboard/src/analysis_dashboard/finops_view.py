"""
Data-fetching helpers for the FinOps dashboard tab.

Deliberately kept free of any Streamlit imports — app.py renders, this
module just answers "what's the current state of the wallet / ledger /
audit trail", so it can also be reused by generate_finops_audit.py.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from x402_core.audit import read_audit_trail
from x402_core.constants import USDC_CONTRACT_ADDRESS, USDC_DECIMALS, atomic_to_usd
from x402_core.finops import get_governor

# `balanceOf(address)` selector, ERC-20 standard.
_BALANCE_OF_SELECTOR = "0x70a08231"


async def fetch_usdc_balance(wallet_address: str, rpc_url: str | None = None) -> float | None:
    """Reads the USDC balance of `wallet_address` on Base Sepolia via a
    raw `eth_call` JSON-RPC request — no wallet/web3 dependency needed for
    a read-only balance check."""
    rpc_url = rpc_url or os.getenv("BASE_SEPOLIA_RPC_URL", "")
    if not rpc_url or not wallet_address:
        return None
    padded_address = wallet_address.lower().replace("0x", "").rjust(64, "0")
    call_data = _BALANCE_OF_SELECTOR + padded_address
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": USDC_CONTRACT_ADDRESS, "data": call_data}, "latest"],
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(rpc_url, json=body)
            resp.raise_for_status()
            data = resp.json()
        raw_hex = data.get("result")
        if not raw_hex:
            return None
        atomic = int(raw_hex, 16)
        return atomic_to_usd(atomic, USDC_DECIMALS)
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def summarize_ledger(audit_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates the shared audit trail into spend/earn totals and a
    chronological, human-labeled event stream for the dashboard."""
    spent_usd = 0.0
    earned_usd = 0.0
    settlements = 0
    challenges = 0
    verifications_failed = 0

    for entry in audit_entries:
        event = entry.get("event", "")
        if event in ("challenge_issued", "reciprocal_challenge_issued"):
            challenges += 1
        if event in ("verify_attempted", "reciprocal_verify_attempted") and entry.get("is_valid") is False:
            verifications_failed += 1
        if event in ("settlement_recorded", "reciprocal_settlement_recorded") and entry.get("success"):
            settlements += 1
            amount_atomic = entry.get("amount_atomic")
            usd = atomic_to_usd(amount_atomic) if amount_atomic is not None else 0.0
            if entry.get("direction") == "earned":
                earned_usd += usd
            else:
                spent_usd += usd

    return {
        "spent_usd": spent_usd,
        "earned_usd": earned_usd,
        "settlements": settlements,
        "challenges_issued": challenges,
        "verifications_failed": verifications_failed,
        "net_usd": earned_usd - spent_usd,
    }


def get_velocity_snapshot(session_id: str) -> float:
    """Current spend velocity (USD/hour) for a session, from the shared
    FinOps governor ledger."""
    return get_governor().velocity_usd_per_hour(session_id)
