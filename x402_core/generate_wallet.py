"""
Generates a fresh Base Sepolia wallet and prints the address + private key
straight to stdout — nothing else.

Deliberately uses print(), not the `logging` module: logging handlers can
end up writing to a log file, a log aggregator, or a monitoring backend
without you realizing it, and a private key has no business landing in
any of those. print() only ever goes to this terminal.

This script also never writes anything to disk (.env or otherwise) — you
decide where the key goes. Copy it into your own .env yourself, or export
it directly for a single session:

    uv run --package x402_core python x402_core/generate_wallet.py

    export AGENT_WALLET_PRIVATE_KEY=0x...   # paste the printed key
    export SERVER_WALLET_ADDRESS=0x...      # if this wallet is the payee too

Once the address below is funded via https://faucet.circle.com/ (manual —
that faucet sits behind a CAPTCHA specifically to block scripted claims,
so this tool doesn't attempt that step), run the audit generator
separately:

    uv run --package x402_core python x402_core/generate_finops_audit.py \
        --resource query_knowledge --price-usd 0.01
"""
from __future__ import annotations

from eth_account import Account


def main() -> None:
    acct = Account.create()

    print("=" * 70)
    print("New Base Sepolia wallet generated — shown ONCE, not saved anywhere.")
    print("=" * 70)
    print()
    print(f"Address:     {acct.address}")
    print(f"Private key: {acct.key.hex()}")
    print()
    print("Next steps:")
    print(f"  1. Fund this address at https://faucet.circle.com/ (select Base Sepolia).")
    print("  2. export AGENT_WALLET_PRIVATE_KEY=<the private key above>")
    print("  3. Once funded, run generate_finops_audit.py in a separate command.")
    print()
    print("This key is not written to any file or log by this script — if you")
    print("want it persisted, add it to your own .env yourself.")


if __name__ == "__main__":
    main()
