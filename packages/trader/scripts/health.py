#!/usr/bin/env python3
"""
Health check script for the trader service.

Checks:
1. Required secrets are accessible
2. Polymarket client can authenticate and connect
3. Stores are being updated (via heartbeat files)

Exit codes:
- 0: Healthy
- 1: Unhealthy
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from domain.paths import EVENT_STORE_HEARTBEAT, OPTION_STORE_HEARTBEAT
from domain.secrets import load_required_secret
from domain.utils import is_market_open
from trader.polymarket_client import PolymarketClient

MAX_STALE_SECONDS = 300  # 5 minutes


def check_secrets() -> bool:
    """Check that required secrets are accessible."""
    required = ["POLYMARKET_API_KEY", "POLYMARKET_WALLET", "MASSIVE_API_KEY"]

    for secret in required:
        try:
            load_required_secret(secret)
        except Exception as e:
            print(f"Missing secret {secret}: {e}")
            return False

    return True


def check_polymarket() -> bool:
    """Check Polymarket client can authenticate and connect."""
    try:
        client = PolymarketClient(
            api_key=load_required_secret("POLYMARKET_API_KEY"),
            wallet_address=load_required_secret("POLYMARKET_WALLET"),
        )
        return client.check_connection()
    except Exception as e:
        print(f"Polymarket connection failed: {e}")
        return False


def check_heartbeat(heartbeat_file: Path, name: str) -> bool:
    """Check a heartbeat file is fresh."""
    try:
        timestamp = datetime.fromisoformat(heartbeat_file.read_text().strip())
        age = datetime.now(UTC) - timestamp

        if age > timedelta(seconds=MAX_STALE_SECONDS):
            print(f"{name}: stale ({age.total_seconds():.0f}s)")
            return False

        print(f"{name}: ok ({age.total_seconds():.0f}s ago)")
        return True

    except Exception as e:
        print(f"{name}: error reading heartbeat: {e}")
        return False


def main() -> int:
    checks = [
        ("secrets", check_secrets),
        ("polymarket", check_polymarket),
        ("event_store", lambda: check_heartbeat(EVENT_STORE_HEARTBEAT, "event_store")),
    ]

    # Only check option_store heartbeat when market is open
    if is_market_open():
        checks.append(("option_store", lambda: check_heartbeat(OPTION_STORE_HEARTBEAT, "option_store")))
    else:
        print("option_store: skipped (market closed)")

    all_passed = True
    for name, check in checks:
        try:
            if check():
                print(f"✓ {name}")
            else:
                print(f"✗ {name}")
                all_passed = False
        except Exception as e:
            print(f"✗ {name}: {e}")
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
