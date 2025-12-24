"""Trader package test fixtures."""

import pytest
from trader.config import Config
from trader.polymarket_client import PolymarketClient


@pytest.fixture
def real_polymarket_client():
    """Create a real PolymarketClient with actual credentials.

    Available to trader tests. Skips test if credentials not set.
    """
    test_config = Config.load()

    if test_config.polymarket_wallet == "test_wallet_address" or test_config.polymarket_api_key == "test_api_key_123":
        pytest.skip("POLYMARKET_WALLET and POLYMARKET_API_KEY required for this test")

    client = PolymarketClient(
        api_key=test_config.polymarket_api_key,
        wallet_address=test_config.polymarket_wallet,
    )
    return client
