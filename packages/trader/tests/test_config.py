"""Tests for configuration module."""

import os

from trader.config import Config


def test_load_with_pytest_running():
    """Test that Config.load() returns safe test values during pytest."""
    config = Config.load()
    assert config.polymarket_wallet is not None
    assert config.polymarket_api_key is not None
    if not os.getenv("RUN_UNSAFE_TESTS"):
        assert config.polymarket_wallet == "test_wallet_address"
        assert config.polymarket_api_key == "test_api_key_123"
    assert config.min_edge == 0.1
    assert config.max_total_amount == 1.0


def test_default_symbols():
    """Test that default symbols list is populated."""
    config = Config(polymarket_wallet="test", polymarket_api_key="test")

    assert len(config.symbols) > 0
    assert "NVDA" in config.symbols
    assert "AAPL" in config.symbols


def test_config_fields():
    """Test Config has expected default values."""
    config = Config(polymarket_wallet="wallet", polymarket_api_key="key")

    assert config.model_loop_interval_ms == 60000
    assert config.polymarket_refresh_interval_s == 300
    assert config.max_days == 14
    assert config.min_confidence_score == 0.4
    assert config.max_spread == 1.0
