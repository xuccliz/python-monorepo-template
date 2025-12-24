"""Real integration tests for PolymarketClient.

These tests fetch real market data and require valid credentials.
Run with: RUN_UNSAFE_TESTS=1 pytest packages/trader/tests/test_polymarket_client_integration.py -v
"""

import pytest
import requests
from domain.models import MarketMetadata, TradeDecision
from py_clob_client.clob_types import CreateOrderOptions, OrderArgs


def fetch_real_markets(limit=5):
    """Fetch real active markets from Polymarket API."""
    url = "https://clob.polymarket.com/markets"
    params = {"active": "true", "closed": "false", "limit": limit}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    market_list = data.get("data", [])

    markets = []
    for market in market_list:
        tokens = market.get("tokens", [])
        if not tokens:
            continue

        yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)
        no_token = next((t for t in tokens if t.get("outcome") == "No"), None)
        if not yes_token:
            continue

        markets.append(
            {
                "question": market.get("question", "Unknown"),
                "yes_token_id": yes_token.get("token_id"),
                "no_token_id": no_token.get("token_id") if no_token else None,
                "fee_rate_bps": market.get("maker_base_fee"),
                "tick_size": str(market.get("minimum_tick_size")),
                "neg_risk": market.get("neg_risk"),
            }
        )

        if len(markets) >= limit:
            break

    return markets


@pytest.mark.unsafe
def test_polymarket_api_health(real_polymarket_client):
    """Test that Polymarket API is accessible and credentials work."""
    client = real_polymarket_client
    assert client._client is not None
    assert client.check_connection() is True


@pytest.mark.unsafe
def test_get_signed_order_vs_clob_client_with_real_markets(real_polymarket_client):
    """Compare get_signed_order vs ClobClient.create_order with real market data.

    Fetches real markets and verifies that our wrapper produces the same
    type of result as ClobClient.create_order would.
    """
    client = real_polymarket_client

    real_markets = fetch_real_markets(limit=10)
    assert len(real_markets) >= 5, "Could not fetch enough real markets from API"

    for market_data in real_markets:
        price = 0.01
        size = 1

        # Build MarketMetadata and TradeDecision for our wrapper
        market = MarketMetadata(
            question=market_data["question"],
            strike_price=0.0,
            yes_token_id=market_data["yes_token_id"],
            no_token_id=market_data["no_token_id"],
            fee_rate_bps=market_data["fee_rate_bps"],
            tick_size=market_data["tick_size"],
            neg_risk=market_data["neg_risk"],
        )

        decision = TradeDecision.execute_trade(
            side="BUY",
            outcome="YES",
            size=size,
            price=price,
            total_amount=price * size,
            reason="Integration test",
        )

        # Call our wrapper
        wrapper_result = client.get_signed_order(market=market, decision=decision)

        # Call library directly for comparison
        order_args = OrderArgs(
            token_id=market_data["yes_token_id"],
            price=price,
            size=size,
            side="BUY",
            fee_rate_bps=market_data["fee_rate_bps"],
        )
        order_options = CreateOrderOptions(
            tick_size=market_data["tick_size"],
            neg_risk=market_data["neg_risk"],
        )
        clob_result = client._client.create_order(order_args, order_options)

        # Compare types
        assert isinstance(wrapper_result, type(clob_result)), (
            f"Different types for market: {market_data['question']}\n"
            f"Wrapper: {type(wrapper_result)}\n"
            f"ClobClient: {type(clob_result)}"
        )

        # Verify both have signature and order
        assert hasattr(wrapper_result, "signature")
        assert hasattr(wrapper_result, "order")
        assert hasattr(clob_result, "signature")
        assert hasattr(clob_result, "order")

        # Verify signatures are valid
        assert wrapper_result.signature
        assert clob_result.signature
        assert len(wrapper_result.signature) > 10
        assert len(clob_result.signature) > 10

        # Verify order structures match
        assert isinstance(wrapper_result.order, type(clob_result.order))
        assert hasattr(wrapper_result.order, "maker")
        assert hasattr(clob_result.order, "maker")

        # Compare order fields - extract actual primitive values
        wrapper_dict = wrapper_result.order.data_dict()
        clob_dict = clob_result.order.data_dict()

        # Assert all fields match except salt (unique per order)
        all_fields = set(wrapper_dict.keys()) | set(clob_dict.keys())
        for field in all_fields:
            if field == "salt":
                continue
            assert wrapper_dict.get(field) == clob_dict.get(field), (
                f"Field '{field}' differs for market: {market_data['question']}\n"
                f"  Wrapper: {wrapper_dict.get(field)}\n"
                f"  Clob:    {clob_dict.get(field)}"
            )
