"""Tests for Polymarket client error handling."""

import asyncio
from unittest.mock import Mock, patch

import pytest
from domain.models import MarketMetadata, TradeDecision
from py_clob_client.clob_types import OrderType, PostOrdersArgs
from py_clob_client.exceptions import PolyApiException
from trader.polymarket_client import PolymarketClient


@pytest.fixture
def mock_client():
    """Create a PolymarketClient with mocked CLOB client."""
    with patch("trader.polymarket_client.ClobClient") as mock_clob:
        # Mock successful credential derivation
        mock_instance = Mock()
        mock_instance.create_or_derive_api_creds.return_value = {"api_key": "test"}
        mock_instance.assert_level_1_auth = Mock(return_value=None)

        mock_builder = Mock()
        mock_instance.builder = mock_builder

        mock_clob.return_value = mock_instance

        client = PolymarketClient(api_key="0x" + "1" * 64, wallet_address="0xtest")
        yield client, mock_instance


class TestAuthentication:
    """Tests for client authentication."""

    def test_authentication_failure(self):
        """Test that authentication failures raise RuntimeError."""
        with patch("trader.polymarket_client.ClobClient") as mock_clob:
            mock_instance = Mock()
            mock_instance.create_or_derive_api_creds.side_effect = Exception("Auth failed")
            mock_clob.return_value = mock_instance

            with pytest.raises(RuntimeError, match="Polymarket client authentication failed"):
                PolymarketClient(api_key="0x" + "1" * 64, wallet_address="0xtest")


class TestConnection:
    """Tests for connection checking."""

    def test_check_connection_success(self, mock_client):
        """Test check_connection returns True when API keys can be fetched."""
        client, mock_clob_instance = mock_client

        # Mock successful API key fetch
        mock_clob_instance.get_api_keys.return_value = {"keys": []}

        assert client.check_connection() is True

    def test_check_connection_failure(self, mock_client):
        """Test check_connection returns False when fetch fails."""
        client, mock_clob_instance = mock_client

        # Mock failure
        mock_clob_instance.get_api_keys.side_effect = Exception("Connection failed")

        assert client.check_connection() is False


class TestPlaceOrders:
    """Tests for place_orders method."""

    def test_place_orders_success(self, mock_client):
        """Test successful order placement."""
        client, mock_clob_instance = mock_client

        mock_clob_instance.post_orders.return_value = [{"success": True, "orderID": "order_123"}]

        mock_order = Mock()
        orders = [PostOrdersArgs(order=mock_order, orderType=OrderType.FAK)]  # type: ignore[arg-type]

        result = client.place_orders(orders)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["orderID"] == "order_123"
        mock_clob_instance.post_orders.assert_called_once_with(args=orders)

    def test_place_orders_rejected(self, mock_client):
        """Test handling when Polymarket API rejects order."""
        client, mock_clob_instance = mock_client

        mock_clob_instance.post_orders.return_value = [{"success": False, "errorMsg": "rejected"}]

        mock_order = Mock()
        orders = [PostOrdersArgs(order=mock_order, orderType=OrderType.FAK)]  # type: ignore[arg-type]

        result = client.place_orders(orders)

        assert isinstance(result, list)
        assert result[0]["errorMsg"] == "rejected"

    def test_place_orders_empty_response(self, mock_client):
        """Test handling of empty response."""
        client, mock_clob_instance = mock_client

        mock_clob_instance.post_orders.return_value = None

        mock_order = Mock()
        orders = [PostOrdersArgs(order=mock_order, orderType=OrderType.FAK)]  # type: ignore[arg-type]

        result = client.place_orders(orders)

        assert result == []


class TestGetSignedOrder:
    """Tests for get_signed_order method."""

    def test_get_signed_order_yes_outcome(self, mock_client):
        """Test creating signed order for YES outcome."""
        client, mock_clob_instance = mock_client

        market = MarketMetadata(
            question="Test market",
            strike_price=100.0,
            yes_token_id="yes_token_123",
            no_token_id="no_token_456",
            fee_rate_bps=100,
            tick_size="0.01",
            neg_risk=False,
        )

        decision = TradeDecision(
            should_trade=True,
            side="BUY",
            outcome="YES",
            size=10,
            price=0.5,
            total_amount=5.0,
            reason="Test trade",
        )

        mock_signed_order = Mock()
        mock_clob_instance.builder.create_order.return_value = mock_signed_order

        result = client.get_signed_order(market, decision)

        assert result is mock_signed_order
        mock_clob_instance.assert_level_1_auth.assert_called_once()

        # Verify order args
        call_args = mock_clob_instance.builder.create_order.call_args
        order_args = call_args.kwargs["order_args"]
        assert order_args.token_id == "yes_token_123"
        assert order_args.price == 0.5
        assert order_args.size == 10
        assert order_args.side == "BUY"
        assert order_args.fee_rate_bps == 100

    def test_get_signed_order_no_outcome(self, mock_client):
        """Test creating signed order for NO outcome."""
        client, mock_clob_instance = mock_client

        market = MarketMetadata(
            question="Test market",
            strike_price=100.0,
            yes_token_id="yes_token_123",
            no_token_id="no_token_456",
            fee_rate_bps=50,
            tick_size="0.01",
            neg_risk=True,
        )

        decision = TradeDecision(
            should_trade=True,
            side="BUY",
            outcome="NO",
            size=20,
            price=0.3,
            total_amount=6.0,
            reason="Test trade",
        )

        mock_signed_order = Mock()
        mock_clob_instance.builder.create_order.return_value = mock_signed_order

        result = client.get_signed_order(market, decision)

        assert result is mock_signed_order

        # Verify NO token was used
        call_args = mock_clob_instance.builder.create_order.call_args
        order_args = call_args.kwargs["order_args"]
        assert order_args.token_id == "no_token_456"

    def test_get_signed_order_uses_defaults(self, mock_client):
        """Test that defaults are used when market fields are None."""
        client, mock_clob_instance = mock_client

        market = MarketMetadata(
            question="Test market",
            strike_price=100.0,
            yes_token_id="yes_token_123",
            no_token_id="no_token_456",
            fee_rate_bps=None,  # Should default to 0
            tick_size=None,  # Should default to "0.01"
            neg_risk=None,  # Should default to False
        )

        decision = TradeDecision(
            should_trade=True,
            side="BUY",
            outcome="YES",
            size=10,
            price=0.5,
            total_amount=5.0,
            reason="Test trade",
        )

        mock_signed_order = Mock()
        mock_clob_instance.builder.create_order.return_value = mock_signed_order

        client.get_signed_order(market, decision)

        call_args = mock_clob_instance.builder.create_order.call_args
        order_args = call_args.kwargs["order_args"]
        order_options = call_args.kwargs["options"]

        assert order_args.fee_rate_bps == 0
        assert order_options.tick_size == "0.01"
        assert order_options.neg_risk is False

    def test_get_signed_order_invalid_outcome(self, mock_client):
        """Test that invalid outcome raises ValueError."""
        client, mock_clob_instance = mock_client

        market = MarketMetadata(
            question="Test market",
            strike_price=100.0,
            yes_token_id="yes_token_123",
            no_token_id="no_token_456",
        )

        decision = TradeDecision(
            should_trade=True,
            side="BUY",
            outcome=None,  # Invalid
            size=10,
            price=0.5,
            total_amount=5.0,
            reason="Test trade",
        )

        with pytest.raises(ValueError, match="Invalid outcome"):
            client.get_signed_order(market, decision)

    def test_get_signed_order_missing_token_id(self, mock_client):
        """Test that missing token_id raises ValueError."""
        client, mock_clob_instance = mock_client

        market = MarketMetadata(
            question="Test market",
            strike_price=100.0,
            yes_token_id=None,  # Missing
            no_token_id="no_token_456",
        )

        decision = TradeDecision(
            should_trade=True,
            side="BUY",
            outcome="YES",
            size=10,
            price=0.5,
            total_amount=5.0,
            reason="Test trade",
        )

        with pytest.raises(ValueError, match="Missing token_id"):
            client.get_signed_order(market, decision)


class TestGetFillAndKillOrderObject:
    """Tests for get_fill_and_kill_order_object method."""

    def test_creates_fak_order(self, mock_client):
        """Test that FAK order object is created correctly."""
        client, _ = mock_client

        mock_signed_order = Mock()
        result = client.get_fill_and_kill_order_object(mock_signed_order)

        assert isinstance(result, PostOrdersArgs)
        assert result.order is mock_signed_order
        assert result.orderType == OrderType.FAK


class TestPlaceOrdersPolyApiException:
    """Tests for PolyApiException handling in place_orders."""

    def test_place_orders_poly_api_exception(self, mock_client):
        """Test handling of PolyApiException during order placement."""
        client, mock_clob_instance = mock_client

        # Create a mock response object for PolyApiException
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "API rate limit exceeded"

        mock_clob_instance.post_orders.side_effect = PolyApiException(resp=mock_response)

        mock_order = Mock()
        orders = [PostOrdersArgs(order=mock_order, orderType=OrderType.FAK)]  # type: ignore[arg-type]

        with pytest.raises(PolyApiException):
            client.place_orders(orders)


class TestPlaceOrdersAsync:
    """Tests for place_orders_async method."""

    def test_place_orders_async_success(self, mock_client):
        """Test async order placement."""
        client, mock_clob_instance = mock_client

        mock_clob_instance.post_orders.return_value = [{"success": True, "orderID": "async_order_123"}]

        mock_order = Mock()
        orders = [PostOrdersArgs(order=mock_order, orderType=OrderType.FAK)]  # type: ignore[arg-type]

        result = asyncio.run(client.place_orders_async(orders))

        assert isinstance(result, list)
        assert result[0]["orderID"] == "async_order_123"
