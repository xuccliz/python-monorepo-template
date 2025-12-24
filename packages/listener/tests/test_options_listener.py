"""
Unit tests for OptionsQuoteListener.

Tests cover:
- Initialization with explicit symbols
- Subscription building
- Message handling
- Queue integration
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from domain.models import OptionQuoteEvent
from massive.websocket.models import EquityQuote


class TestOptionsQuoteListener:
    """Tests for OptionsQuoteListener class."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock the API key secret loading."""
        with patch("listener.options_listener.load_required_secret") as mock:
            mock.return_value = "test_api_key"
            yield mock

    @pytest.fixture
    def event_queue(self):
        """Create an asyncio queue for events."""
        return asyncio.Queue()

    @pytest.fixture
    def sample_symbols(self):
        """Sample option symbols for testing."""
        return ["O:NVDA250117C00140000", "O:NVDA250117P00140000"]

    def test_init_with_explicit_symbols(self, mock_api_key, event_queue, sample_symbols):
        """Test initialization with explicit option symbols."""
        from listener.options_listener import OptionsQuoteListener

        listener = OptionsQuoteListener(
            event_queue=event_queue,
            option_symbols=sample_symbols,
        )

        assert listener.option_symbols == sample_symbols
        assert listener.event_queue is event_queue
        assert listener._client is None
        assert listener._msg_count == 0
        mock_api_key.assert_called_once_with("MASSIVE_API_KEY")

    def test_init_with_iterable_symbols(self, mock_api_key, event_queue):
        """Test initialization with iterable (generator/set) of symbols."""
        from listener.options_listener import OptionsQuoteListener

        symbols_set = {"O:AAPL250117C00180000", "O:AAPL250117P00180000"}

        listener = OptionsQuoteListener(
            event_queue=event_queue,
            option_symbols=symbols_set,
        )

        # Should be converted to list
        assert isinstance(listener.option_symbols, list)
        assert len(listener.option_symbols) == 2

    def test_build_subscriptions(self, mock_api_key, event_queue, sample_symbols):
        """Test that subscription strings are built correctly."""
        from listener.options_listener import OptionsQuoteListener

        listener = OptionsQuoteListener(
            event_queue=event_queue,
            option_symbols=sample_symbols,
        )

        subs = listener._build_subscriptions()

        assert len(subs) == 2
        assert subs[0] == "Q.O:NVDA250117C00140000"
        assert subs[1] == "Q.O:NVDA250117P00140000"

    def test_build_subscriptions_empty(self, mock_api_key, event_queue):
        """Test subscription building with empty symbols list."""
        from listener.options_listener import OptionsQuoteListener

        listener = OptionsQuoteListener(
            event_queue=event_queue,
            option_symbols=[],
        )

        subs = listener._build_subscriptions()

        assert subs == []


class TestHandleMessage:
    """Tests for message handling logic."""

    @pytest.fixture
    def mock_api_key(self):
        with patch("listener.options_listener.load_required_secret") as mock:
            mock.return_value = "test_api_key"
            yield mock

    @pytest.fixture
    def event_queue(self):
        return asyncio.Queue()

    @pytest.fixture
    def listener(self, mock_api_key, event_queue):
        from listener.options_listener import OptionsQuoteListener

        return OptionsQuoteListener(
            event_queue=event_queue,
            option_symbols=["O:NVDA250117C00140000"],
        )

    def _create_equity_quote(
        self,
        symbol: str = "O:NVDA250117C00140000",
        event_type: str = "Q",
        bid_price: float | None = 5.50,
        ask_price: float | None = 5.75,
        timestamp: int | None = None,
    ) -> EquityQuote:
        """Create an EquityQuote instance for testing."""
        if timestamp is None:
            timestamp = int(datetime(2026, 1, 5, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)
        kwargs = {
            "event_type": event_type,
            "symbol": symbol,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "timestamp": timestamp,
        }
        return EquityQuote(**kwargs)

    @pytest.mark.asyncio
    async def test_handle_valid_message(self, listener, event_queue):
        """Test handling a valid equity quote message."""
        quote = self._create_equity_quote()

        await listener._handle_message([quote])

        assert listener._msg_count == 1
        assert not event_queue.empty()

        result = event_queue.get_nowait()
        assert isinstance(result, OptionQuoteEvent)
        assert result.occ_symbol == "O:NVDA250117C00140000"
        assert result.bid == 5.50
        assert result.ask == 5.75

    @pytest.mark.asyncio
    async def test_handle_message_wrong_event_type(self, listener, event_queue):
        """Test that non-Q event types are ignored."""
        quote = self._create_equity_quote(event_type="T")  # Trade, not quote

        await listener._handle_message([quote])

        assert listener._msg_count == 0
        assert event_queue.empty()

    @pytest.mark.asyncio
    async def test_handle_message_missing_symbol(self, listener, event_queue):
        """Test that messages without symbol are ignored."""
        quote = self._create_equity_quote(symbol=None)  # type: ignore[arg-type]

        await listener._handle_message([quote])

        assert listener._msg_count == 1  # Counter still increments
        assert event_queue.empty()  # But no event pushed

    @pytest.mark.asyncio
    async def test_handle_message_missing_timestamp(self, listener, event_queue):
        """Test that messages without timestamp are ignored."""
        kwargs = {
            "event_type": "Q",
            "symbol": "O:NVDA250117C00140000",
            "bid_price": 5.50,
            "ask_price": 5.75,
            "timestamp": None,
        }
        quote = EquityQuote(**kwargs)

        await listener._handle_message([quote])

        assert listener._msg_count == 1
        assert event_queue.empty()

    @pytest.mark.asyncio
    async def test_handle_message_zero_prices(self, listener, event_queue):
        """Test handling message with None bid/ask (uses defaults)."""
        quote = self._create_equity_quote(bid_price=None, ask_price=None)

        await listener._handle_message([quote])

        assert listener._msg_count == 1
        assert not event_queue.empty()

        result = event_queue.get_nowait()
        assert result.bid == 0.0
        assert result.ask == 0.0

    @pytest.mark.asyncio
    async def test_handle_multiple_messages(self, listener, event_queue):
        """Test handling multiple messages in a batch."""
        messages = [self._create_equity_quote(symbol=f"O:NVDA250117C0014000{i}", bid_price=5.0 + i) for i in range(3)]

        await listener._handle_message(messages)

        assert listener._msg_count == 3
        assert event_queue.qsize() == 3

    @pytest.mark.asyncio
    async def test_handle_non_equity_quote_message(self, listener, event_queue):
        """Test that non-EquityQuote messages are silently ignored."""
        non_quote_msg = MagicMock()  # Not an EquityQuote instance

        await listener._handle_message([non_quote_msg])

        assert listener._msg_count == 0
        assert event_queue.empty()


class TestRunAndClose:
    """Tests for run and close methods."""

    @pytest.fixture
    def mock_api_key(self):
        with patch("listener.options_listener.load_required_secret") as mock:
            mock.return_value = "test_api_key"
            yield mock

    @pytest.fixture
    def event_queue(self):
        return asyncio.Queue()

    @pytest.mark.asyncio
    async def test_run_raises_on_empty_symbols(self, mock_api_key, event_queue):
        """Test that run() raises RuntimeError if no symbols to subscribe."""
        from listener.options_listener import OptionsQuoteListener

        listener = OptionsQuoteListener(
            event_queue=event_queue,
            option_symbols=[],
        )

        with pytest.raises(RuntimeError, match="No option symbols to subscribe"):
            await listener.run()

    @pytest.mark.asyncio
    async def test_close_without_client(self, mock_api_key, event_queue):
        """Test that close() works even if client was never initialized."""
        from listener.options_listener import OptionsQuoteListener

        listener = OptionsQuoteListener(
            event_queue=event_queue,
            option_symbols=["O:NVDA250117C00140000"],
        )

        # Should not raise
        await listener.close()

    @pytest.mark.asyncio
    async def test_close_with_client(self, mock_api_key, event_queue):
        """Test that close() calls client.close() if client exists."""
        from listener.options_listener import OptionsQuoteListener

        listener = OptionsQuoteListener(
            event_queue=event_queue,
            option_symbols=["O:NVDA250117C00140000"],
        )

        mock_client = AsyncMock()
        listener._client = mock_client

        await listener.close()

        mock_client.close.assert_called_once()


class TestFetchOptionsSymbols:
    """Tests for fetch_options_symbols function."""

    def test_fetch_with_empty_tickers(self):
        """Test that empty tickers returns empty list."""
        from listener.options_listener import fetch_options_symbols

        with patch("listener.options_listener.RESTClient"):
            result = fetch_options_symbols("test_key", [])

            assert result == []

    def test_fetch_handles_http_response(self):
        """Test that HTTPResponse result is handled gracefully."""
        from listener.options_listener import fetch_options_symbols
        from urllib3 import HTTPResponse

        with patch("listener.options_listener.RESTClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.list_options_contracts.return_value = HTTPResponse()

            result = fetch_options_symbols("test_key", ["NVDA"])

            assert result == []

    def test_fetch_handles_exception(self):
        """Test that exceptions during fetch are logged and don't crash."""
        from listener.options_listener import fetch_options_symbols

        with patch("listener.options_listener.RESTClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.list_options_contracts.side_effect = Exception("API error")

            result = fetch_options_symbols("test_key", ["NVDA"])

            assert result == []

    def test_fetch_with_contracts(self):
        """Test successful fetch of option contracts."""
        from listener.options_listener import fetch_options_symbols

        mock_contract = MagicMock()
        mock_contract.ticker = "O:NVDA250117C00140000"

        with patch("listener.options_listener.RESTClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.list_options_contracts.return_value = iter([mock_contract])

            result = fetch_options_symbols("test_key", ["NVDA"], limit_per_ticker=10)

            assert len(result) == 1
            assert result[0] == "O:NVDA250117C00140000"

    def test_fetch_respects_limit(self):
        """Test that limit_per_ticker is respected."""
        from listener.options_listener import fetch_options_symbols

        contracts = [MagicMock(ticker=f"O:NVDA2501{i:02d}C00140000") for i in range(10)]

        with patch("listener.options_listener.RESTClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.list_options_contracts.return_value = iter(contracts)

            result = fetch_options_symbols("test_key", ["NVDA"], limit_per_ticker=3)

            assert len(result) == 3
