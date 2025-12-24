"""Tests for EventStore."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from domain.models import EventMetadata, MarketMetadata
from store.event_store import EventReader, EventStore


class TestEventStore:
    """Tests for EventStore."""

    @pytest.fixture
    def mock_events(self) -> list[EventMetadata]:
        """Sample events for testing."""
        return [
            EventMetadata(
                symbol="NVDA",
                question="Will NVIDIA (NVDA) finish week of January 5 above___?",
                question_id="event1",
                end_date="2026-01-10T00:00:00Z",
                markets=[
                    MarketMetadata(
                        question="Will NVIDIA (NVDA) finish week of January 5 above $140?",
                        question_id="market1",
                        strike_price=140.0,
                        yes_token_id="yes1",
                        no_token_id="no1",
                    )
                ],
            ),
            EventMetadata(
                symbol="NVDA",
                question="Will NVIDIA (NVDA) finish week of January 12 above___?",
                question_id="event1b",
                end_date="2026-01-17T00:00:00Z",
                markets=[
                    MarketMetadata(
                        question="Will NVIDIA (NVDA) finish week of January 12 above $145?",
                        question_id="market1b",
                        strike_price=145.0,
                        yes_token_id="yes1b",
                        no_token_id="no1b",
                    )
                ],
            ),
            EventMetadata(
                symbol="AAPL",
                question="Will Apple (AAPL) finish week of January 5 above___?",
                question_id="event2",
                end_date="2026-01-10T00:00:00Z",
                markets=[],
            ),
            EventMetadata(
                symbol="TSLA",
                question="Will Tesla (TSLA) finish week of January 5 above___?",
                question_id="event3",
                end_date="2026-01-10T00:00:00Z",
                markets=[],
            ),
        ]

    def test_empty_store(self) -> None:
        store = EventStore()

        assert len(store.get_all()) == 0
        assert store.get_all() == []
        assert store.last_refresh() is None

    def test_get_by_symbol_returns_empty_list_for_empty(self) -> None:
        store = EventStore()

        assert store.get_by_symbol("NVDA") == []

    @patch("store.event_store.fetch_stock_events")
    def test_refresh_populates_store(self, mock_fetch: MagicMock, mock_events: list[EventMetadata]) -> None:
        mock_fetch.return_value = mock_events
        store = EventStore()

        count = store.refresh()

        assert count == 4
        assert len(store.get_all()) == 4
        assert store.last_refresh() is not None

    @patch("store.event_store.fetch_stock_events")
    def test_get_by_symbol_returns_all_events_for_symbol(
        self, mock_fetch: MagicMock, mock_events: list[EventMetadata]
    ) -> None:
        mock_fetch.return_value = mock_events
        store = EventStore()
        store.refresh()

        events = store.get_by_symbol("NVDA")

        assert len(events) == 2
        assert all(e.symbol == "NVDA" for e in events)
        question_ids = {e.question_id for e in events}
        assert question_ids == {"event1", "event1b"}

    @patch("store.event_store.fetch_stock_events")
    def test_get_by_symbol_returns_empty_list_for_missing(
        self, mock_fetch: MagicMock, mock_events: list[EventMetadata]
    ) -> None:
        mock_fetch.return_value = mock_events
        store = EventStore()
        store.refresh()

        assert store.get_by_symbol("MSFT") == []

    @patch("store.event_store.fetch_stock_events")
    def test_get_all_returns_all_events(self, mock_fetch: MagicMock, mock_events: list[EventMetadata]) -> None:
        mock_fetch.return_value = mock_events
        store = EventStore()
        store.refresh()

        all_events = store.get_all()

        assert len(all_events) == 4
        symbols = {e.symbol for e in all_events}
        assert symbols == {"NVDA", "AAPL", "TSLA"}

    @patch("store.event_store.fetch_stock_events")
    def test_clear_removes_all_events(self, mock_fetch: MagicMock, mock_events: list[EventMetadata]) -> None:
        mock_fetch.return_value = mock_events
        store = EventStore()
        store.refresh()

        store.clear()

        assert store.get_all() == []
        assert store.last_refresh() is None

    @patch("store.event_store.fetch_stock_events")
    def test_refresh_replaces_old_events(self, mock_fetch: MagicMock) -> None:
        # First refresh
        mock_fetch.return_value = [
            EventMetadata(symbol="NVDA", question_id="old1"),
        ]
        store = EventStore()
        store.refresh()

        # Second refresh with different events
        mock_fetch.return_value = [
            EventMetadata(symbol="AAPL", question_id="new1"),
            EventMetadata(symbol="TSLA", question_id="new2"),
        ]
        store.refresh()

        assert len(store.get_all()) == 2
        assert store.get_by_symbol("NVDA") == []
        assert len(store.get_by_symbol("AAPL")) == 1

    @patch("store.event_store.fetch_stock_events")
    def test_refresh_skips_events_without_symbol(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = [
            EventMetadata(symbol="NVDA", question_id="1"),
            EventMetadata(symbol=None, question_id="2"),  # No symbol
            EventMetadata(symbol="AAPL", question_id="3"),
        ]
        store = EventStore()

        count = store.refresh()

        assert count == 2
        assert len(store.get_all()) == 2

    @patch("store.event_store.fetch_stock_events")
    def test_last_refresh_updates_on_refresh(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = []
        store = EventStore()

        before = datetime.now(UTC)
        store.refresh()
        after = datetime.now(UTC)

        refresh_time = store.last_refresh()
        assert refresh_time is not None
        assert before <= refresh_time <= after


class TestEventReaderProtocol:
    """Tests that EventStore satisfies EventReader protocol."""

    def test_event_store_is_event_reader(self) -> None:
        """EventStore should satisfy the EventReader protocol."""
        store = EventStore()

        # This function accepts EventReader
        def use_reader(reader: EventReader) -> int:
            return len(reader.get_all())

        # EventStore should work as EventReader
        result = use_reader(store)
        assert result == 0


class TestGetMarket:
    """Tests for EventStore.get_market()."""

    @pytest.fixture
    def store_with_markets(self) -> EventStore:
        """Store with events containing markets."""
        store = EventStore()
        store._events = {
            "NVDA": [
                EventMetadata(
                    symbol="NVDA",
                    question="Will NVIDIA finish week of Jan 5 above___?",
                    end_date="2026-01-10T00:00:00Z",
                    markets=[
                        MarketMetadata(strike_price=140.0, yes_price=0.65, no_price=0.35),
                        MarketMetadata(strike_price=150.0, yes_price=0.40, no_price=0.60),
                    ],
                ),
                EventMetadata(
                    symbol="NVDA",
                    question="Will NVIDIA finish week of Jan 12 above___?",
                    end_date="2026-01-17T00:00:00Z",
                    markets=[
                        MarketMetadata(strike_price=145.0, yes_price=0.55, no_price=0.45),
                    ],
                ),
            ],
            "AAPL": [
                EventMetadata(
                    symbol="AAPL",
                    question="Will Apple finish above___?",
                    end_date="2026-01-10T00:00:00Z",
                    markets=[],
                ),
            ],
        }
        return store

    def test_get_market_returns_matching_market(self, store_with_markets: EventStore) -> None:
        market = store_with_markets.get_market("NVDA", "2026-01-10T00:00:00Z", 140.0)

        assert market is not None
        assert market.strike_price == 140.0
        assert market.yes_price == 0.65

    def test_get_market_returns_market_for_different_end_date(self, store_with_markets: EventStore) -> None:
        market = store_with_markets.get_market("NVDA", "2026-01-17T00:00:00Z", 145.0)

        assert market is not None
        assert market.strike_price == 145.0
        assert market.yes_price == 0.55

    def test_get_market_returns_none_for_wrong_end_date(self, store_with_markets: EventStore) -> None:
        market = store_with_markets.get_market("NVDA", "2026-01-17T00:00:00Z", 140.0)

        assert market is None

    def test_get_market_returns_none_for_missing_strike(self, store_with_markets: EventStore) -> None:
        market = store_with_markets.get_market("NVDA", "2026-01-10T00:00:00Z", 999.0)

        assert market is None

    def test_get_market_returns_none_for_missing_symbol(self, store_with_markets: EventStore) -> None:
        market = store_with_markets.get_market("MSFT", "2026-01-10T00:00:00Z", 140.0)

        assert market is None

    def test_get_market_returns_none_for_empty_markets(self, store_with_markets: EventStore) -> None:
        market = store_with_markets.get_market("AAPL", "2026-01-10T00:00:00Z", 200.0)

        assert market is None

    def test_get_market_returns_none_for_empty_store(self) -> None:
        store = EventStore()

        market = store.get_market("NVDA", "2026-01-10T00:00:00Z", 140.0)

        assert market is None


class TestGetPolymarketProb:
    """Tests for EventStore.get_polymarket_prob()."""

    @pytest.fixture
    def store_with_markets(self) -> EventStore:
        """Store with events containing markets."""
        store = EventStore()
        store._events = {
            "NVDA": [
                EventMetadata(
                    symbol="NVDA",
                    question="Will NVIDIA finish above___?",
                    end_date="2026-01-10T00:00:00Z",
                    markets=[
                        MarketMetadata(strike_price=140.0, yes_price=0.65, no_price=0.35),
                        MarketMetadata(strike_price=150.0, yes_price=0.40, no_price=0.60),
                    ],
                ),
            ],
        }
        return store

    def test_get_polymarket_prob_above_returns_yes_price(self, store_with_markets: EventStore) -> None:
        prob = store_with_markets.get_polymarket_prob("NVDA", "2026-01-10T00:00:00Z", 140.0, direction="above")

        assert prob == 0.65

    def test_get_polymarket_prob_below_returns_no_price(self, store_with_markets: EventStore) -> None:
        prob = store_with_markets.get_polymarket_prob("NVDA", "2026-01-10T00:00:00Z", 140.0, direction="below")

        assert prob == 0.35

    def test_get_polymarket_prob_none_direction_returns_no_price(self, store_with_markets: EventStore) -> None:
        prob = store_with_markets.get_polymarket_prob("NVDA", "2026-01-10T00:00:00Z", 140.0, direction=None)

        assert prob == 0.35

    def test_get_polymarket_prob_default_direction_returns_no_price(self, store_with_markets: EventStore) -> None:
        prob = store_with_markets.get_polymarket_prob("NVDA", "2026-01-10T00:00:00Z", 140.0)

        assert prob == 0.35

    def test_get_polymarket_prob_returns_none_for_missing_market(self, store_with_markets: EventStore) -> None:
        prob = store_with_markets.get_polymarket_prob("NVDA", "2026-01-10T00:00:00Z", 999.0, direction="above")

        assert prob is None

    def test_get_polymarket_prob_returns_none_for_missing_symbol(self, store_with_markets: EventStore) -> None:
        prob = store_with_markets.get_polymarket_prob("MSFT", "2026-01-10T00:00:00Z", 140.0, direction="above")

        assert prob is None
