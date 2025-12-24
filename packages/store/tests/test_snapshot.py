"""Tests for OptionSurfaceSnapshot."""

from datetime import UTC, datetime

import pytest
from domain.models import OptionPoint, OptionQuoteEvent, OptionSurfaceSnapshot
from domain.types import Symbol
from domain.utils import make_expiry_datetime
from store.option_store import OptionStore
from store.snapshot import build_surface_snapshot

# Use a valid Symbol for tests
TEST_SYMBOL: Symbol = "NVDA"
OTHER_SYMBOL: Symbol = "AAPL"
TEST_EXPIRY = make_expiry_datetime("2025-01-17")
OTHER_EXPIRY = make_expiry_datetime("2025-06-20")


class TestOptionPoint:
    """Tests for OptionPoint."""

    def test_frozen(self) -> None:
        point = OptionPoint(strike_price=100.0, option_type="call", bid=5.0, ask=5.5, mid=5.25, spread=0.5)
        with pytest.raises(AttributeError):
            point.strike_price = 200.0  # type: ignore[misc]


class TestOptionSurfaceSnapshot:
    """Tests for OptionSurfaceSnapshot."""

    @pytest.fixture
    def snapshot(self) -> OptionSurfaceSnapshot:
        calls = (
            OptionPoint(strike_price=90.0, option_type="call", bid=12.0, ask=12.5, mid=12.25, spread=0.5),
            OptionPoint(strike_price=100.0, option_type="call", bid=5.0, ask=5.5, mid=5.25, spread=0.5),
            OptionPoint(strike_price=110.0, option_type="call", bid=1.0, ask=1.5, mid=1.25, spread=0.5),
        )
        puts = (
            OptionPoint(strike_price=90.0, option_type="put", bid=1.0, ask=1.5, mid=1.25, spread=0.5),
            OptionPoint(strike_price=100.0, option_type="put", bid=4.0, ask=4.5, mid=4.25, spread=0.5),
            OptionPoint(strike_price=110.0, option_type="put", bid=10.0, ask=10.5, mid=10.25, spread=0.5),
        )
        return OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)

    def test_call_strikes(self, snapshot: OptionSurfaceSnapshot) -> None:
        assert snapshot.call_strikes == [90.0, 100.0, 110.0]

    def test_put_strikes(self, snapshot: OptionSurfaceSnapshot) -> None:
        assert snapshot.put_strikes == [90.0, 100.0, 110.0]

    def test_all_strikes(self, snapshot: OptionSurfaceSnapshot) -> None:
        assert snapshot.all_strikes == [90.0, 100.0, 110.0]

    def test_get_call_found(self, snapshot: OptionSurfaceSnapshot) -> None:
        call = snapshot.get_call(100.0)
        assert call is not None
        assert call.strike_price == 100.0
        assert call.option_type == "call"

    def test_get_call_not_found(self, snapshot: OptionSurfaceSnapshot) -> None:
        assert snapshot.get_call(999.0) is None

    def test_get_put_found(self, snapshot: OptionSurfaceSnapshot) -> None:
        put = snapshot.get_put(100.0)
        assert put is not None
        assert put.strike_price == 100.0
        assert put.option_type == "put"

    def test_get_put_not_found(self, snapshot: OptionSurfaceSnapshot) -> None:
        assert snapshot.get_put(999.0) is None


class TestBuildSurfaceSnapshot:
    """Tests for build_surface_snapshot."""

    @pytest.fixture
    def store(self) -> OptionStore:
        """Create a store with test quotes."""
        store = OptionStore()
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # NVDA calls and puts for TEST_EXPIRY
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA250117C00090000", bid=12.0, ask=12.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA250117C00100000", bid=5.0, ask=5.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA250117P00100000", bid=4.0, ask=4.5, ts=ts))

        # Different symbol - should be filtered
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:AAPL250117C00100000", bid=3.0, ask=3.5, ts=ts))

        # Different expiration - should be filtered
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA250620C00100000", bid=8.0, ask=8.5, ts=ts))

        return store

    def test_filters_by_symbol_and_expiration(self, store: OptionStore) -> None:
        snapshot = build_surface_snapshot(
            store=store,
            symbol=TEST_SYMBOL,
            expiration_date=TEST_EXPIRY,
        )

        assert snapshot.symbol == TEST_SYMBOL
        assert snapshot.expiration_date == TEST_EXPIRY
        assert len(snapshot.calls) == 2
        assert len(snapshot.puts) == 1

    def test_sorts_by_strike(self, store: OptionStore) -> None:
        snapshot = build_surface_snapshot(
            store=store,
            symbol=TEST_SYMBOL,
            expiration_date=TEST_EXPIRY,
        )

        assert snapshot.call_strikes == [90.0, 100.0]

    def test_max_spread_filter(self) -> None:
        store = OptionStore()
        ts = datetime.now(tz=UTC)

        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA250117C00100000", bid=5.0, ask=5.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA250117C00110000", bid=1.0, ask=3.0, ts=ts))  # Wide spread

        snapshot = build_surface_snapshot(
            store=store,
            symbol=TEST_SYMBOL,
            expiration_date=TEST_EXPIRY,
            max_spread=1.0,
        )

        assert len(snapshot.calls) == 1
        assert snapshot.calls[0].strike_price == 100.0

    def test_empty_result(self) -> None:
        store = OptionStore()

        snapshot = build_surface_snapshot(
            store=store,
            symbol=TEST_SYMBOL,
            expiration_date=TEST_EXPIRY,
        )

        assert len(snapshot.calls) == 0
        assert len(snapshot.puts) == 0
