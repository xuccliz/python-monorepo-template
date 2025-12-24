"""Tests for OptionSurfaceSnapshot."""

from datetime import UTC, datetime

import pytest
from domain.models import OptionPoint, OptionState, OptionSurfaceSnapshot
from domain.types import Symbol, make_expiry_datetime
from store.snapshot import build_surface_snapshot

# Use a valid Symbol for tests
TEST_SYMBOL: Symbol = "NVDA"
OTHER_SYMBOL: Symbol = "AAPL"
TEST_EXPIRY = make_expiry_datetime("2025-01-17")
OTHER_EXPIRY = make_expiry_datetime("2025-06-20")


class TestOptionPoint:
    """Tests for OptionPoint."""

    def test_frozen(self) -> None:
        point = OptionPoint(strike=100.0, option_type="call", bid=5.0, ask=5.5, mid=5.25, spread=0.5)
        with pytest.raises(AttributeError):
            point.strike = 200.0  # type: ignore[misc]


class TestOptionSurfaceSnapshot:
    """Tests for OptionSurfaceSnapshot."""

    @pytest.fixture
    def snapshot(self) -> OptionSurfaceSnapshot:
        calls = (
            OptionPoint(strike=90.0, option_type="call", bid=12.0, ask=12.5, mid=12.25, spread=0.5),
            OptionPoint(strike=100.0, option_type="call", bid=5.0, ask=5.5, mid=5.25, spread=0.5),
            OptionPoint(strike=110.0, option_type="call", bid=1.0, ask=1.5, mid=1.25, spread=0.5),
        )
        puts = (
            OptionPoint(strike=90.0, option_type="put", bid=1.0, ask=1.5, mid=1.25, spread=0.5),
            OptionPoint(strike=100.0, option_type="put", bid=4.0, ask=4.5, mid=4.25, spread=0.5),
            OptionPoint(strike=110.0, option_type="put", bid=10.0, ask=10.5, mid=10.25, spread=0.5),
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
        assert call.strike == 100.0
        assert call.option_type == "call"

    def test_get_call_not_found(self, snapshot: OptionSurfaceSnapshot) -> None:
        assert snapshot.get_call(999.0) is None

    def test_get_put_found(self, snapshot: OptionSurfaceSnapshot) -> None:
        put = snapshot.get_put(100.0)
        assert put is not None
        assert put.strike == 100.0
        assert put.option_type == "put"

    def test_get_put_not_found(self, snapshot: OptionSurfaceSnapshot) -> None:
        assert snapshot.get_put(999.0) is None


class TestBuildSurfaceSnapshot:
    """Tests for build_surface_snapshot."""

    @pytest.fixture
    def states(self) -> list[OptionState]:
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        return [
            OptionState(
                occ_symbol="O:NVDA250117C00090000",
                symbol=TEST_SYMBOL,
                strike_price=90.0,
                expiration_date=TEST_EXPIRY,
                option_type="call",
                bid=12.0,
                ask=12.5,
                mid=12.25,
                spread=0.5,
                last_updated=ts,
            ),
            OptionState(
                occ_symbol="O:NVDA250117C00100000",
                symbol=TEST_SYMBOL,
                strike_price=100.0,
                expiration_date=TEST_EXPIRY,
                option_type="call",
                bid=5.0,
                ask=5.5,
                mid=5.25,
                spread=0.5,
                last_updated=ts,
            ),
            OptionState(
                occ_symbol="O:NVDA250117P00100000",
                symbol=TEST_SYMBOL,
                strike_price=100.0,
                expiration_date=TEST_EXPIRY,
                option_type="put",
                bid=4.0,
                ask=4.5,
                mid=4.25,
                spread=0.5,
                last_updated=ts,
            ),
            # Different symbol - should be filtered
            OptionState(
                occ_symbol="O:AAPL250117C00100000",
                symbol=OTHER_SYMBOL,
                strike_price=100.0,
                expiration_date=TEST_EXPIRY,
                option_type="call",
                bid=3.0,
                ask=3.5,
                mid=3.25,
                spread=0.5,
                last_updated=ts,
            ),
            # Different expiration_date - should be filtered
            OptionState(
                occ_symbol="O:NVDA250620C00100000",
                symbol=TEST_SYMBOL,
                strike_price=100.0,
                expiration_date=OTHER_EXPIRY,
                option_type="call",
                bid=8.0,
                ask=8.5,
                mid=8.25,
                spread=0.5,
                last_updated=ts,
            ),
        ]

    def test_filters_by_symbol_and_expiration(self, states: list[OptionState]) -> None:
        snapshot = build_surface_snapshot(
            states=states,
            symbol=TEST_SYMBOL,
            expiration_date=TEST_EXPIRY,
        )

        assert snapshot.symbol == TEST_SYMBOL
        assert snapshot.expiration_date == TEST_EXPIRY
        assert len(snapshot.calls) == 2
        assert len(snapshot.puts) == 1

    def test_sorts_by_strike(self, states: list[OptionState]) -> None:
        snapshot = build_surface_snapshot(
            states=states,
            symbol=TEST_SYMBOL,
            expiration_date=TEST_EXPIRY,
        )

        assert snapshot.call_strikes == [90.0, 100.0]

    def test_max_spread_filter(self) -> None:
        ts = datetime.now(tz=UTC)
        states = [
            OptionState(
                occ_symbol="O:NVDA250117C00100000",
                symbol=TEST_SYMBOL,
                strike_price=100.0,
                expiration_date=TEST_EXPIRY,
                option_type="call",
                bid=5.0,
                ask=5.5,
                mid=5.25,
                spread=0.5,
                last_updated=ts,
            ),
            OptionState(
                occ_symbol="O:NVDA250117C00110000",
                symbol=TEST_SYMBOL,
                strike_price=110.0,
                expiration_date=TEST_EXPIRY,
                option_type="call",
                bid=1.0,
                ask=3.0,
                mid=2.0,
                spread=2.0,  # Wide spread
                last_updated=ts,
            ),
        ]

        snapshot = build_surface_snapshot(
            states=states,
            symbol=TEST_SYMBOL,
            expiration_date=TEST_EXPIRY,
            max_spread=1.0,
        )

        assert len(snapshot.calls) == 1
        assert snapshot.calls[0].strike == 100.0

    def test_empty_result(self) -> None:
        snapshot = build_surface_snapshot(
            states=[],
            symbol=TEST_SYMBOL,
            expiration_date=TEST_EXPIRY,
        )

        assert len(snapshot.calls) == 0
        assert len(snapshot.puts) == 0
