"""Tests for OptionStore."""

from datetime import UTC, datetime

import pytest
from domain.models import OptionQuoteEvent
from domain.utils import make_expiry_datetime
from store.option_store import OptionStore, parse_occ_symbol


class TestParseOccSymbol:
    """Tests for OCC symbol parsing."""

    def test_valid_call_symbol(self) -> None:
        result = parse_occ_symbol("O:NVDA260117C00140000")
        assert result is not None
        assert result.symbol == "NVDA"
        assert result.expiration_date == make_expiry_datetime("2026-01-17")
        assert result.option_type == "call"
        assert result.strike_price == 140.0

    def test_valid_put_symbol(self) -> None:
        result = parse_occ_symbol("O:AAPL250620P00200000")
        assert result is not None
        assert result.symbol == "AAPL"
        assert result.expiration_date == make_expiry_datetime("2025-06-20")
        assert result.option_type == "put"
        assert result.strike_price == 200.0

    def test_fractional_strike(self) -> None:
        result = parse_occ_symbol("O:NVDA250117C00450500")
        assert result is not None
        assert result.strike_price == 450.5

    def test_invalid_symbol_no_prefix(self) -> None:
        result = parse_occ_symbol("NVDA260117C00140000")
        assert result is None

    def test_invalid_symbol_bad_format(self) -> None:
        result = parse_occ_symbol("O:INVALID")
        assert result is None

    def test_invalid_symbol_empty(self) -> None:
        result = parse_occ_symbol("")
        assert result is None


class TestOptionStore:
    """Tests for OptionStore."""

    @pytest.fixture
    def store(self) -> OptionStore:
        return OptionStore()

    @pytest.fixture
    def valid_quote(self) -> OptionQuoteEvent:
        return OptionQuoteEvent(
            occ_symbol="O:NVDA260117C00140000",
            bid=5.0,
            ask=5.5,
            ts=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        )

    def test_apply_quote_creates_state(self, store: OptionStore, valid_quote: OptionQuoteEvent) -> None:
        state = store.apply_quote(valid_quote)

        assert state is not None
        assert state.occ_symbol == "O:NVDA260117C00140000"
        assert state.symbol == "NVDA"
        assert state.strike_price == 140.0
        assert state.expiration_date == make_expiry_datetime("2026-01-17")
        assert state.option_type == "call"
        assert state.bid == 5.0
        assert state.ask == 5.5
        assert state.mid == 5.25
        assert state.spread == 0.5

    def test_apply_quote_updates_existing(self, store: OptionStore) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        quote1 = OptionQuoteEvent(occ_symbol="O:NVDA260117C00140000", bid=5.0, ask=5.5, ts=ts)
        quote2 = OptionQuoteEvent(occ_symbol="O:NVDA260117C00140000", bid=6.0, ask=6.5, ts=ts)

        store.apply_quote(quote1)
        state = store.apply_quote(quote2)

        assert state is not None
        assert state.bid == 6.0
        assert state.ask == 6.5
        assert store.count() == 1

    def test_apply_quote_rejects_invalid_symbol(self, store: OptionStore) -> None:
        quote = OptionQuoteEvent(
            occ_symbol="INVALID",
            bid=5.0,
            ask=5.5,
            ts=datetime.now(tz=UTC),
        )
        state = store.apply_quote(quote)
        assert state is None
        assert store.count() == 0

    def test_apply_quote_rejects_negative_bid(self, store: OptionStore) -> None:
        quote = OptionQuoteEvent(
            occ_symbol="O:NVDA260117C00140000",
            bid=-1.0,
            ask=5.5,
            ts=datetime.now(tz=UTC),
        )
        state = store.apply_quote(quote)
        assert state is None

    def test_apply_quote_rejects_negative_ask(self, store: OptionStore) -> None:
        quote = OptionQuoteEvent(
            occ_symbol="O:NVDA260117C00140000",
            bid=5.0,
            ask=-1.0,
            ts=datetime.now(tz=UTC),
        )
        state = store.apply_quote(quote)
        assert state is None

    def test_apply_quote_rejects_crossed_market(self, store: OptionStore) -> None:
        quote = OptionQuoteEvent(
            occ_symbol="O:NVDA260117C00140000",
            bid=6.0,
            ask=5.0,
            ts=datetime.now(tz=UTC),
        )
        state = store.apply_quote(quote)
        assert state is None

    def test_get_returns_none_for_missing(self, store: OptionStore) -> None:
        assert store.get("O:MISSING") is None

    def test_get_returns_state(self, store: OptionStore, valid_quote: OptionQuoteEvent) -> None:
        store.apply_quote(valid_quote)
        state = store.get(valid_quote.occ_symbol)
        assert state is not None
        assert state.occ_symbol == valid_quote.occ_symbol

    def test_get_by_symbol(self, store: OptionStore) -> None:
        ts = datetime.now(tz=UTC)
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260117C00140000", bid=5.0, ask=5.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260117P00140000", bid=3.0, ask=3.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:AAPL250117C00200000", bid=2.0, ask=2.5, ts=ts))

        nvda_states = store.get_by_symbol("NVDA")
        assert len(nvda_states) == 2
        assert all(s.symbol == "NVDA" for s in nvda_states)

    def test_get_pair(self, store: OptionStore) -> None:
        ts = datetime.now(tz=UTC)
        expiry = make_expiry_datetime("2026-01-17")
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260117C00140000", bid=5.0, ask=5.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260117P00140000", bid=3.0, ask=3.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260117C00150000", bid=2.0, ask=2.5, ts=ts))

        call, put = store.get_pair("NVDA", expiry, 140.0)
        assert call is not None
        assert put is not None
        assert call.strike_price == 140.0
        assert put.strike_price == 140.0

    def test_get_strikes(self, store: OptionStore) -> None:
        ts = datetime.now(tz=UTC)
        expiry = make_expiry_datetime("2026-01-17")
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260117C00140000", bid=5.0, ask=5.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260117C00150000", bid=3.0, ask=3.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260117C00130000", bid=7.0, ask=7.5, ts=ts))

        strikes = store.get_strikes("NVDA", expiry)
        assert strikes == [130.0, 140.0, 150.0]

    def test_clear(self, store: OptionStore, valid_quote: OptionQuoteEvent) -> None:
        store.apply_quote(valid_quote)
        assert store.count() == 1
        store.clear()
        assert store.count() == 0

    def test_get_by_symbol_returns_empty_for_missing(self, store: OptionStore) -> None:
        """Test get_by_symbol returns empty list for missing symbol."""
        result = store.get_by_symbol("MSFT")
        assert result == []

    def test_get_expiries(self, store: OptionStore) -> None:
        """Test get_expiries returns sorted expiries for a symbol."""
        ts = datetime.now(tz=UTC)
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260117C00140000", bid=5.0, ask=5.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260217C00140000", bid=5.0, ask=5.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA251217C00140000", bid=5.0, ask=5.5, ts=ts))

        expiries = store.get_expiries("NVDA")

        assert len(expiries) == 3
        # Should be sorted
        assert expiries[0] < expiries[1] < expiries[2]

    def test_get_expiries_returns_empty_for_missing_symbol(self, store: OptionStore) -> None:
        """Test get_expiries returns empty list for missing symbol."""
        result = store.get_expiries("MSFT")
        assert result == []

    def test_get_expiration_dates(self, store: OptionStore) -> None:
        """Test get_expiration_dates returns all expiries across symbols."""
        ts = datetime.now(tz=UTC)
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260117C00140000", bid=5.0, ask=5.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:AAPL260117C00200000", bid=5.0, ask=5.5, ts=ts))
        store.apply_quote(OptionQuoteEvent(occ_symbol="O:NVDA260217C00140000", bid=5.0, ask=5.5, ts=ts))

        expiration_dates = store.get_expiration_dates()

        assert len(expiration_dates) == 2  # Two unique expiries
        assert make_expiry_datetime("2026-01-17") in expiration_dates
        assert make_expiry_datetime("2026-02-17") in expiration_dates

    def test_get_pair_returns_none_for_missing(self, store: OptionStore) -> None:
        """Test get_pair returns (None, None) for missing data."""
        expiry = make_expiry_datetime("2026-01-17")

        call, put = store.get_pair("NVDA", expiry, 140.0)

        assert call is None
        assert put is None

    def test_get_strikes_returns_empty_for_missing(self, store: OptionStore) -> None:
        """Test get_strikes returns empty list for missing symbol/expiry."""
        expiry = make_expiry_datetime("2026-01-17")

        strikes = store.get_strikes("NVDA", expiry)

        assert strikes == []


class TestParseOccSymbolUnknownSymbol:
    """Tests for OCC symbol parsing with unknown symbols."""

    def test_unknown_symbol_returns_none(self) -> None:
        """Test that unknown symbols (not in SYMBOLS) return None."""
        # XYZ is not in the SYMBOLS literal type
        result = parse_occ_symbol("O:XYZ260117C00140000")
        assert result is None
