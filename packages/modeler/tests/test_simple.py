"""Tests for simple probability estimator."""

import pytest
from domain.models import OptionPoint, OptionSurfaceSnapshot
from domain.types import Symbol, make_expiry_datetime
from modeler.models.simple import estimate_probability_simple

# Use a valid Symbol for tests
TEST_SYMBOL: Symbol = "NVDA"
TEST_EXPIRY = make_expiry_datetime("2025-01-17")


class TestEstimateProbabilitySimple:
    """Tests for estimate_probability_simple."""

    @pytest.fixture
    def snapshot(self) -> OptionSurfaceSnapshot:
        """ATM option with equal call/put prices -> 50% probability."""
        calls = (OptionPoint(strike=100.0, option_type="call", bid=4.8, ask=5.2, mid=5.0, spread=0.4),)
        puts = (OptionPoint(strike=100.0, option_type="put", bid=4.8, ask=5.2, mid=5.0, spread=0.4),)
        return OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)

    def test_equal_call_put_gives_50_percent(self, snapshot: OptionSurfaceSnapshot) -> None:
        result = estimate_probability_simple(snapshot=snapshot, strike=100.0)

        assert result is not None
        assert result.strike_price == 100.0
        assert result.prob_above == pytest.approx(0.5)
        assert result.prob_below == pytest.approx(0.5)

    def test_higher_call_gives_higher_prob_above(self) -> None:
        """Call worth more than put -> higher probability of finishing above strike."""
        calls = (OptionPoint(strike=100.0, option_type="call", bid=7.0, ask=8.0, mid=7.5, spread=1.0),)
        puts = (OptionPoint(strike=100.0, option_type="put", bid=2.0, ask=3.0, mid=2.5, spread=1.0),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)

        result = estimate_probability_simple(snapshot=snapshot, strike=100.0)

        assert result is not None
        # P(above) = 7.5 / (7.5 + 2.5) = 0.75
        assert result.prob_above == pytest.approx(0.75)
        assert result.prob_below == pytest.approx(0.25)

    def test_higher_put_gives_lower_prob_above(self) -> None:
        """Put worth more than call -> lower probability of finishing above strike."""
        calls = (OptionPoint(strike=100.0, option_type="call", bid=2.0, ask=3.0, mid=2.5, spread=1.0),)
        puts = (OptionPoint(strike=100.0, option_type="put", bid=7.0, ask=8.0, mid=7.5, spread=1.0),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)

        result = estimate_probability_simple(snapshot=snapshot, strike=100.0)

        assert result is not None
        # P(above) = 2.5 / (2.5 + 7.5) = 0.25
        assert result.prob_above == pytest.approx(0.25)
        assert result.prob_below == pytest.approx(0.75)

    def test_missing_call_returns_none(self) -> None:
        puts = (OptionPoint(strike=100.0, option_type="put", bid=4.8, ask=5.2, mid=5.0, spread=0.4),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=(), puts=puts)

        result = estimate_probability_simple(snapshot=snapshot, strike=100.0)
        assert result is None

    def test_missing_put_returns_none(self) -> None:
        calls = (OptionPoint(strike=100.0, option_type="call", bid=4.8, ask=5.2, mid=5.0, spread=0.4),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=())

        result = estimate_probability_simple(snapshot=snapshot, strike=100.0)
        assert result is None

    def test_strike_not_found_returns_none(self, snapshot: OptionSurfaceSnapshot) -> None:
        result = estimate_probability_simple(snapshot=snapshot, strike=999.0)
        assert result is None

    def test_zero_call_mid_returns_none(self) -> None:
        calls = (OptionPoint(strike=100.0, option_type="call", bid=0.0, ask=0.0, mid=0.0, spread=0.0),)
        puts = (OptionPoint(strike=100.0, option_type="put", bid=4.8, ask=5.2, mid=5.0, spread=0.4),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)

        result = estimate_probability_simple(snapshot=snapshot, strike=100.0)
        assert result is None

    def test_zero_put_mid_returns_none(self) -> None:
        calls = (OptionPoint(strike=100.0, option_type="call", bid=4.8, ask=5.2, mid=5.0, spread=0.4),)
        puts = (OptionPoint(strike=100.0, option_type="put", bid=0.0, ask=0.0, mid=0.0, spread=0.0),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)

        result = estimate_probability_simple(snapshot=snapshot, strike=100.0)
        assert result is None

    def test_max_spread_filter_rejects_wide_call(self) -> None:
        calls = (OptionPoint(strike=100.0, option_type="call", bid=4.0, ask=6.0, mid=5.0, spread=2.0),)
        puts = (OptionPoint(strike=100.0, option_type="put", bid=4.8, ask=5.2, mid=5.0, spread=0.4),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)

        result = estimate_probability_simple(snapshot=snapshot, strike=100.0, max_spread=1.0)
        assert result is None

    def test_max_spread_filter_rejects_wide_put(self) -> None:
        calls = (OptionPoint(strike=100.0, option_type="call", bid=4.8, ask=5.2, mid=5.0, spread=0.4),)
        puts = (OptionPoint(strike=100.0, option_type="put", bid=4.0, ask=6.0, mid=5.0, spread=2.0),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)

        result = estimate_probability_simple(snapshot=snapshot, strike=100.0, max_spread=1.0)
        assert result is None

    def test_max_spread_filter_accepts_tight_quotes(self, snapshot: OptionSurfaceSnapshot) -> None:
        result = estimate_probability_simple(snapshot=snapshot, strike=100.0, max_spread=1.0)
        assert result is not None
