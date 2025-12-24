"""Tests for slope-based probability estimator."""

import pytest
from domain.models import OptionPoint, OptionSurfaceSnapshot
from domain.types import Symbol, make_expiry_datetime
from modeler.models.slope import estimate_probability_slope

# Use a valid Symbol for tests
TEST_SYMBOL: Symbol = "NVDA"
TEST_EXPIRY = make_expiry_datetime("2025-01-17")


class TestEstimateProbabilitySlope:
    """Tests for estimate_probability_slope."""

    @pytest.fixture
    def snapshot(self) -> OptionSurfaceSnapshot:
        """
        Call prices decreasing with strike (normal behavior).
        Slope = (1.0 - 9.0) / (110 - 90) = -8/20 = -0.4
        P(above) = -(-0.4) / 1.0 = 0.4
        """
        calls = (
            OptionPoint(strike=90.0, option_type="call", bid=8.5, ask=9.5, mid=9.0, spread=1.0),
            OptionPoint(strike=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike=110.0, option_type="call", bid=0.5, ask=1.5, mid=1.0, spread=1.0),
        )
        return OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=())

    def test_basic_slope_calculation(self, snapshot: OptionSurfaceSnapshot) -> None:
        result = estimate_probability_slope(snapshot=snapshot, strike=100.0)

        assert result is not None
        assert result.strike_price == 100.0
        # Slope = (1.0 - 9.0) / (110 - 90) = -0.4
        # P(above) = 0.4
        assert result.prob_above == pytest.approx(0.4)
        assert result.prob_below == pytest.approx(0.6)

    def test_with_discount_factor(self, snapshot: OptionSurfaceSnapshot) -> None:
        # With discount = 0.95, P(above) = 0.4 / 0.95 ≈ 0.421
        result = estimate_probability_slope(snapshot=snapshot, strike=100.0, discount=0.95)

        assert result is not None
        assert result.prob_above == pytest.approx(0.4 / 0.95, rel=0.01)

    def test_window_2(self) -> None:
        """Test with window=2 (uses 2 strikes on each side)."""
        calls = (
            OptionPoint(strike=80.0, option_type="call", bid=14.5, ask=15.5, mid=15.0, spread=1.0),
            OptionPoint(strike=90.0, option_type="call", bid=9.5, ask=10.5, mid=10.0, spread=1.0),
            OptionPoint(strike=100.0, option_type="call", bid=5.5, ask=6.5, mid=6.0, spread=1.0),
            OptionPoint(strike=110.0, option_type="call", bid=2.5, ask=3.5, mid=3.0, spread=1.0),
            OptionPoint(strike=120.0, option_type="call", bid=0.5, ask=1.5, mid=1.0, spread=1.0),
        )
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=())

        result = estimate_probability_slope(snapshot=snapshot, strike=100.0, window=2)

        assert result is not None
        # Slope = (1.0 - 15.0) / (120 - 80) = -14/40 = -0.35
        assert result.prob_above == pytest.approx(0.35)

    def test_insufficient_calls_returns_none(self) -> None:
        calls = (OptionPoint(strike=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=())

        result = estimate_probability_slope(snapshot=snapshot, strike=100.0)
        assert result is None

    def test_strike_at_edge_returns_none(self) -> None:
        """Strike at edge of chain can't compute slope."""
        calls = (
            OptionPoint(strike=90.0, option_type="call", bid=8.5, ask=9.5, mid=9.0, spread=1.0),
            OptionPoint(strike=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike=110.0, option_type="call", bid=0.5, ask=1.5, mid=1.0, spread=1.0),
        )
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=())

        # Strike at left edge
        result = estimate_probability_slope(snapshot=snapshot, strike=90.0)
        assert result is None

        # Strike at right edge
        result = estimate_probability_slope(snapshot=snapshot, strike=110.0)
        assert result is None

    def test_max_spread_filter(self) -> None:
        calls = (
            OptionPoint(strike=90.0, option_type="call", bid=8.0, ask=10.0, mid=9.0, spread=2.0),  # Wide
            OptionPoint(strike=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike=110.0, option_type="call", bid=0.5, ask=1.5, mid=1.0, spread=1.0),
        )
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=())

        result = estimate_probability_slope(snapshot=snapshot, strike=100.0, max_spread=1.5)
        assert result is None

    def test_clamps_probability_to_valid_range(self) -> None:
        """Extreme slopes should be clamped to [0, 1]."""
        # Steep slope that would give prob > 1
        calls = (
            OptionPoint(strike=99.0, option_type="call", bid=9.5, ask=10.5, mid=10.0, spread=1.0),
            OptionPoint(strike=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike=101.0, option_type="call", bid=0.0, ask=0.0, mid=0.0, spread=0.0),
        )
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=())

        result = estimate_probability_slope(snapshot=snapshot, strike=100.0)

        assert result is not None
        assert 0.0 <= result.prob_above <= 1.0
        prob_below = result.prob_below
        assert prob_below is not None
        assert 0.0 <= prob_below <= 1.0

    def test_empty_calls_returns_none(self) -> None:
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=(), puts=())

        result = estimate_probability_slope(snapshot=snapshot, strike=100.0)
        assert result is None
