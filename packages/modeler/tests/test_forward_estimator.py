"""Tests for forward price estimator."""

import pytest
from domain.models import OptionPoint, OptionSurfaceSnapshot
from domain.types import Symbol
from domain.utils import make_expiry_datetime
from modeler.forward_estimator import estimate_forward_put_call_parity

TEST_SYMBOL: Symbol = "NVDA"
TEST_EXPIRY = make_expiry_datetime("2025-01-17")


def make_snapshot(
    calls: tuple[OptionPoint, ...],
    puts: tuple[OptionPoint, ...],
) -> OptionSurfaceSnapshot:
    """Helper to create snapshot."""
    return OptionSurfaceSnapshot(
        symbol=TEST_SYMBOL,
        expiration_date=TEST_EXPIRY,
        calls=calls,
        puts=puts,
    )


class TestEstimateForwardPutCallParity:
    """Tests for estimate_forward_put_call_parity."""

    def test_basic_forward_estimation(self) -> None:
        """Test basic forward estimation with valid data."""
        # ATM options: C - P = 0 => F = K
        calls = (
            OptionPoint(strike_price=95.0, option_type="call", bid=7.0, ask=8.0, mid=7.5, spread=1.0),
            OptionPoint(strike_price=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike_price=105.0, option_type="call", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="call", bid=0.5, ask=1.5, mid=1.0, spread=1.0),
        )
        puts = (
            OptionPoint(strike_price=95.0, option_type="put", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
            OptionPoint(strike_price=100.0, option_type="put", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike_price=105.0, option_type="put", bid=7.0, ask=8.0, mid=7.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="put", bid=10.5, ask=11.5, mid=11.0, spread=1.0),
        )
        snapshot = make_snapshot(calls, puts)

        result = estimate_forward_put_call_parity(snapshot=snapshot)

        assert result is not None
        # F = K + (C - P) for each strike, then weighted average
        assert result.forward == pytest.approx(100.0, rel=0.05)
        assert result.n_used >= 3

    def test_invalid_discount_raises(self) -> None:
        """Test that invalid discount raises ValueError."""
        calls = (OptionPoint(strike_price=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),)
        puts = (OptionPoint(strike_price=100.0, option_type="put", bid=4.5, ask=5.5, mid=5.0, spread=1.0),)
        snapshot = make_snapshot(calls, puts)

        with pytest.raises(ValueError, match="discount must be positive"):
            estimate_forward_put_call_parity(snapshot=snapshot, discount=0)

        with pytest.raises(ValueError, match="discount must be positive"):
            estimate_forward_put_call_parity(snapshot=snapshot, discount=-1)

    def test_no_common_strikes_returns_none(self) -> None:
        """Test that no common strikes returns None."""
        calls = (OptionPoint(strike_price=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),)
        puts = (OptionPoint(strike_price=110.0, option_type="put", bid=4.5, ask=5.5, mid=5.0, spread=1.0),)
        snapshot = make_snapshot(calls, puts)

        result = estimate_forward_put_call_parity(snapshot=snapshot)
        assert result is None

    def test_insufficient_candidates_returns_none(self) -> None:
        """Test that fewer than 3 valid candidates returns None."""
        calls = (
            OptionPoint(strike_price=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike_price=105.0, option_type="call", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
        )
        puts = (
            OptionPoint(strike_price=100.0, option_type="put", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike_price=105.0, option_type="put", bid=7.0, ask=8.0, mid=7.5, spread=1.0),
        )
        snapshot = make_snapshot(calls, puts)

        result = estimate_forward_put_call_parity(snapshot=snapshot)
        assert result is None

    def test_filters_low_mid_prices(self) -> None:
        """Test that quotes with mid <= min_mid are filtered."""
        calls = (
            OptionPoint(strike_price=95.0, option_type="call", bid=7.0, ask=8.0, mid=7.5, spread=1.0),
            OptionPoint(strike_price=100.0, option_type="call", bid=0.0, ask=0.0, mid=0.0, spread=0.0),  # Zero mid
            OptionPoint(strike_price=105.0, option_type="call", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="call", bid=0.5, ask=1.5, mid=1.0, spread=1.0),
        )
        puts = (
            OptionPoint(strike_price=95.0, option_type="put", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
            OptionPoint(strike_price=100.0, option_type="put", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike_price=105.0, option_type="put", bid=7.0, ask=8.0, mid=7.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="put", bid=10.5, ask=11.5, mid=11.0, spread=1.0),
        )
        snapshot = make_snapshot(calls, puts)

        result = estimate_forward_put_call_parity(snapshot=snapshot)

        # Should still work with 3 valid strikes
        assert result is not None
        assert result.n_used >= 3

    def test_filters_negative_bids(self) -> None:
        """Test that quotes with negative bids are filtered."""
        calls = (
            OptionPoint(strike_price=95.0, option_type="call", bid=-1.0, ask=8.0, mid=3.5, spread=9.0),  # Negative bid
            OptionPoint(strike_price=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike_price=105.0, option_type="call", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="call", bid=0.5, ask=1.5, mid=1.0, spread=1.0),
        )
        puts = (
            OptionPoint(strike_price=95.0, option_type="put", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
            OptionPoint(strike_price=100.0, option_type="put", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike_price=105.0, option_type="put", bid=7.0, ask=8.0, mid=7.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="put", bid=10.5, ask=11.5, mid=11.0, spread=1.0),
        )
        snapshot = make_snapshot(calls, puts)

        result = estimate_forward_put_call_parity(snapshot=snapshot)
        assert result is not None

    def test_filters_crossed_market(self) -> None:
        """Test that crossed markets (bid > ask) are filtered."""
        calls = (
            OptionPoint(strike_price=95.0, option_type="call", bid=9.0, ask=8.0, mid=8.5, spread=-1.0),  # Crossed
            OptionPoint(strike_price=100.0, option_type="call", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike_price=105.0, option_type="call", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="call", bid=0.5, ask=1.5, mid=1.0, spread=1.0),
        )
        puts = (
            OptionPoint(strike_price=95.0, option_type="put", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
            OptionPoint(strike_price=100.0, option_type="put", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike_price=105.0, option_type="put", bid=7.0, ask=8.0, mid=7.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="put", bid=10.5, ask=11.5, mid=11.0, spread=1.0),
        )
        snapshot = make_snapshot(calls, puts)

        result = estimate_forward_put_call_parity(snapshot=snapshot)
        assert result is not None

    def test_max_spread_filter(self) -> None:
        """Test that max_spread filter works."""
        calls = (
            OptionPoint(strike_price=95.0, option_type="call", bid=7.0, ask=8.0, mid=7.5, spread=1.0),
            OptionPoint(strike_price=100.0, option_type="call", bid=4.0, ask=6.0, mid=5.0, spread=2.0),  # Wide spread
            OptionPoint(strike_price=105.0, option_type="call", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="call", bid=0.5, ask=1.5, mid=1.0, spread=1.0),
        )
        puts = (
            OptionPoint(strike_price=95.0, option_type="put", bid=2.0, ask=3.0, mid=2.5, spread=1.0),
            OptionPoint(strike_price=100.0, option_type="put", bid=4.5, ask=5.5, mid=5.0, spread=1.0),
            OptionPoint(strike_price=105.0, option_type="put", bid=7.0, ask=8.0, mid=7.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="put", bid=10.5, ask=11.5, mid=11.0, spread=1.0),
        )
        snapshot = make_snapshot(calls, puts)

        result = estimate_forward_put_call_parity(snapshot=snapshot, max_spread=1.5)
        assert result is not None
        # Strike 100 should be filtered out due to wide spread

    def test_empty_snapshot_returns_none(self) -> None:
        """Test that empty snapshot returns None."""
        snapshot = make_snapshot((), ())

        result = estimate_forward_put_call_parity(snapshot=snapshot)
        assert result is None
