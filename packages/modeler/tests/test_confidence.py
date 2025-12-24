"""Tests for confidence score computation."""

import pytest
from domain.models import OptionPoint, OptionSurfaceSnapshot, StrikeProbability
from domain.types import Symbol, make_expiry_datetime
from modeler.confidence_score import compute_confidence

# Use a valid Symbol for tests
TEST_SYMBOL: Symbol = "NVDA"
TEST_EXPIRY = make_expiry_datetime("2025-01-17")


class TestComputeConfidence:
    """Tests for compute_confidence."""

    @pytest.fixture
    def snapshot(self) -> OptionSurfaceSnapshot:
        """Good quality surface with tight spreads and monotonic calls."""
        calls = (
            OptionPoint(strike=90.0, option_type="call", bid=11.9, ask=12.1, mid=12.0, spread=0.2),
            OptionPoint(strike=100.0, option_type="call", bid=4.9, ask=5.1, mid=5.0, spread=0.2),
            OptionPoint(strike=110.0, option_type="call", bid=0.9, ask=1.1, mid=1.0, spread=0.2),
        )
        puts = (
            OptionPoint(strike=90.0, option_type="put", bid=0.9, ask=1.1, mid=1.0, spread=0.2),
            OptionPoint(strike=100.0, option_type="put", bid=4.9, ask=5.1, mid=5.0, spread=0.2),
            OptionPoint(strike=110.0, option_type="put", bid=11.9, ask=12.1, mid=12.0, spread=0.2),
        )
        return OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)

    def test_high_confidence_when_estimators_agree(self, snapshot: OptionSurfaceSnapshot) -> None:
        prob_simple = StrikeProbability(strike_price=100.0, prob_above=0.5)
        prob_slope = StrikeProbability(strike_price=100.0, prob_above=0.5)

        confidence, diagnostics = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=prob_simple,
            prob_slope=prob_slope,
        )

        assert confidence > 0.7
        assert diagnostics.agreement == pytest.approx(1.0)

    def test_low_confidence_when_estimators_disagree(self, snapshot: OptionSurfaceSnapshot) -> None:
        prob_simple = StrikeProbability(strike_price=100.0, prob_above=0.8)
        prob_slope = StrikeProbability(strike_price=100.0, prob_above=0.2)

        confidence, diagnostics = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=prob_simple,
            prob_slope=prob_slope,
        )

        # Agreement score should be very low due to 0.6 delta
        assert diagnostics.agreement < 0.1
        # Overall confidence lower than when estimators agree
        assert confidence < 0.7

    def test_zero_agreement_when_estimator_missing(self, snapshot: OptionSurfaceSnapshot) -> None:
        prob_simple = StrikeProbability(strike_price=100.0, prob_above=0.5)

        confidence, diagnostics = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=prob_simple,
            prob_slope=None,
        )

        assert diagnostics.agreement == 0.0

    def test_zero_agreement_when_both_missing(self, snapshot: OptionSurfaceSnapshot) -> None:
        confidence, diagnostics = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=None,
            prob_slope=None,
        )

        assert diagnostics.agreement == 0.0

    def test_liquidity_score_with_tight_spreads(self, snapshot: OptionSurfaceSnapshot) -> None:
        prob = StrikeProbability(strike_price=100.0, prob_above=0.5)

        _, diagnostics = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=prob,
            prob_slope=prob,
        )

        # Spread = 0.2, mid = 5.0, relative = 0.04 -> high liquidity score
        assert diagnostics.liquidity > 0.9

    def test_liquidity_score_with_wide_spreads(self) -> None:
        calls = (
            OptionPoint(strike=90.0, option_type="call", bid=10.0, ask=14.0, mid=12.0, spread=4.0),
            OptionPoint(strike=100.0, option_type="call", bid=3.0, ask=7.0, mid=5.0, spread=4.0),
            OptionPoint(strike=110.0, option_type="call", bid=0.0, ask=2.0, mid=1.0, spread=2.0),
        )
        puts = (OptionPoint(strike=100.0, option_type="put", bid=3.0, ask=7.0, mid=5.0, spread=4.0),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)
        prob = StrikeProbability(strike_price=100.0, prob_above=0.5)

        _, diagnostics = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=prob,
            prob_slope=prob,
        )

        # Spread = 4.0, mid = 5.0, relative = 0.8 -> very low liquidity
        assert diagnostics.liquidity == 0.0

    def test_zero_liquidity_when_quotes_missing(self) -> None:
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=(), puts=())
        prob = StrikeProbability(strike_price=100.0, prob_above=0.5)

        _, diagnostics = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=prob,
            prob_slope=prob,
        )

        assert diagnostics.liquidity == 0.0

    def test_monotonicity_score_good(self, snapshot: OptionSurfaceSnapshot) -> None:
        """Calls decrease with strike -> monotonicity = 1.0."""
        prob = StrikeProbability(strike_price=100.0, prob_above=0.5)

        _, diagnostics = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=prob,
            prob_slope=prob,
        )

        assert diagnostics.monotonicity == 1.0

    def test_monotonicity_score_bad(self) -> None:
        """Calls increase with strike (arbitrage) -> monotonicity = 0.0."""
        calls = (
            OptionPoint(strike=90.0, option_type="call", bid=0.9, ask=1.1, mid=1.0, spread=0.2),
            OptionPoint(strike=100.0, option_type="call", bid=4.9, ask=5.1, mid=5.0, spread=0.2),
            OptionPoint(strike=110.0, option_type="call", bid=11.9, ask=12.1, mid=12.0, spread=0.2),
        )
        puts = (OptionPoint(strike=100.0, option_type="put", bid=4.9, ask=5.1, mid=5.0, spread=0.2),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)
        prob = StrikeProbability(strike_price=100.0, prob_above=0.5)

        _, diagnostics = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=prob,
            prob_slope=prob,
        )

        assert diagnostics.monotonicity == 0.0

    def test_confidence_clamped_to_valid_range(self, snapshot: OptionSurfaceSnapshot) -> None:
        prob = StrikeProbability(strike_price=100.0, prob_above=0.5)

        confidence, _ = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=prob,
            prob_slope=prob,
        )

        assert 0.0 <= confidence <= 1.0

    def test_spacing_score_at_edge(self) -> None:
        """Strike at edge of chain -> spacing = 0."""
        calls = (
            OptionPoint(strike=100.0, option_type="call", bid=4.9, ask=5.1, mid=5.0, spread=0.2),
            OptionPoint(strike=110.0, option_type="call", bid=0.9, ask=1.1, mid=1.0, spread=0.2),
        )
        puts = (OptionPoint(strike=100.0, option_type="put", bid=4.9, ask=5.1, mid=5.0, spread=0.2),)
        snapshot = OptionSurfaceSnapshot(symbol=TEST_SYMBOL, expiration_date=TEST_EXPIRY, calls=calls, puts=puts)
        prob = StrikeProbability(strike_price=100.0, prob_above=0.5)

        _, diagnostics = compute_confidence(
            snapshot=snapshot,
            strike=100.0,
            prob_simple=prob,
            prob_slope=prob,
        )

        assert diagnostics.spacing == 0.0
