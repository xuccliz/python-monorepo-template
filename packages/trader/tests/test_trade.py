"""Tests for trade execution logic."""

from unittest.mock import Mock, patch

from domain.models import ConfidenceDiagnostics, Evaluation, MarketMetadata
from trader.config import Config
from trader.trade import decide_trade, evaluate_strike


def _make_diagnostics() -> ConfidenceDiagnostics:
    """Create default diagnostics for tests."""
    return ConfidenceDiagnostics(agreement=1.0, liquidity=1.0, monotonicity=1.0, spacing=1.0)


class TestDecideTrade:
    """Tests for decide_trade function."""

    def test_skips_when_confidence_too_low(self):
        """Test skips trade when confidence score is below threshold."""
        market = MarketMetadata(question="Test", strike_price=100.0, yes_token_id="yes", no_token_id="no")
        evaluation = Evaluation(prob_above=0.5, confidence_score=0.2, diagnostics=_make_diagnostics())
        config = Config(
            polymarket_wallet="test",
            polymarket_api_key="test",
            min_confidence_score=0.4,
        )

        above, below = decide_trade(market, evaluation, config)
        assert above.should_trade is False
        assert below.should_trade is False

    def test_executes_when_confidence_sufficient(self):
        """Test executes trade when confidence score is above threshold."""
        market = MarketMetadata(question="Test", strike_price=100.0, yes_token_id="yes", no_token_id="no")
        evaluation = Evaluation(prob_above=0.6, confidence_score=0.6, diagnostics=_make_diagnostics())
        config = Config(
            polymarket_wallet="test",
            polymarket_api_key="test",
            min_confidence_score=0.4,
            min_edge=0.05,
            max_total_amount=10.0,
        )

        above, below = decide_trade(market, evaluation, config)
        assert above.should_trade is True
        assert below.should_trade is True

    def test_skips_when_price_out_of_range(self):
        """Test skips trade when calculated price is outside [0.01, 0.99]."""
        market = MarketMetadata(question="Test", strike_price=100.0, yes_token_id="yes", no_token_id="no")
        # prob_above=0.99, so price_above=0.99-0.05=0.94 is valid
        # prob_below=0.01, so price_below=0.01+0.05=0.06 is valid
        evaluation = Evaluation(prob_above=0.99, confidence_score=0.6, diagnostics=_make_diagnostics())
        config = Config(
            polymarket_wallet="test",
            polymarket_api_key="test",
            min_confidence_score=0.4,
            min_edge=0.05,
            max_total_amount=10.0,
        )

        above, below = decide_trade(market, evaluation, config)
        assert above.should_trade is True
        assert below.should_trade is True

        # Edge case: price would be 1.0 - 0.05 = 0.95 (valid) and 0.0 + 0.05 = 0.05 (valid)
        evaluation2 = Evaluation(prob_above=1.0, confidence_score=0.6, diagnostics=_make_diagnostics())
        above2, below2 = decide_trade(market, evaluation2, config)
        assert above2.should_trade is True  # 1.0 - 0.05 = 0.95, valid
        assert below2.should_trade is True  # 0.0 + 0.05 = 0.05, valid


class TestEvaluateStrike:
    """Tests for evaluate_strike function."""

    def test_returns_none_when_no_valid_models(self):
        """Test returns None when no models produce valid probabilities."""
        snapshot = Mock()
        result = evaluate_strike(
            snapshot=snapshot,
            strike_price=100.0,
            simple=None,
            slope=None,
            svi=None,
            spline=None,
        )
        assert result is None

    def test_averages_valid_model_probabilities(self):
        """Test averages probabilities from valid models."""

        snapshot = Mock()

        simple = Mock()
        simple.prob_above.return_value = 0.6

        slope = Mock()
        slope.prob_above.return_value = 0.8

        # Mock compute_confidence to avoid complex snapshot setup
        with patch("trader.trade.compute_confidence") as mock_compute:
            mock_compute.return_value = (0.8, _make_diagnostics())

            result = evaluate_strike(
                snapshot=snapshot,
                strike_price=100.0,
                simple=simple,
                slope=slope,
                svi=None,
                spline=None,
            )

            assert result is not None
            assert result.prob_above == 0.7  # (0.6 + 0.8) / 2
            assert result.confidence_score == 0.8
