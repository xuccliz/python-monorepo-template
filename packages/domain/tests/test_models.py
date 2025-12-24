"""Tests for domain models."""

from datetime import UTC, datetime

import pytest
from domain.models import (
    ConfidenceDiagnostics,
    Evaluation,
    ModelPrediction,
    OptionPoint,
    OptionSurfaceSnapshot,
    TradeDecision,
)


class TestEvaluation:
    """Tests for Evaluation model."""

    def test_prob_below_property(self) -> None:
        """Test that prob_below is computed correctly."""
        diagnostics = ConfidenceDiagnostics(agreement=1.0, liquidity=1.0, monotonicity=1.0, spacing=1.0)
        evaluation = Evaluation(prob_above=0.7, confidence_score=0.9, diagnostics=diagnostics)

        assert evaluation.prob_below == pytest.approx(0.3)

    def test_prob_below_edge_cases(self) -> None:
        """Test prob_below at edge values."""
        diagnostics = ConfidenceDiagnostics(agreement=1.0, liquidity=1.0, monotonicity=1.0, spacing=1.0)

        # prob_above = 0 -> prob_below = 1
        eval_zero = Evaluation(prob_above=0.0, confidence_score=0.5, diagnostics=diagnostics)
        assert eval_zero.prob_below == pytest.approx(1.0)

        # prob_above = 1 -> prob_below = 0
        eval_one = Evaluation(prob_above=1.0, confidence_score=0.5, diagnostics=diagnostics)
        assert eval_one.prob_below == pytest.approx(0.0)


class TestModelPrediction:
    """Tests for ModelPrediction model."""

    def test_prob_below_with_value(self) -> None:
        """Test prob_below when prob_above has a value."""
        prediction = ModelPrediction(model_name="simple", prob_above=0.6)

        assert prediction.prob_below == pytest.approx(0.4)

    def test_prob_below_when_none(self) -> None:
        """Test prob_below returns None when prob_above is None."""
        prediction = ModelPrediction(model_name="simple", prob_above=None)

        assert prediction.prob_below is None


class TestTradeDecision:
    """Tests for TradeDecision model."""

    def test_execute_trade_factory(self) -> None:
        """Test execute_trade class method creates correct decision."""
        decision = TradeDecision.execute_trade(
            side="BUY",
            outcome="YES",
            size=100,
            price=0.65,
            total_amount=65.0,
            reason="High confidence",
        )

        assert decision.should_trade is True
        assert decision.side == "BUY"
        assert decision.outcome == "YES"
        assert decision.size == 100
        assert decision.price == 0.65
        assert decision.total_amount == 65.0
        assert decision.reason == "High confidence"

    def test_skip_trade_factory(self) -> None:
        """Test skip_trade class method creates correct decision."""
        decision = TradeDecision.skip_trade(reason="Low confidence")

        assert decision.should_trade is False
        assert decision.side is None
        assert decision.outcome is None
        assert decision.size is None
        assert decision.price is None
        assert decision.total_amount is None
        assert decision.reason == "Low confidence"


class TestOptionSurfaceSnapshot:
    """Tests for OptionSurfaceSnapshot model."""

    @pytest.fixture
    def snapshot(self) -> OptionSurfaceSnapshot:
        """Create a test snapshot."""
        calls = (
            OptionPoint(strike_price=90.0, option_type="call", bid=10.0, ask=11.0, mid=10.5, spread=1.0),
            OptionPoint(strike_price=100.0, option_type="call", bid=5.0, ask=6.0, mid=5.5, spread=1.0),
            OptionPoint(strike_price=110.0, option_type="call", bid=1.0, ask=2.0, mid=1.5, spread=1.0),
        )
        puts = (
            OptionPoint(strike_price=90.0, option_type="put", bid=1.0, ask=2.0, mid=1.5, spread=1.0),
            OptionPoint(strike_price=100.0, option_type="put", bid=5.0, ask=6.0, mid=5.5, spread=1.0),
        )
        return OptionSurfaceSnapshot(
            symbol="NVDA",
            expiration_date=datetime(2026, 1, 17, 21, 0, 0, tzinfo=UTC),
            calls=calls,
            puts=puts,
        )

    def test_call_strikes(self, snapshot: OptionSurfaceSnapshot) -> None:
        """Test call_strikes property."""
        assert snapshot.call_strikes == [90.0, 100.0, 110.0]

    def test_put_strikes(self, snapshot: OptionSurfaceSnapshot) -> None:
        """Test put_strikes property."""
        assert snapshot.put_strikes == [90.0, 100.0]

    def test_all_strikes(self, snapshot: OptionSurfaceSnapshot) -> None:
        """Test all_strikes property returns sorted unique strikes."""
        assert snapshot.all_strikes == [90.0, 100.0, 110.0]
