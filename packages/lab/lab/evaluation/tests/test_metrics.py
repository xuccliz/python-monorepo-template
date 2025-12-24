"""Unit tests for evaluation metrics."""

from evaluation.metrics import (
    compute_accuracy,
    compute_brier_score,
    compute_log_loss,
)


class TestBrierScore:
    def test_perfect_predictions(self) -> None:
        """Perfect predictions should have Brier score of 0."""
        predicted = [1.0, 0.0, 1.0, 0.0]
        actual = [True, False, True, False]
        assert compute_brier_score(predicted, actual) == 0.0

    def test_worst_predictions(self) -> None:
        """Completely wrong predictions should have Brier score of 1."""
        predicted = [0.0, 1.0, 0.0, 1.0]
        actual = [True, False, True, False]
        assert compute_brier_score(predicted, actual) == 1.0

    def test_uncertain_predictions(self) -> None:
        """50% predictions should have Brier score of 0.25."""
        predicted = [0.5, 0.5, 0.5, 0.5]
        actual = [True, False, True, False]
        score = compute_brier_score(predicted, actual)
        assert score is not None
        assert abs(score - 0.25) < 1e-10

    def test_empty_input(self) -> None:
        """Empty input should return None."""
        assert compute_brier_score([], []) is None


class TestLogLoss:
    def test_confident_correct(self) -> None:
        """High confidence correct predictions should have low log loss."""
        predicted = [0.99, 0.01, 0.99, 0.01]
        actual = [True, False, True, False]
        score = compute_log_loss(predicted, actual)
        assert score is not None
        assert score < 0.1

    def test_confident_wrong(self) -> None:
        """High confidence wrong predictions should have high log loss."""
        predicted = [0.01, 0.99, 0.01, 0.99]
        actual = [True, False, True, False]
        score = compute_log_loss(predicted, actual)
        assert score is not None
        assert score > 3.0

    def test_empty_input(self) -> None:
        """Empty input should return None."""
        assert compute_log_loss([], []) is None


class TestAccuracy:
    def test_all_correct(self) -> None:
        """All correct predictions should have accuracy of 1.0."""
        predicted = [0.9, 0.1, 0.8, 0.2]
        actual = [True, False, True, False]
        # Use uncertainty_band=0 to count all predictions
        assert compute_accuracy(predicted, actual, uncertainty_band=0) == 1.0

    def test_all_wrong(self) -> None:
        """All wrong predictions should have accuracy of 0.0."""
        predicted = [0.1, 0.9, 0.2, 0.8]
        actual = [True, False, True, False]
        # Use uncertainty_band=0 to count all predictions
        assert compute_accuracy(predicted, actual, uncertainty_band=0) == 0.0

    def test_half_correct(self) -> None:
        """Half correct predictions should have accuracy of 0.5."""
        predicted = [0.9, 0.9, 0.1, 0.1]
        actual = [True, False, True, False]
        # Use uncertainty_band=0 to count all predictions
        assert compute_accuracy(predicted, actual, uncertainty_band=0) == 0.5

    def test_empty_input(self) -> None:
        """Empty input should return None."""
        assert compute_accuracy([], []) is None

    def test_uncertain_predictions_excluded(self) -> None:
        """Predictions near threshold should be excluded from accuracy."""
        # All predictions are at exactly 0.5 (uncertain)
        predicted = [0.5, 0.5, 0.5, 0.5]
        actual = [True, False, True, False]
        # With default uncertainty_band=0.05, all 0.5 predictions are excluded
        assert compute_accuracy(predicted, actual) is None

    def test_uncertain_band_filters_near_threshold(self) -> None:
        """Only decisive predictions should be counted."""
        # 0.52 and 0.48 are within 0.05 of threshold, so excluded
        # 0.9 and 0.1 are decisive
        predicted = [0.9, 0.52, 0.48, 0.1]
        actual = [True, True, False, False]
        # Only 0.9 (correct) and 0.1 (correct) should be counted = 100%
        assert compute_accuracy(predicted, actual, uncertainty_band=0.05) == 1.0
