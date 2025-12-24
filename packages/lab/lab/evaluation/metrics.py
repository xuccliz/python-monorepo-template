"""
Evaluation metrics for model performance.

Computes accuracy metrics like Brier score, log loss, and calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .evaluator import EvaluationResult, EvaluationSummary


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """Metrics for a single model."""

    model_name: str
    n_predictions: int
    n_dropped: int  # Bug #7 fix: Track dropped predictions (prob_above was None)
    brier_score: float | None
    log_loss: float | None
    accuracy: float | None
    mean_prediction: float | None


def compute_brier_score(predicted: list[float], actual: list[bool]) -> float | None:
    """
    Compute Brier score (mean squared error of probability predictions).

    Brier score = mean((predicted - actual)^2)
    Lower is better. Range: [0, 1]

    Args:
        predicted: List of predicted probabilities (0-1)
        actual: List of actual outcomes (True/False)

    Returns:
        Brier score or None if no valid predictions
    """
    if not predicted or len(predicted) != len(actual):
        return None

    total = 0.0
    for p, a in zip(predicted, actual, strict=True):
        actual_val = 1.0 if a else 0.0
        total += (p - actual_val) ** 2

    return total / len(predicted)


def compute_log_loss(predicted: list[float], actual: list[bool], eps: float = 1e-15) -> float | None:
    """
    Compute log loss (cross-entropy loss).

    log_loss = -mean(actual * log(pred) + (1-actual) * log(1-pred))
    Lower is better.

    Args:
        predicted: List of predicted probabilities (0-1)
        actual: List of actual outcomes (True/False)
        eps: Small value to avoid log(0)

    Returns:
        Log loss or None if no valid predictions
    """
    if not predicted or len(predicted) != len(actual):
        return None

    total = 0.0
    for p, a in zip(predicted, actual, strict=True):
        # Clip predictions to avoid log(0)
        p = max(eps, min(1 - eps, p))
        actual_val = 1.0 if a else 0.0

        total += -(actual_val * math.log(p) + (1 - actual_val) * math.log(1 - p))

    return total / len(predicted)


def compute_accuracy(
    predicted: list[float],
    actual: list[bool],
    threshold: float = 0.5,
    uncertainty_band: float = 0.05,
) -> float | None:
    """
    Compute accuracy (percentage of correct predictions).

    A prediction is correct if (pred >= threshold) == actual.

    Bug #9 fix: Predictions within uncertainty_band of threshold are excluded
    from accuracy calculation since they represent "uncertain" predictions.

    Args:
        predicted: List of predicted probabilities (0-1)
        actual: List of actual outcomes (True/False)
        threshold: Decision threshold (default 0.5)
        uncertainty_band: Predictions within this distance of threshold are excluded

    Returns:
        Accuracy (0-1) or None if no valid predictions
    """
    if not predicted or len(predicted) != len(actual):
        return None

    # Filter out uncertain predictions (those too close to threshold)
    decisive_pairs = [(p, a) for p, a in zip(predicted, actual, strict=True) if abs(p - threshold) >= uncertainty_band]

    if not decisive_pairs:
        return None  # All predictions were uncertain

    correct = sum(1 for p, a in decisive_pairs if (p >= threshold) == a)

    return correct / len(decisive_pairs)


def compute_model_metrics(
    model_name: str,
    results: list[EvaluationResult],
) -> ModelMetrics:
    """
    Compute all metrics for a single model.

    Args:
        model_name: Name of the model
        results: List of evaluation results

    Returns:
        ModelMetrics with computed values
    """
    predicted: list[float] = []
    actual: list[bool] = []
    n_dropped = 0  # Bug #7 fix: Count dropped predictions

    for result in results:
        if result.actual_above is None:
            continue

        # Find this model's prediction
        found_prediction = False
        for pred in result.predictions:
            if pred.model_name == model_name:
                found_prediction = True
                if pred.prob_above is not None:
                    predicted.append(pred.prob_above)
                    actual.append(result.actual_above)
                else:
                    # Bug #7 fix: Track when prediction exists but prob_above is None
                    n_dropped += 1
                break

        # Also count if model had no prediction at all for this result
        if not found_prediction:
            n_dropped += 1

    if not predicted:
        return ModelMetrics(
            model_name=model_name,
            n_predictions=0,
            n_dropped=n_dropped,
            brier_score=None,
            log_loss=None,
            accuracy=None,
            mean_prediction=None,
        )

    return ModelMetrics(
        model_name=model_name,
        n_predictions=len(predicted),
        n_dropped=n_dropped,
        brier_score=compute_brier_score(predicted, actual),
        log_loss=compute_log_loss(predicted, actual),
        accuracy=compute_accuracy(predicted, actual),
        mean_prediction=sum(predicted) / len(predicted),
    )


def compute_all_metrics(summary: EvaluationSummary) -> dict[str, ModelMetrics]:
    """
    Compute metrics for all models in an evaluation summary.

    Args:
        summary: Evaluation summary with results

    Returns:
        Dict mapping model name to ModelMetrics
    """
    # Collect all model names
    model_names: set[str] = set()
    for result in summary.results:
        for pred in result.predictions:
            model_names.add(pred.model_name)

    return {name: compute_model_metrics(name, summary.results) for name in sorted(model_names)}


def print_metrics_table(metrics: dict[str, ModelMetrics]) -> None:
    """Print a formatted table of metrics."""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION METRICS")
    print("=" * 80)
    print(f"{'Model':<12} {'N':>6} {'Drop':>6} {'Brier':>10} {'LogLoss':>10} {'Accuracy':>10}")
    print("-" * 80)

    for name, m in metrics.items():
        n = m.n_predictions
        dropped = m.n_dropped
        brier = f"{m.brier_score:.4f}" if m.brier_score is not None else "N/A"
        logloss = f"{m.log_loss:.4f}" if m.log_loss is not None else "N/A"
        acc = f"{m.accuracy:.1%}" if m.accuracy is not None else "N/A"

        print(f"{name:<12} {n:>6} {dropped:>6} {brier:>10} {logloss:>10} {acc:>10}")

    print("=" * 80)
