"""
Model evaluation package.

Evaluate probability models on historical options data.
"""

from domain.models import ModelPrediction

from .evaluator import (
    EvaluationResult,
    EvaluationSummary,
    evaluate_case,
    run_evaluation,
)
from .historical_data import (
    DataNotCachedError,
    EvaluationCase,
    build_historical_snapshot,
    fetch_stock_close,
    generate_evaluation_cases,
    generate_synthetic_cases,
)
from .metrics import (
    ModelMetrics,
    compute_all_metrics,
    compute_brier_score,
    compute_log_loss,
    print_metrics_table,
)

__all__ = [
    # Evaluator
    "EvaluationResult",
    "EvaluationSummary",
    "ModelPrediction",
    "evaluate_case",
    "run_evaluation",
    # Historical data
    "DataNotCachedError",
    "EvaluationCase",
    "build_historical_snapshot",
    "fetch_stock_close",
    "generate_evaluation_cases",
    "generate_synthetic_cases",
    # Metrics
    "ModelMetrics",
    "compute_all_metrics",
    "compute_brier_score",
    "compute_log_loss",
    "print_metrics_table",
]
