"""
Model evaluation engine.

Runs probability models on historical data and compares predictions
to actual outcomes.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from domain.models import ModelPrediction, OptionSurfaceSnapshot
from domain.types import Symbol
from modeler import (
    build_simple_model,
    build_slope_model,
    build_spline_model,
    build_svi_model,
)

from .historical_data import (
    DataNotCachedError,
    EvaluationCase,
    build_historical_snapshot,
    fetch_stock_close,
    generate_evaluation_cases,
    generate_synthetic_cases,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Complete result for one evaluation case."""

    case: EvaluationCase
    predictions: list[ModelPrediction]
    actual_close: float | None  # Stock close on expiration
    actual_above: bool | None  # True if close > strike


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""

    results: list[EvaluationResult] = field(default_factory=list)
    total_cases: int = 0
    successful_cases: int = 0

    def add_result(self, result: EvaluationResult) -> None:
        self.results.append(result)
        self.total_cases += 1
        if result.actual_above is not None:
            self.successful_cases += 1


# Model builder registry
MODEL_BUILDERS: dict[str, Callable[..., object | None]] = {
    "simple": build_simple_model,
    "slope": build_slope_model,
    "svi": build_svi_model,
    "spline": build_spline_model,
}


def run_models_on_snapshot(
    snapshot: OptionSurfaceSnapshot,
    strike: float,
    T: float,
    model_names: list[str] | None = None,
) -> list[ModelPrediction]:
    """
    Run specified models on a snapshot and get predictions.

    Args:
        snapshot: Option surface snapshot
        strike: Strike price to evaluate
        T: Time to expiry in years
        model_names: Models to run (default: all)

    Returns:
        List of ModelPrediction objects
    """
    if model_names is None:
        model_names = list(MODEL_BUILDERS.keys())

    predictions: list[ModelPrediction] = []

    for name in model_names:
        builder = MODEL_BUILDERS.get(name)
        if builder is None:
            predictions.append(ModelPrediction(model_name=name, prob_above=None, error="Unknown model"))
            continue

        try:
            # Bug #8 fix: Use introspection to determine call signature
            sig = inspect.signature(builder)
            params = sig.parameters

            # Check if 'T' is a required parameter
            needs_T = "T" in params and params["T"].default is inspect.Parameter.empty

            if needs_T:
                model = builder(snapshot=snapshot, T=T)
            else:
                model = builder(snapshot)

            if model is None:
                predictions.append(ModelPrediction(model_name=name, prob_above=None, error="Build failed"))
                continue

            prob_above_fn = getattr(model, "prob_above", None)
            if prob_above_fn is None:
                predictions.append(ModelPrediction(model_name=name, prob_above=None, error="No prob_above method"))
                continue

            prob = prob_above_fn(strike)
            # Bug #3 fix: Check if prob_above returned None and record as error
            if prob is None:
                predictions.append(ModelPrediction(model_name=name, prob_above=None, error="No data for strike"))
            else:
                predictions.append(ModelPrediction(model_name=name, prob_above=prob))

        except Exception as e:
            predictions.append(ModelPrediction(model_name=name, prob_above=None, error=str(e)))

    return predictions


def evaluate_case(
    case: EvaluationCase,
    model_names: list[str] | None = None,
    use_baseline_only: bool = False,
) -> EvaluationResult | None:
    """
    Evaluate models on a single case.

    Args:
        case: Evaluation case with symbol, dates, strike
        model_names: Models to evaluate
        use_baseline_only: If True, only use baseline model (no option data needed)

    Returns:
        EvaluationResult with predictions and actual outcome, or None if data not available
    """
    # Get actual outcome first (only needs stock data)
    actual_close = None
    try:
        actual_close = fetch_stock_close(case.symbol, case.expiration_date)
    except DataNotCachedError:
        # Expected - data not in cache, leave actual_close as None
        pass
    except Exception as e:
        # Bug #2 fix: Log unexpected exceptions instead of silently swallowing
        logger.warning(f"Unexpected error fetching stock close for {case.symbol} on {case.expiration_date}: {e}")

    actual_above = None
    if actual_close is not None:
        actual_above = actual_close > case.strike_price

    # If baseline only mode, just return 0.5 probability (coin flip)
    if use_baseline_only:
        predictions = [ModelPrediction(model_name="baseline", prob_above=0.5)]
        return EvaluationResult(
            case=case,
            predictions=predictions,
            actual_close=actual_close,
            actual_above=actual_above,
        )

    # Build historical snapshot
    try:
        snapshot = build_historical_snapshot(
            symbol=case.symbol,
            expiration_date=case.expiration_date,
            data_date=case.prediction_date,
        )
    except DataNotCachedError:
        return None

    if snapshot is None:
        # No option data - use baseline
        predictions = [ModelPrediction(model_name="baseline", prob_above=0.5)]
        return EvaluationResult(
            case=case,
            predictions=predictions,
            actual_close=actual_close,
            actual_above=actual_above,
        )

    # Bug #1 fix: Calculate time to expiry using market close (4pm EST = 21:00 UTC)
    # This gives more accurate T values, especially for near-expiry options
    market_close_time = timedelta(hours=21)  # 4pm EST in UTC
    pred_dt = datetime.combine(case.prediction_date, datetime.min.time()).replace(tzinfo=UTC) + market_close_time
    exp_dt = datetime.combine(case.expiration_date, datetime.min.time()).replace(tzinfo=UTC) + market_close_time
    T = (exp_dt - pred_dt).total_seconds() / (365.0 * 24 * 3600)

    if T <= 0:
        predictions = [ModelPrediction(model_name="baseline", prob_above=0.5, error="T <= 0")]
        return EvaluationResult(
            case=case,
            predictions=predictions,
            actual_close=actual_close,
            actual_above=actual_above,
        )

    # Run models
    predictions = run_models_on_snapshot(snapshot, case.strike_price, T, model_names)

    return EvaluationResult(
        case=case,
        predictions=predictions,
        actual_close=actual_close,
        actual_above=actual_above,
    )


def run_evaluation(
    symbol: Symbol,
    prediction_dates: list,
    days_to_expiry: int = 7,
    n_strikes: int = 5,
    model_names: list[str] | None = None,
    use_baseline_only: bool = False,
    verbose: bool = True,
) -> EvaluationSummary:
    """
    Run full evaluation across multiple dates.

    Args:
        symbol: Stock ticker to evaluate
        prediction_dates: List of dates to make predictions from
        days_to_expiry: Target days to expiration for each prediction
        n_strikes: Number of strikes to evaluate per date
        model_names: Models to evaluate (default: all)
        use_baseline_only: If True, skip volatility models (only needs stock data)
        verbose: Print progress

    Returns:
        EvaluationSummary with all results
    """
    summary = EvaluationSummary()

    for pred_date in prediction_dates:
        if verbose:
            print(f"\nEvaluating {symbol} from {pred_date}...")

        try:
            cases = generate_evaluation_cases(
                symbol=symbol,
                prediction_date=pred_date,
                days_to_expiry=days_to_expiry,
                n_strikes=n_strikes,
            )
        except DataNotCachedError:
            if use_baseline_only:
                if verbose:
                    print("  Missing option metadata, generating synthetic cases...")
                cases = generate_synthetic_cases(
                    symbol=symbol,
                    prediction_date=pred_date,
                    days_to_expiry=days_to_expiry,
                    n_strikes=n_strikes,
                )
            else:
                if verbose:
                    print(f"  Skipping {pred_date}: option data not cached")
                continue

        if not cases:
            if verbose:
                print(f"  No evaluation cases generated for {pred_date}")
            continue

        for case in cases:
            result = evaluate_case(case, model_names, use_baseline_only=use_baseline_only)
            if result is None:
                continue
            summary.add_result(result)

            if verbose:
                status = "✓" if result.actual_above is not None else "✗"
                if result.actual_above is True:
                    outcome = "above"
                elif result.actual_above is False:
                    outcome = "below"
                else:
                    outcome = "N/A"
                print(f"  {status} Strike={case.strike_price:.0f}, Actual={outcome}")

    if verbose:
        print(f"\nTotal: {summary.total_cases} cases, {summary.successful_cases} with outcomes")

    return summary
