"""
Parameter sweep for model hyperparameter optimization.

Runs evaluations with different parameter combinations and finds optimal values.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import cast, get_args

from domain.types import Symbol
from lab.evaluation.database import get_option_daily_dates
from lab.evaluation.evaluator import EvaluationSummary
from lab.evaluation.metrics import ModelMetrics, compute_all_metrics


@dataclass(frozen=True, slots=True)
class ParamConfig:
    """Parameter configuration for a sweep run."""

    max_spread: float | None
    slope_window: int
    discount: float
    trim_pct: float
    spline_smoothing: float | None


@dataclass
class SweepResult:
    """Result of a parameter sweep run."""

    config: ParamConfig
    summary: EvaluationSummary
    metrics: dict[str, ModelMetrics]

    @property
    def best_brier(self) -> float | None:
        """Return best (lowest) Brier score across all models."""
        scores = [m.brier_score for m in self.metrics.values() if m.brier_score is not None]
        return min(scores) if scores else None

    @property
    def best_accuracy(self) -> float | None:
        """Return best (highest) accuracy across all models."""
        scores = [m.accuracy for m in self.metrics.values() if m.accuracy is not None]
        return max(scores) if scores else None


def run_sweep_evaluation(
    symbol: Symbol,
    prediction_dates: list[date],
    config: ParamConfig,
    days_to_expiry: int = 7,
    n_strikes: int = 5,
) -> SweepResult:
    """
    Run evaluation with specific parameter configuration.

    This is a modified version of run_evaluation that uses the param config.
    """
    import logging
    from datetime import UTC, datetime

    from domain.models import ModelPrediction, OptionSurfaceSnapshot
    from lab.evaluation.historical_data import (
        DataNotCachedError,
        EvaluationCase,
        build_historical_snapshot,
        fetch_stock_close,
        generate_evaluation_cases,
    )
    from modeler import (
        build_simple_model,
        build_slope_model,
        build_spline_model,
        build_svi_model,
    )

    logger = logging.getLogger(__name__)

    def run_models_on_snapshot_with_params(
        snapshot: OptionSurfaceSnapshot,
        strike: float,
        T: float,
        config: ParamConfig,
    ) -> list[ModelPrediction]:
        """Run models with specific parameter configuration."""
        predictions: list[ModelPrediction] = []

        # Simple model
        try:
            model = build_simple_model(snapshot, max_spread=config.max_spread)
            prob = model.prob_above(strike)
            if prob is None:
                predictions.append(ModelPrediction(model_name="simple", prob_above=None, error="No data"))
            else:
                predictions.append(ModelPrediction(model_name="simple", prob_above=prob))
        except Exception as e:
            predictions.append(ModelPrediction(model_name="simple", prob_above=None, error=str(e)))

        # Slope model
        try:
            model = build_slope_model(
                snapshot,
                window=config.slope_window,
                discount=config.discount,
                max_spread=config.max_spread,
            )
            prob = model.prob_above(strike)
            if prob is None:
                predictions.append(ModelPrediction(model_name="slope", prob_above=None, error="No data"))
            else:
                predictions.append(ModelPrediction(model_name="slope", prob_above=prob))
        except Exception as e:
            predictions.append(ModelPrediction(model_name="slope", prob_above=None, error=str(e)))

        # SVI model
        try:
            model = build_svi_model(
                snapshot=snapshot,
                T=T,
                discount=config.discount,
                max_spread=config.max_spread,
                trim_pct=config.trim_pct,
            )
            if model is None:
                predictions.append(ModelPrediction(model_name="svi", prob_above=None, error="Build failed"))
            else:
                prob = model.prob_above(strike)
                predictions.append(ModelPrediction(model_name="svi", prob_above=prob))
        except Exception as e:
            predictions.append(ModelPrediction(model_name="svi", prob_above=None, error=str(e)))

        # Spline model
        try:
            model = build_spline_model(
                snapshot=snapshot,
                T=T,
                discount=config.discount,
                max_spread=config.max_spread,
                trim_pct=config.trim_pct,
                smoothing=config.spline_smoothing,
            )
            if model is None:
                predictions.append(ModelPrediction(model_name="spline", prob_above=None, error="Build failed"))
            else:
                prob = model.prob_above(strike)
                predictions.append(ModelPrediction(model_name="spline", prob_above=prob))
        except Exception as e:
            predictions.append(ModelPrediction(model_name="spline", prob_above=None, error=str(e)))

        return predictions

    def evaluate_case_with_params(
        case: EvaluationCase,
        config: ParamConfig,
    ):
        """Evaluate a single case with specific parameters."""
        from lab.evaluation.evaluator import EvaluationResult

        # Get actual outcome
        actual_close = None
        try:
            actual_close = fetch_stock_close(case.symbol, case.expiration_date)
        except DataNotCachedError:
            pass
        except Exception as e:
            logger.warning(f"Error fetching close: {e}")

        actual_above = None
        if actual_close is not None:
            actual_above = actual_close > case.strike_price

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
            return None

        # Calculate T
        market_close = timedelta(hours=21)
        pred_dt = datetime.combine(case.prediction_date, datetime.min.time()).replace(tzinfo=UTC) + market_close
        exp_dt = datetime.combine(case.expiration_date, datetime.min.time()).replace(tzinfo=UTC) + market_close
        T = (exp_dt - pred_dt).total_seconds() / (365.0 * 24 * 3600)

        if T <= 0:
            return None

        predictions = run_models_on_snapshot_with_params(snapshot, case.strike_price, T, config)

        return EvaluationResult(
            case=case,
            predictions=predictions,
            actual_close=actual_close,
            actual_above=actual_above,
        )

    # Run evaluation
    summary = EvaluationSummary()

    for pred_date in prediction_dates:
        try:
            cases = generate_evaluation_cases(
                symbol=symbol,
                prediction_date=pred_date,
                days_to_expiry=days_to_expiry,
                n_strikes=n_strikes,
            )
        except DataNotCachedError:
            continue

        if not cases:
            continue

        for case in cases:
            result = evaluate_case_with_params(case, config)
            if result is not None:
                summary.add_result(result)

    metrics = compute_all_metrics(summary)

    return SweepResult(config=config, summary=summary, metrics=metrics)


def run_parameter_sweep(
    symbols: list[Symbol] | None = None,
    max_dates_per_symbol: int = 10,
    verbose: bool = True,
) -> list[SweepResult]:
    """
    Run parameter sweep across all combinations.

    Args:
        symbols: Symbols to evaluate (default: all with data)
        max_dates_per_symbol: Maximum prediction dates per symbol
        verbose: Print progress

    Returns:
        List of SweepResult sorted by best Brier score
    """
    # Define parameter grid
    param_grid = {
        "max_spread": [None, 0.5, 1.0, 2.0],
        "slope_window": [1, 2, 3],
        "discount": [1.0],  # Keep at 1.0 for simplicity
        "trim_pct": [0.01, 0.02, 0.05],
        "spline_smoothing": [None, 0.01, 0.1, 1.0],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    configs = [ParamConfig(**dict(zip(keys, values, strict=True))) for values in combinations]

    if verbose:
        print(f"Testing {len(configs)} parameter combinations")

    # Get symbols with data
    if symbols is None:
        symbols = list(get_args(Symbol))

    # Collect prediction dates
    symbol_dates: dict[str, list[date]] = {}
    for symbol in symbols:
        dates = get_option_daily_dates(symbol)
        if dates:
            # Sample evenly across available dates
            step = max(1, len(dates) // max_dates_per_symbol)
            sampled = dates[::step][:max_dates_per_symbol]
            symbol_dates[symbol] = sampled
            if verbose:
                print(f"  {symbol}: {len(sampled)} dates")

    if not symbol_dates:
        print("No data available for any symbol")
        return []

    # Run sweep
    results: list[SweepResult] = []

    for i, config in enumerate(configs):
        if verbose:
            print(
                f"\n[{i + 1}/{len(configs)}] Testing: max_spread={config.max_spread}, "
                f"window={config.slope_window}, trim={config.trim_pct}, smooth={config.spline_smoothing}"
            )

        # Combine results from all symbols
        combined_summary = EvaluationSummary()

        for symbol, dates in symbol_dates.items():
            result = run_sweep_evaluation(
                symbol=cast(Symbol, symbol),
                prediction_dates=dates,
                config=config,
            )
            for r in result.summary.results:
                combined_summary.add_result(r)

        combined_metrics = compute_all_metrics(combined_summary)

        sweep_result = SweepResult(
            config=config,
            summary=combined_summary,
            metrics=combined_metrics,
        )
        results.append(sweep_result)

        if verbose:
            for name, m in combined_metrics.items():
                brier = f"{m.brier_score:.4f}" if m.brier_score else "N/A"
                acc = f"{m.accuracy:.1%}" if m.accuracy else "N/A"
                print(f"    {name}: N={m.n_predictions}, Brier={brier}, Acc={acc}")

    # Sort by best Brier score
    results.sort(key=lambda r: r.best_brier or float("inf"))

    return results


def print_sweep_summary(results: list[SweepResult], top_n: int = 10) -> None:
    """Print summary of sweep results."""
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP RESULTS - TOP CONFIGURATIONS")
    print("=" * 80)

    for i, result in enumerate(results[:top_n]):
        config = result.config
        if result.best_brier:
            print(f"\n#{i + 1}: Brier={result.best_brier:.4f}, Accuracy={result.best_accuracy:.1%}")
        else:
            print(f"\n#{i + 1}: No valid predictions")
        print(
            f"    max_spread={config.max_spread}, window={config.slope_window}, "
            f"trim_pct={config.trim_pct}, smoothing={config.spline_smoothing}"
        )

        for name, m in result.metrics.items():
            if m.brier_score is not None:
                print(f"      {name}: Brier={m.brier_score:.4f}, Acc={m.accuracy:.1%}, N={m.n_predictions}")


def save_sweep_results(results: list[SweepResult], output_path: Path) -> None:
    """Save sweep results to JSON."""
    data = []
    for result in results:
        entry = {
            "config": {
                "max_spread": result.config.max_spread,
                "slope_window": result.config.slope_window,
                "discount": result.config.discount,
                "trim_pct": result.config.trim_pct,
                "spline_smoothing": result.config.spline_smoothing,
            },
            "total_cases": result.summary.total_cases,
            "successful_cases": result.summary.successful_cases,
            "best_brier": result.best_brier,
            "best_accuracy": result.best_accuracy,
            "models": {
                name: {
                    "n_predictions": m.n_predictions,
                    "n_dropped": m.n_dropped,
                    "brier_score": m.brier_score,
                    "log_loss": m.log_loss,
                    "accuracy": m.accuracy,
                    "mean_prediction": m.mean_prediction,
                }
                for name, m in result.metrics.items()
            },
        }
        data.append(entry)

    output_path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    """Run parameter sweep from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run parameter sweep for model optimization")
    parser.add_argument("--symbols", nargs="*", help="Symbols to evaluate")
    parser.add_argument("--max-dates", type=int, default=10, help="Max prediction dates per symbol")
    parser.add_argument("--output", type=str, default="sweep_results.json", help="Output file path")
    args = parser.parse_args()

    symbols = args.symbols
    if symbols:
        from domain.types import is_symbol

        symbols = [s for s in symbols if is_symbol(s)]

    results = run_parameter_sweep(
        symbols=cast(list[Symbol] | None, symbols),
        max_dates_per_symbol=args.max_dates,
        verbose=True,
    )

    if results:
        print_sweep_summary(results)
        save_sweep_results(results, Path(args.output))


if __name__ == "__main__":
    main()
