"""
CLI entry point for model evaluation.

Usage:
    # Run full evaluation on all symbols with auto-detected dates
    python -m evaluation.main --baseline

    # Run on specific symbol with specific dates
    python -m evaluation.main --symbol NVDA --dates 2025-12-01 --baseline
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import cast, get_args

from domain.types import Symbol, is_symbol

from .database import get_option_daily_dates, get_stock_date_range
from .evaluator import EvaluationSummary, run_evaluation
from .metrics import ModelMetrics, compute_all_metrics, print_metrics_table
from .report_template import render_evaluation_report


def generate_monthly_dates(
    from_date: date,
    to_date: date,
    skip_recent_days: int = 14,
) -> list[date]:
    """Generate monthly sample dates from a date range.

    Args:
        from_date: Start date
        to_date: End date
        skip_recent_days: Skip dates within this many days of to_date (unreliable data)

    Returns:
        List of monthly sample dates (15th of each month)
    """
    dates = []
    # Start from the 15th of the first month
    current = from_date.replace(day=15)
    if current < from_date:
        # Move to next month if we're before from_date
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    # End date should be skip_recent_days before to_date
    end_date = to_date - timedelta(days=skip_recent_days)

    while current <= end_date:
        dates.append(current)
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return dates


def build_metrics_table_html(metrics: dict[str, ModelMetrics]) -> str:
    """Build HTML table for metrics."""
    rows = ""
    for name, m in metrics.items():
        n = m.n_predictions
        dropped = m.n_dropped

        # Brier score (lower is better, 0.25 is baseline)
        if m.brier_score is not None:
            brier_class = (
                "metric-good" if m.brier_score < 0.2 else "metric-warn" if m.brier_score < 0.25 else "metric-bad"
            )
            brier = f'<span class="{brier_class}">{m.brier_score:.4f}</span>'
        else:
            brier = '<span class="na">N/A</span>'

        # Log loss (lower is better)
        if m.log_loss is not None:
            logloss_class = "metric-good" if m.log_loss < 0.6 else "metric-warn" if m.log_loss < 0.7 else "metric-bad"
            logloss = f'<span class="{logloss_class}">{m.log_loss:.4f}</span>'
        else:
            logloss = '<span class="na">N/A</span>'

        # Accuracy (higher is better)
        if m.accuracy is not None:
            acc_class = "metric-good" if m.accuracy > 0.55 else "metric-warn" if m.accuracy > 0.5 else "metric-bad"
            acc = f'<span class="{acc_class}">{m.accuracy:.1%}</span>'
        else:
            acc = '<span class="na">N/A</span>'

        # Dropped count (lower is better)
        drop_class = "metric-good" if dropped == 0 else "metric-warn" if dropped < n * 0.1 else "metric-bad"
        drop_html = f'<span class="{drop_class}">{dropped}</span>'

        rows += f"""<tr>
            <td class="model-name">{name}</td>
            <td>{n}</td>
            <td>{drop_html}</td>
            <td>{brier}</td>
            <td>{logloss}</td>
            <td>{acc}</td>
        </tr>\n"""

    return f"""<table>
        <thead><tr>
            <th>Model</th>
            <th>N</th>
            <th>Dropped</th>
            <th>Brier Score</th>
            <th>Log Loss</th>
            <th>Accuracy</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def generate_html_report(
    symbol_summaries: dict[str, EvaluationSummary],
    combined_metrics: dict[str, ModelMetrics],
    output_path: Path,
) -> None:
    """Generate HTML evaluation report."""
    # Summary cards
    total_cases = sum(s.total_cases for s in symbol_summaries.values())
    successful_cases = sum(s.successful_cases for s in symbol_summaries.values())

    # Find best model by accuracy
    best_model = max(combined_metrics.values(), key=lambda m: m.accuracy or 0, default=None)
    best_acc = best_model.accuracy if best_model else None

    # Best accuracy formatting
    if best_acc:
        acc_class = "good" if best_acc > 0.55 else "warn" if best_acc > 0.5 else ""
        acc_value = f"{best_acc:.1%}"
    else:
        acc_class = ""
        acc_value = "N/A"

    summary_cards = f"""
    <div class="summary-card">
        <div class="label">Total Cases</div>
        <div class="value">{total_cases}</div>
    </div>
    <div class="summary-card">
        <div class="label">With Outcomes</div>
        <div class="value">{successful_cases}</div>
    </div>
    <div class="summary-card">
        <div class="label">Symbols</div>
        <div class="value">{len(symbol_summaries)}</div>
    </div>
    <div class="summary-card">
        <div class="label">Best Accuracy</div>
        <div class="value {acc_class}">{acc_value}</div>
    </div>
    """

    # Combined metrics table
    combined_table = build_metrics_table_html(combined_metrics)

    # Symbol tabs and content
    symbols = sorted(symbol_summaries.keys())
    symbol_tabs = ""
    symbol_content = ""

    for i, sym in enumerate(symbols):
        active = "active" if i == 0 else ""
        symbol_tabs += f'<div class="tab {active}" onclick="showSymbol(\'{sym}\')">{sym}</div>\n'

        summary = symbol_summaries[sym]
        metrics = compute_all_metrics(summary)

        symbol_content += f'<div class="symbol-content {active}" id="symbol-{sym}">\n'
        symbol_content += f"<h3>{sym}</h3>\n"
        symbol_content += f'<p class="meta">{summary.total_cases} cases, {summary.successful_cases} with outcomes</p>\n'
        symbol_content += build_metrics_table_html(metrics)
        symbol_content += "</div>\n"

    html = render_evaluation_report(
        summary_cards=summary_cards,
        combined_table=combined_table,
        symbol_tabs=symbol_tabs,
        symbol_content=symbol_content,
    )

    output_path.write_text(html)
    print(f"\nHTML report saved to: file://{output_path.absolute()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate probability models on historical data")
    parser.add_argument(
        "--symbols", nargs="*", default=None, help="Stock symbols to evaluate (default: all symbols with cached data)"
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        default=None,
        help="Prediction dates YYYY-MM-DD (default: monthly samples from cached data)",
    )
    parser.add_argument("--days", type=int, default=7, help="Days to expiry (default: 7)")
    parser.add_argument("--strikes", type=int, default=10, help="Number of strikes per date (default: 10)")
    parser.add_argument("--models", nargs="+", help="Models to evaluate (default: all)")
    parser.add_argument("--baseline", action="store_true", help="Use baseline only (no option data needed)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser after generating report")

    args = parser.parse_args()

    # Determine symbols to evaluate
    if args.symbols:
        symbols = [s for s in args.symbols if is_symbol(s)]
        if not symbols:
            print("No valid symbols provided")
            return
    else:
        # Use all symbols from domain
        symbols = list(get_args(Symbol))
        print(f"Using all {len(symbols)} symbols: {', '.join(symbols)}")

    # Per-symbol summaries for HTML report
    symbol_summaries: dict[str, EvaluationSummary] = {}

    # Combined summary across all symbols
    combined_summary = EvaluationSummary()

    for symbol in symbols:
        # Check if we have cached data for this symbol
        min_date, max_date = get_stock_date_range(symbol)
        if min_date is None or max_date is None:
            print(f"\nSkipping {symbol}: No cached stock data")
            continue

        # Determine dates to evaluate
        if args.dates:
            prediction_dates = []
            for date_str in args.dates:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                    prediction_dates.append(dt)
                except ValueError:
                    print(f"Invalid date format: {date_str} (expected YYYY-MM-DD)")
                    continue
        else:
            # Use option daily dates if available (for real model evaluation)
            # Otherwise fall back to monthly stock dates (baseline only)
            option_dates = get_option_daily_dates(symbol)
            if option_dates and not args.baseline:
                # Sample every 3rd date to avoid too many evaluations
                prediction_dates = option_dates[::3]
                print(f"Using {len(prediction_dates)} dates with option data")
            else:
                # Auto-generate monthly dates from cached stock data
                prediction_dates = generate_monthly_dates(min_date, max_date)

        if not prediction_dates:
            print(f"\nSkipping {symbol}: No valid dates")
            continue

        print(f"\n{'=' * 60}")
        print(f"EVALUATING {symbol}: {len(prediction_dates)} dates ({min_date} to {max_date})")
        print(f"Days to expiry: {args.days}, Strikes per date: {args.strikes}")
        if args.baseline:
            print("Mode: Baseline only (no volatility models)")
        print("=" * 60)

        # Run evaluation
        summary = run_evaluation(
            symbol=cast(Symbol, symbol),
            prediction_dates=prediction_dates,
            days_to_expiry=args.days,
            n_strikes=args.strikes,
            model_names=args.models,
            use_baseline_only=args.baseline,
            verbose=True,
        )

        # Store per-symbol summary
        symbol_summaries[symbol] = summary

        # Print per-symbol metrics
        metrics = compute_all_metrics(summary)
        print_metrics_table(metrics)

        # Add to combined summary
        for result in summary.results:
            combined_summary.add_result(result)

    # Print combined metrics if multiple symbols
    if len(symbols) > 1:
        print("\n" + "=" * 60)
        print("COMBINED METRICS (ALL SYMBOLS)")
        print("=" * 60)
        combined_metrics = compute_all_metrics(combined_summary)
        print_metrics_table(combined_metrics)
    else:
        combined_metrics = compute_all_metrics(combined_summary)

    # Generate HTML report
    if symbol_summaries:
        output_path = Path("evaluation_report.html")
        generate_html_report(symbol_summaries, combined_metrics, output_path)

        if not args.no_browser:
            import subprocess

            subprocess.Popen(["open", str(output_path.absolute())])


if __name__ == "__main__":
    main()
