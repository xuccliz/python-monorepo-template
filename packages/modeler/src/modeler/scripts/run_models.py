"""
Run all probability models for each expiry in the next N days.

Usage:
    python -m modeler.scripts.run_models --ticker NVDA --strike 140
    python -m modeler.scripts.run_models --all-symbols --days 30
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import get_args

from domain.models import ExpiryPredictions, ModelPrediction, OptionQuoteEvent
from domain.secrets import load_required_secret
from domain.types import Symbol, is_symbol
from dotenv import load_dotenv
from massive import RESTClient
from massive.rest.models.snapshot import OptionContractSnapshot
from modeler import (
    build_simple_model,
    build_slope_model,
    build_spline_model,
    build_svi_model,
    compute_confidence,
    estimate_forward_put_call_parity,
)
from store import EventStore, OptionStore, build_surface_snapshot

load_dotenv()


def run_models_for_expiry(
    *,
    store: OptionStore,
    symbol: Symbol,
    expiration_date: datetime,
    strike_price: float,
    max_spread: float | None = 1.0,
    event_store: EventStore | None = None,
) -> ExpiryPredictions:
    """Run all models for a single expiry and return predictions."""
    today = datetime.now(UTC).date()
    tte_days = (expiration_date.date() - today).days
    T = tte_days / 365.0

    snapshot = build_surface_snapshot(
        store=store,
        symbol=symbol,
        expiration_date=expiration_date,
        max_spread=max_spread,
    )

    predictions: list[ModelPrediction] = []

    # Forward estimate
    fwd_est = estimate_forward_put_call_parity(snapshot=snapshot, max_spread=max_spread)
    forward = fwd_est.forward if fwd_est else None

    # Simple model
    simple = build_simple_model(snapshot, max_spread=max_spread)
    prob_simple = simple.prob_above(strike_price)
    predictions.append(ModelPrediction(model_name="simple", prob_above=prob_simple, forward=forward))

    # Slope model
    slope = build_slope_model(snapshot, max_spread=max_spread)
    prob_slope = slope.prob_above(strike_price)
    predictions.append(ModelPrediction(model_name="slope", prob_above=prob_slope, forward=forward))

    # SVI model
    if T > 0 and (svi := build_svi_model(snapshot=snapshot, T=T, max_spread=max_spread)):
        predictions.append(
            ModelPrediction(
                model_name="svi",
                prob_above=svi.prob_above(strike_price),
                forward=svi.fit.forward,
                extra={"n_points": svi.fit.n_points},
            )
        )
    else:
        predictions.append(ModelPrediction(model_name="svi", prob_above=None))

    # Spline model
    if T > 0 and (spline := build_spline_model(snapshot=snapshot, T=T, max_spread=max_spread)):
        predictions.append(
            ModelPrediction(
                model_name="spline",
                prob_above=spline.prob_above(strike_price),
                forward=spline.fit.forward,
                extra={"n_points": spline.fit.n_points},
            )
        )
    else:
        predictions.append(ModelPrediction(model_name="spline", prob_above=None))

    # Polymarket
    polymarket_bid: float | None = None
    polymarket_ask: float | None = None
    if event_store:
        events = event_store.get_by_symbol(symbol)
        pm_end_date = next((e.end_date for e in events if e.end_date == expiration_date), None)
        if pm_end_date:
            market = event_store.get_market(symbol, pm_end_date, strike_price)
            if market:
                polymarket_bid = float(market.best_bid) if market.best_bid else None
                polymarket_ask = float(market.best_ask) if market.best_ask else None
            predictions.append(
                ModelPrediction(
                    model_name="polymarket",
                    prob_above=event_store.get_polymarket_prob(symbol, pm_end_date, strike_price, direction="above"),
                )
            )
        else:
            predictions.append(ModelPrediction(model_name="polymarket", prob_above=None))

    # Compute confidence score
    confidence_score, _ = compute_confidence(
        snapshot=snapshot,
        strike_price=strike_price,
        prob_simple=prob_simple,
        prob_slope=prob_slope,
    )

    return ExpiryPredictions(
        expiration_date=expiration_date,
        tte_days=tte_days,
        strike_price=strike_price,
        predictions=predictions,
        confidence_score=confidence_score,
        polymarket_bid=polymarket_bid,
        polymarket_ask=polymarket_ask,
    )


def get_expiries(store: OptionStore, symbol: Symbol, n_days: int) -> list[datetime]:
    """Get expiries for symbol within next N days (including today)."""
    today = datetime.now(UTC).date()
    cutoff = today + timedelta(days=n_days)
    states = store.get_by_symbol(symbol)
    return sorted({s.expiration_date for s in states if today <= s.expiration_date.date() <= cutoff})


def fetch_options_chain(ticker: str, expiration_lte: date | None = None) -> OptionStore:
    """Fetch options chain via REST API."""
    api_key = load_required_secret("MASSIVE_API_KEY")
    client = RESTClient(api_key=api_key)
    store = OptionStore()

    print(f"Fetching options chain for {ticker}...")

    if not is_symbol(ticker):
        print(f"Unknown ticker: {ticker}")
        return store

    symbol: Symbol = ticker
    today = datetime.now(UTC).date()
    result = client.list_snapshot_options_chain(
        underlying_asset=ticker,
        params={
            "expiration_date.gte": today.isoformat(),
            "expiration_date.lte": (expiration_lte or today + timedelta(days=30)).isoformat(),
        },
    )

    count = 0
    for item in result:
        if isinstance(item, bytes):
            continue
        snapshot: OptionContractSnapshot = item
        details = snapshot.details
        if not details:
            continue

        strike_price = details.strike_price
        exp_date_str = details.expiration_date
        contract_type = details.contract_type
        if not strike_price or not exp_date_str or not contract_type:
            continue

        option_type = "call" if contract_type == "call" else "put"

        bid, ask = 0.0, 0.0
        if snapshot.last_quote:
            bid = getattr(snapshot.last_quote, "bid", None) or getattr(snapshot.last_quote, "bid_price", None) or 0.0
            ask = getattr(snapshot.last_quote, "ask", None) or getattr(snapshot.last_quote, "ask_price", None) or 0.0
        elif snapshot.day and snapshot.day.close:
            spread = max(0.01, snapshot.day.close * 0.005)
            bid = snapshot.day.close - spread / 2
            ask = snapshot.day.close + spread / 2

        if bid <= 0 or ask <= 0:
            continue

        exp_parts = exp_date_str.split("-")
        yy = exp_parts[0][2:]
        mm = exp_parts[1]
        dd = exp_parts[2]
        type_char = "C" if option_type == "call" else "P"
        strike_int = int(strike_price * 1000)
        occ_symbol = f"O:{symbol}{yy}{mm}{dd}{type_char}{strike_int:08d}"

        store.apply_quote(OptionQuoteEvent(occ_symbol=occ_symbol, bid=bid, ask=ask, ts=datetime.now(tz=UTC)))
        count += 1

    print(f"Loaded {count} option contracts into store")
    return store


def run_for_ticker(
    ticker: str, strike: float | None, days: int, event_store: EventStore
) -> dict[str, list[ExpiryPredictions]]:
    """Run models for a single ticker. Returns {expiry_str: [predictions]}."""
    results: dict[str, list[ExpiryPredictions]] = defaultdict(list)

    if not is_symbol(ticker):
        print(f"Unknown ticker: {ticker}")
        return results

    symbol: Symbol = ticker

    store = fetch_options_chain(symbol, expiration_lte=datetime.now(UTC).date() + timedelta(days=days))
    if store.count() == 0:
        print(f"No quotes fetched for {symbol}.")
        return results

    expiries = get_expiries(store, symbol, days)
    if not expiries:
        print(f"No expiries found in next {days} days for {symbol}")
        return results

    print(f"Found {len(expiries)} expiries for {symbol}")

    if strike:
        strikes = [strike]
    else:
        strikes = store.get_strikes(symbol, expiries[0]) if expiries else []

    for exp in expiries:
        exp_str = exp.strftime("%Y-%m-%d")
        for s in strikes:
            pred = run_models_for_expiry(
                store=store, symbol=symbol, expiration_date=exp, strike_price=s, event_store=event_store
            )
            results[exp_str].append(pred)

    return results


def generate_html_report(all_results: dict[str, dict[str, list[ExpiryPredictions]]], output_path: Path) -> None:
    """Generate HTML report with tabs for dates and symbols."""
    from modeler.scripts.report_template import render_html

    dates = sorted(all_results.keys())

    # Build date tabs
    date_tabs = ""
    for i, d in enumerate(dates):
        active = "active" if i == 0 else ""
        date_tabs += f'<div class="tab {active}" onclick="showDate(\'{d}\')">{d}</div>\n'

    # Build date content
    date_content = ""
    for i, d in enumerate(dates):
        active = "active" if i == 0 else ""
        date_content += f'<div class="date-content {active}" id="date-{d}">\n'
        date_content += f'<div class="symbol-tabs" id="symbol-tabs-{d}">\n'

        date_symbols = sorted(all_results[d].keys())
        for j, sym in enumerate(date_symbols):
            sym_active = "active" if j == 0 else ""
            date_content += (
                f"<div class=\"symbol-tab {sym_active}\" onclick=\"showSymbol('{d}', '{sym}')\">{sym}</div>\n"
            )

        date_content += "</div>\n"

        # Symbol content
        for j, sym in enumerate(date_symbols):
            sym_active = "active" if j == 0 else ""
            preds = all_results[d][sym]
            if not preds:
                continue

            forward = preds[0].predictions[0].forward
            fwd_str = f"Forward: ${forward:.2f}" if forward else ""

            date_content += f'<div class="symbol-content {sym_active}" id="symbol-{d}-{sym}">\n'
            date_content += f'<p class="meta">{sym} expiring {d} ({preds[0].tte_days} days) | {fwd_str}</p>\n'
            date_content += "<table>\n<thead><tr><th>Strike</th><th>Simple</th><th>Slope</th><th>SVI</th>"
            date_content += '<th>Spline</th><th class="polymarket">Polymarket</th>'
            date_content += '<th class="polymarket">Bid</th><th class="polymarket">Ask</th>'
            date_content += "<th>Conf</th></tr></thead>\n<tbody>\n"

            for pred in preds:
                date_content += f'<tr><td class="strike">${pred.strike_price:.0f}</td>'
                for p in pred.predictions:
                    is_pm = p.model_name == "polymarket"
                    td_class = 'class="polymarket"' if is_pm else ""
                    if p.prob_above is not None:
                        pct = p.prob_above * 100
                        css = "prob-high" if pct >= 70 else "prob-mid" if pct >= 30 else "prob-low"
                        date_content += f'<td {td_class}><span class="prob {css}">{pct:.1f}%</span></td>'
                    else:
                        date_content += f'<td {td_class}><span class="na">—</span></td>'
                # Bid/Ask columns
                if pred.polymarket_bid is not None:
                    date_content += f'<td class="polymarket">{pred.polymarket_bid:.2f}</td>'
                else:
                    date_content += '<td class="polymarket"><span class="na">—</span></td>'
                if pred.polymarket_ask is not None:
                    date_content += f'<td class="polymarket">{pred.polymarket_ask:.2f}</td>'
                else:
                    date_content += '<td class="polymarket"><span class="na">—</span></td>'
                # Confidence column
                if pred.confidence_score is not None:
                    conf_pct = pred.confidence_score * 100
                    conf_css = "conf-high" if conf_pct >= 60 else "conf-mid" if conf_pct >= 40 else "conf-low"
                    date_content += f'<td><span class="conf {conf_css}">{conf_pct:.0f}%</span></td>'
                else:
                    date_content += '<td><span class="na">—</span></td>'
                date_content += "</tr>\n"

            date_content += "</tbody></table></div>\n"

        date_content += "</div>\n"

    html = render_html(date_tabs=date_tabs, date_content=date_content)
    output_path.write_text(html)
    print(f"\nHTML report saved to: file://{output_path.absolute()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probability models for options")
    parser.add_argument("--ticker", default="NVDA", help="Underlying ticker")
    parser.add_argument("--all-symbols", action="store_true", help="Run for all symbols")
    parser.add_argument("--strike", type=float, default=None, help="Strike price (omit for all strikes)")
    parser.add_argument("--days", type=int, default=30, help="Days ahead to look for expiries")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser after generating report")
    args = parser.parse_args()

    if args.all_symbols:
        tickers = list(get_args(Symbol))
    else:
        tickers = [args.ticker]

    # Shared event store
    event_store = EventStore()
    print("Fetching Polymarket events...")
    print(f"Loaded {event_store.refresh()} Polymarket events")

    # Collect all results: {date: {symbol: [predictions]}}
    all_results: dict[str, dict[str, list[ExpiryPredictions]]] = defaultdict(dict)

    for ticker in tickers:
        ticker_results = run_for_ticker(ticker, args.strike, args.days, event_store)
        for date_str, preds in ticker_results.items():
            all_results[date_str][ticker] = preds

    if not all_results:
        print("No results to report.")
        return

    # Generate HTML
    output_path = Path("predictions_report.html")
    generate_html_report(all_results, output_path)

    if not args.no_browser:
        import subprocess

        subprocess.Popen(["open", str(output_path.absolute())])


if __name__ == "__main__":
    main()
