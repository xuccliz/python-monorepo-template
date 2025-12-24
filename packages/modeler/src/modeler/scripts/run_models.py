"""
Run all probability models for each expiry in the next N days.

Usage:
    python -m modeler.scripts.run_models --ticker NVDA --strike 140
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
from math import isnan

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
    estimate_forward_put_call_parity,
)
from store import EventStore, OptionStore, build_surface_snapshot

load_dotenv()


def run_models_for_expiry(
    *,
    store: OptionStore,
    symbol: Symbol,
    expiration_date: datetime,
    strike: float,
    max_spread: float | None = 1.0,
    event_store: EventStore | None = None,
) -> ExpiryPredictions:
    """Run all models for a single expiry and return predictions."""
    today = datetime.now(UTC).date()
    tte_days = (expiration_date.date() - today).days
    T = tte_days / 365.0

    snapshot = build_surface_snapshot(
        states=store.get_by_symbol(symbol),
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
    prob = simple.prob_above(strike)
    predictions.append(
        ModelPrediction(
            model_name="simple",
            prob_above=None if isnan(prob) else prob,
            forward=forward,
        )
    )

    # Slope model
    slope = build_slope_model(snapshot, max_spread=max_spread)
    prob = slope.prob_above(strike)
    predictions.append(
        ModelPrediction(
            model_name="slope",
            prob_above=None if isnan(prob) else prob,
            forward=forward,
        )
    )

    # SVI model
    if T > 0 and (svi := build_svi_model(snapshot=snapshot, T=T, max_spread=max_spread)):
        predictions.append(
            ModelPrediction(
                model_name="svi",
                prob_above=svi.prob_above(strike),
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
                prob_above=spline.prob_above(strike),
                forward=spline.fit.forward,
                extra={"n_points": spline.fit.n_points},
            )
        )
    else:
        predictions.append(ModelPrediction(model_name="spline", prob_above=None))

    # Polymarket
    if event_store:
        exp_iso = expiration_date.isoformat().replace("+00:00", "Z")
        events = event_store.get_by_symbol(symbol)
        pm_end_date = next((e.end_date for e in events if e.end_date == exp_iso), None)
        if pm_end_date:
            predictions.append(
                ModelPrediction(
                    model_name="polymarket",
                    prob_above=event_store.get_polymarket_prob(symbol, pm_end_date, strike, direction="above"),
                )
            )
        else:
            predictions.append(ModelPrediction(model_name="polymarket", prob_above=None))

    return ExpiryPredictions(
        expiration_date=expiration_date,
        tte_days=tte_days,
        strike_price=strike,
        predictions=predictions,
    )


def get_expiries(store: OptionStore, symbol: Symbol, n_days: int) -> list[datetime]:
    """Get expiries for symbol within next N days (including today)."""
    today = datetime.now(UTC).date()
    cutoff = today + timedelta(days=n_days)
    states = store.get_by_symbol(symbol)
    return sorted({s.expiration_date for s in states if today <= s.expiration_date.date() <= cutoff})


def print_predictions(results: list[ExpiryPredictions]) -> None:
    """Pretty print predictions."""
    print("\n" + "=" * 80)
    print("MODEL PREDICTIONS")
    print("=" * 80)

    for r in results:
        print(f"\nExpiry: {r.expiration_date:%Y-%m-%d} ({r.tte_days} days) | Strike: {r.strike_price}")
        print("-" * 60)

        for p in r.predictions:
            if p.prob_above is not None:
                prob_str = f"P(above)={p.prob_above:.1%}, P(below)={p.prob_below:.1%}"
                fwd_str = f", F={p.forward:.2f}" if p.forward else ""
                extra_str = f", {p.extra}" if p.extra else ""
                print(f"  {p.model_name:10s}: {prob_str}{fwd_str}{extra_str}")
            else:
                print(f"  {p.model_name:10s}: insufficient data")

    print("\n" + "=" * 80)


def fetch_options_chain(ticker: str, expiration_lte: date | None = None) -> OptionStore:
    """Fetch options chain via REST API."""
    api_key = load_required_secret("MASSIVE_API_KEY")
    client = RESTClient(api_key=api_key)
    store = OptionStore()

    print(f"Fetching options chain for {ticker}...")

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
        if not details or not details.ticker:
            continue

        # Get bid/ask from last_quote or estimate from close
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

        store.apply_quote(
            OptionQuoteEvent(
                occ_symbol=details.ticker,
                bid=bid,
                ask=ask,
                ts=datetime.now(tz=UTC),
            )
        )
        count += 1

    print(f"Loaded {count} option contracts into store")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probability models for options")
    parser.add_argument("--ticker", default="NVDA", help="Underlying ticker")
    parser.add_argument("--strike", type=float, default=None, help="Strike price (omit for all strikes)")
    parser.add_argument("--days", type=int, default=30, help="Days ahead to look for expiries")
    args = parser.parse_args()

    if not is_symbol(args.ticker):
        print(f"Unknown ticker: {args.ticker}")
        return

    symbol: Symbol = args.ticker

    store = fetch_options_chain(symbol, expiration_lte=datetime.now(UTC).date() + timedelta(days=args.days))
    if store.count() == 0:
        print("No quotes fetched. Check API key.")
        return

    event_store = EventStore()
    print("Fetching Polymarket events...")
    print(f"Loaded {event_store.refresh()} Polymarket events")

    expiries = get_expiries(store, symbol, args.days)
    if not expiries:
        print(f"No expiries found in next {args.days} days")
        return

    print(f"\nFound {len(expiries)} expiries: {[e.strftime('%Y-%m-%d') for e in expiries]}")

    # Get strikes to run
    if args.strike:
        strikes = [args.strike]
    else:
        strikes = store.get_strikes(symbol)
        print(f"Running for {len(strikes)} strikes: {strikes[0]:.0f} - {strikes[-1]:.0f}")

    results = [
        run_models_for_expiry(
            store=store,
            symbol=symbol,
            expiration_date=exp,
            strike=strike,
            event_store=event_store,
        )
        for exp in expiries
        for strike in strikes
    ]

    print_predictions(results)


if __name__ == "__main__":
    main()
