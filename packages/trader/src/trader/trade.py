"""
Trade execution logic for the trading bot.

Contains functions for:
- Building and executing trades
- Evaluating strike prices
- Making trade decisions
"""

import logging
from datetime import datetime

from domain.models import (
    Evaluation,
    MarketMetadata,
    OptionSurfaceSnapshot,
    OrderResult,
    TradeDecision,
)
from domain.types import Symbol
from domain.utils import time_to_expiry_years
from modeler.confidence_score import compute_confidence
from modeler.models import (
    SimpleModel,
    SlopeModel,
    SplineModel,
    SVIModel,
    build_simple_model,
    build_slope_model,
    build_spline_model,
    build_svi_model,
)
from py_clob_client.clob_types import PostOrdersArgs
from store import EventStore, OptionStore, build_surface_snapshot

from trader.config import Config
from trader.polymarket_client import PolymarketClient
from trader.utils import filter_expiries, get_unique_expiries, log_order_results

logger = logging.getLogger(__name__)


async def trade_symbol(
    symbol: Symbol,
    config: Config,
    client: PolymarketClient,
    event_store: EventStore,
    option_store: OptionStore,
) -> None:
    """Trade a single symbol across all expiries."""
    all_expiries = get_unique_expiries(event_store, option_store)
    valid_expiries = filter_expiries(all_expiries, config.max_days)

    if not valid_expiries:
        logger.warning("No valid expiries found for symbol: %s", symbol)
        return

    for expiry in valid_expiries:
        await trade_expiry(
            symbol=symbol,
            expiry=expiry,
            config=config,
            client=client,
            event_store=event_store,
            option_store=option_store,
        )


async def trade_expiry(
    symbol: Symbol,
    expiry: datetime,
    config: Config,
    client: PolymarketClient,
    event_store: EventStore,
    option_store: OptionStore,
) -> None:
    """Trade a single symbol/expiry combination."""
    event = event_store.get_by_symbol_and_end_date(symbol, expiry)

    if not event or not event.markets:
        logger.warning("[%s %s] no event/markets found in event_store", symbol, expiry.date())
        return

    snapshot = build_surface_snapshot(
        store=option_store,
        symbol=symbol,
        expiration_date=expiry,
        max_spread=config.max_spread,
    )

    polymarket_strikes = [m.strike_price for m in event.markets]
    option_strikes = snapshot.all_strikes
    matching = set(polymarket_strikes) & set(option_strikes)
    logger.debug(
        "[%s %s] strikes: %d polymarket, %d options, %d matching",
        symbol,
        expiry.date(),
        len(polymarket_strikes),
        len(option_strikes),
        len(matching),
    )

    T = time_to_expiry_years(expiry)
    if T <= 0:
        logger.warning("Invalid option end date %s: already expired", expiry)
        return

    simple = build_simple_model(snapshot=snapshot, max_spread=config.max_spread) if "simple" in config.models else None
    slope = build_slope_model(snapshot=snapshot, max_spread=config.max_spread) if "slope" in config.models else None
    svi = build_svi_model(snapshot=snapshot, T=T, max_spread=config.max_spread) if "svi" in config.models else None
    spline = (
        build_spline_model(snapshot=snapshot, T=T, max_spread=config.max_spread) if "spline" in config.models else None
    )

    orders: list[tuple[OrderResult, PostOrdersArgs]] = []
    for market in event.markets:
        try:
            market_orders = process_market(
                market=market,
                snapshot=snapshot,
                simple=simple,
                slope=slope,
                svi=svi,
                spline=spline,
                config=config,
                client=client,
            )
            orders.extend(market_orders)
        except Exception:
            logger.exception("[strike=%s] failed to process market", market.strike_price)

    if not orders:
        logger.info("[%s %s] no order passed filters", symbol, expiry.date())
        return

    # Submit orders in batches of 15 (Polymarket limit)
    for i in range(0, len(orders), 15):
        batch = orders[i : i + 15]
        batch_orders = [o[1] for o in batch]
        batch_meta = [o[0] for o in batch]
        results = await client.place_orders_async(batch_orders)
        log_order_results(results, batch_meta, symbol, expiry)


def process_market(
    market: MarketMetadata,
    snapshot: OptionSurfaceSnapshot,
    simple: SimpleModel | None,
    slope: SlopeModel | None,
    svi: SVIModel | None,
    spline: SplineModel | None,
    config: Config,
    client: PolymarketClient,
) -> list[tuple[OrderResult, PostOrdersArgs]]:
    """Evaluate, decide, and sign orders for a single market."""
    evaluation = evaluate_strike(
        snapshot=snapshot,
        strike_price=market.strike_price,
        simple=simple,
        slope=slope,
        svi=svi,
        spline=spline,
    )

    if evaluation is None:
        return []

    above_decision, below_decision = decide_trade(
        market=market,
        evaluation=evaluation,
        config=config,
    )

    results: list[tuple[OrderResult, PostOrdersArgs]] = []

    if above_decision.should_trade:
        meta_above = OrderResult(
            strike=market.strike_price,
            outcome="YES",
            size=above_decision.size or 0,
            price=above_decision.price or 0,
            total=above_decision.total_amount or 0,
            prob=evaluation.prob_above,
            confidence=evaluation.confidence_score,
        )
        order = client.build_order(market=market, decision=above_decision)
        results.append((meta_above, order))

    if below_decision.should_trade:
        meta_below = OrderResult(
            strike=market.strike_price,
            outcome="NO",
            size=below_decision.size or 0,
            price=below_decision.price or 0,
            total=below_decision.total_amount or 0,
            prob=evaluation.prob_below,
            confidence=evaluation.confidence_score,
        )
        order = client.build_order(market=market, decision=below_decision)
        results.append((meta_below, order))

    return results


def evaluate_strike(
    snapshot: OptionSurfaceSnapshot,
    strike_price: float,
    simple: SimpleModel | None,
    slope: SlopeModel | None,
    svi: SVIModel | None,
    spline: SplineModel | None,
) -> Evaluation | None:
    """Evaluate strike price using pre-built models and confidence score."""
    p_simple = simple.prob_above(strike_price) if simple else None
    p_slope = slope.prob_above(strike_price) if slope else None
    p_svi = svi.prob_above(strike_price) if svi else None
    p_spline = spline.prob_above(strike_price) if spline else None

    # Need at least one valid probability
    probs = [p for p in [p_simple, p_slope, p_svi, p_spline] if p is not None]
    if not probs:
        logger.warning("[strike=%s] skipped: no valid model probabilities", strike_price)
        return None

    prob_above = sum(probs) / len(probs)

    confidence_score, diagnostics = compute_confidence(
        snapshot=snapshot,
        strike_price=strike_price,
        prob_simple=p_simple,
        prob_slope=p_slope,
        prob_svi=p_svi,
        prob_spline=p_spline,
    )

    return Evaluation(prob_above=prob_above, confidence_score=confidence_score, diagnostics=diagnostics)


def decide_trade(
    market: MarketMetadata,
    evaluation: Evaluation,
    config: Config,
) -> tuple[TradeDecision, TradeDecision]:
    """Decide whether to trade based on evaluation."""
    score = evaluation.confidence_score
    if score < config.min_confidence_score:
        logger.debug(
            "[strike=%s] skipped: confidence %.2f < %.2f",
            market.strike_price,
            score,
            config.min_confidence_score,
        )
        skip = TradeDecision.skip_trade(reason="Confidence score too low")
        return skip, skip

    side = "BUY"
    total_amount = config.max_total_amount * score
    price_above = evaluation.prob_above - config.min_edge
    price_below = evaluation.prob_below + config.min_edge

    # Skip if prices are outside valid range [0.01, 0.99]
    above_decision: TradeDecision
    below_decision: TradeDecision

    if price_above < 0.01 or price_above > 0.99:
        logger.debug("[strike=%s YES] skipped: price %.3f outside [0.01, 0.99]", market.strike_price, price_above)
        above_decision = TradeDecision.skip_trade(reason=f"Price {price_above:.3f} outside valid range")
    else:
        size_above = int(total_amount / price_above)
        if size_above < 1:
            above_decision = TradeDecision.skip_trade(reason=f"Size {size_above} too small")
        else:
            above_decision = TradeDecision.execute_trade(
                side=side,
                outcome="YES",
                size=size_above,
                price=price_above,
                total_amount=total_amount,
                reason=f"Confidence score {score:.2f} above threshold",
            )

    if price_below < 0.01 or price_below > 0.99:
        logger.debug("[strike=%s NO] skipped: price %.3f outside [0.01, 0.99]", market.strike_price, price_below)
        below_decision = TradeDecision.skip_trade(reason=f"Price {price_below:.3f} outside valid range")
    else:
        size_below = int(total_amount / price_below)
        if size_below < 1:
            below_decision = TradeDecision.skip_trade(reason=f"Size {size_below} too small")
        else:
            below_decision = TradeDecision.execute_trade(
                side=side,
                outcome="NO",
                size=size_below,
                price=price_below,
                total_amount=total_amount,
                reason=f"Confidence score {score:.2f} above threshold",
            )

    return above_decision, below_decision
