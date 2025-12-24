#!/usr/bin/env python3
"""Measure end-to-end latency: WSS quote → trade execution."""

import asyncio
import cProfile
import pstats
import sys
from datetime import UTC, datetime, timedelta

from domain.models import OptionQuoteEvent
from store import EventStore, OptionStore
from trader.config import Config
from trader.orchestrator import Orchestrator
from trader.polymarket_client import PolymarketClient
from trader.trade import trade_symbol


async def measure_latency(profile: bool = False):
    config = Config.load()
    config.symbols = ["NVDA"]
    config.model_loop_interval_ms = 100
    config.min_confidence_score = 0.6
    config.min_edge = 0.3
    config.max_total_amount = 1.0

    option_store = OptionStore()
    event_store = EventStore()
    quote_queue: asyncio.Queue[OptionQuoteEvent] = asyncio.Queue(maxsize=10000)

    orchestrator = Orchestrator(
        config=config,
        option_store=option_store,
        event_store=event_store,
        quote_queue=quote_queue,
    )

    print("Refreshing EventStore...")
    event_store.refresh()

    expiries = sorted(event_store.get_expiration_dates())
    today = datetime.now(UTC).date()
    cutoff = today + timedelta(days=config.max_days)
    valid_expiries = [e for e in expiries if today <= e.date() <= cutoff]

    if not valid_expiries:
        print("No valid expiries found")
        return

    target_expiry = valid_expiries[0]
    expiry_str = target_expiry.strftime("%y%m%d")

    event = event_store.get_by_symbol_and_end_date("NVDA", target_expiry)
    if not event or not event.markets:
        print("No NVDA markets found")
        return

    strikes = sorted([m.strike_price for m in event.markets])
    print(f"Target expiry: {target_expiry.date()}, strikes: {strikes[:5]}...")
    ts = datetime.now(UTC)

    # Generate realistic quotes: calls decrease with strike, puts increase
    quotes = []
    n = len(strikes)
    for i, strike in enumerate(strikes):
        call_mid = max(1.0, 30.0 * (1 - i / n))
        put_mid = max(0.5, 30.0 * (i / n))
        spread = 0.2
        quotes.append(
            OptionQuoteEvent(f"O:NVDA{expiry_str}C{int(strike * 1000):08d}", call_mid - spread, call_mid + spread, ts)
        )
        quotes.append(
            OptionQuoteEvent(f"O:NVDA{expiry_str}P{int(strike * 1000):08d}", put_mid - spread, put_mid + spread, ts)
        )

    for quote in quotes:
        await quote_queue.put(quote)

    print(f"Pushed {len(quotes)} quotes to queue")

    orchestrator._client = PolymarketClient(
        api_key=config.polymarket_api_key,
        wallet_address=config.polymarket_wallet,
    )
    orchestrator._running = True

    while not quote_queue.empty():
        quote = await quote_queue.get()
        option_store.apply_quote(quote)
        quote_queue.task_done()

    print("\nMeasuring trade cycle...")

    profiler = None
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    await trade_symbol(
        symbol="NVDA",
        config=config,
        client=orchestrator._client,
        event_store=event_store,
        option_store=option_store,
    )

    if profile and profiler:
        profiler.disable()
        print("\n" + "=" * 60)
        print("PROFILE (top 100 by cumulative time)")
        print("=" * 60)
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(100)


if __name__ == "__main__":
    profile = "--profile" in sys.argv
    asyncio.run(measure_latency(profile=profile))
