"""End-to-end integration test simulating WSS message arrival and trade execution.

Run with: RUN_UNSAFE_TESTS=1 pytest packages/trader/tests/test_e2e_integration.py -v
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pytest
from domain.models import OptionQuoteEvent
from store import EventStore, OptionStore
from trader.config import Config
from trader.orchestrator import Orchestrator
from trader.polymarket_client import PolymarketClient
from trader.trade import trade_symbol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.unsafe
@pytest.mark.asyncio
async def test_e2e_wss_to_trade():
    """End-to-end test: simulate WSS messages → model evaluation → trade execution."""
    config = Config.load()

    config.symbols = ["NVDA"]
    config.model_loop_interval_ms = 100
    config.max_total_amount = 0.10
    config.min_edge = 0.05
    config.min_confidence_score = 0.3

    option_store = OptionStore()
    event_store = EventStore()
    quote_queue: asyncio.Queue[OptionQuoteEvent] = asyncio.Queue(maxsize=10000)

    orchestrator = Orchestrator(
        config=config,
        option_store=option_store,
        event_store=event_store,
        quote_queue=quote_queue,
    )

    # Refresh EventStore from Polymarket API
    count = event_store.refresh()
    if count == 0:
        pytest.skip("No Polymarket events available")

    # Find valid expiry
    expiries = sorted(event_store.get_expiration_dates())
    if not expiries:
        pytest.skip("No expiration dates in EventStore")

    today = datetime.now(UTC).date()
    cutoff = today + timedelta(days=config.max_days)
    valid_expiries = [e for e in expiries if today <= e.date() <= cutoff]
    if not valid_expiries:
        pytest.skip("No valid expiries within max_days")

    target_expiry = valid_expiries[0]
    expiry_str = target_expiry.strftime("%y%m%d")

    event = event_store.get_by_symbol_and_end_date("NVDA", target_expiry)
    if not event or not event.markets:
        pytest.skip("No NVDA markets for target expiry")

    strikes = sorted([m.strike_price for m in event.markets])
    ts = datetime.now(UTC)

    # Explicit option quotes simulating WSS messages
    quotes = [
        OptionQuoteEvent(occ_symbol=f"O:NVDA{expiry_str}C{int(strikes[0] * 1000):08d}", bid=10.0, ask=10.2, ts=ts),
        OptionQuoteEvent(occ_symbol=f"O:NVDA{expiry_str}P{int(strikes[0] * 1000):08d}", bid=0.5, ask=0.7, ts=ts),
        OptionQuoteEvent(occ_symbol=f"O:NVDA{expiry_str}C{int(strikes[1] * 1000):08d}", bid=8.0, ask=8.2, ts=ts),
        OptionQuoteEvent(occ_symbol=f"O:NVDA{expiry_str}P{int(strikes[1] * 1000):08d}", bid=1.0, ask=1.2, ts=ts),
        OptionQuoteEvent(occ_symbol=f"O:NVDA{expiry_str}C{int(strikes[2] * 1000):08d}", bid=6.0, ask=6.2, ts=ts),
        OptionQuoteEvent(occ_symbol=f"O:NVDA{expiry_str}P{int(strikes[2] * 1000):08d}", bid=2.0, ask=2.2, ts=ts),
    ]

    # Push to queue (simulating WSS arrival)
    for quote in quotes:
        await quote_queue.put(quote)

    # Process quotes into OptionStore (state builder)
    while not quote_queue.empty():
        quote = await quote_queue.get()
        option_store.apply_quote(quote)
        quote_queue.task_done()

    # Initialize client and run trade cycle
    orchestrator._client = PolymarketClient(
        api_key=config.polymarket_api_key,
        wallet_address=config.polymarket_wallet,
    )
    orchestrator._running = True

    await trade_symbol(
        symbol="NVDA",
        config=config,
        client=orchestrator._client,
        event_store=event_store,
        option_store=option_store,
    )
