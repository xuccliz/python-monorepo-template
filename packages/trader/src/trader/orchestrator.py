"""
Trading bot orchestrator.

Coordinates four long-running asyncio tasks:
1. Options listener - pushes OptionQuoteEvent into a queue
2. Options state builder - consumes quotes, updates OptionStore
3. Polymarket refresh loop - periodically refreshes EventStore
4. Model/decision loop - evaluates models, compares to Polymarket, triggers trades
"""

import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime

from domain.models import OptionQuoteEvent
from domain.paths import EVENT_STORE_HEARTBEAT, OPTION_STORE_HEARTBEAT
from domain.utils import is_market_open
from listener import OptionsQuoteListener
from store import EventStore, OptionStore

from trader.config import Config
from trader.polymarket_client import PolymarketClient
from trader.trade import trade_symbol
from trader.utils import has_recent_option_data, wait_for_market_open, write_heartbeat

logger = logging.getLogger(__name__)


@dataclass
class Orchestrator:
    """
    Main orchestrator coordinating all trading bot components.

    Components:
    - OptionStore: single-writer state store for options quotes
    - EventStore: Polymarket events catalog
    - OptionsQuoteListener: WebSocket listener for options quotes
    - PolymarketClient: executes trades on Polymarket
    """

    config: Config

    # Stores
    option_store: OptionStore = field(default_factory=OptionStore)
    event_store: EventStore = field(default_factory=EventStore)

    # Queue for quote events (listener -> state builder)
    quote_queue: asyncio.Queue[OptionQuoteEvent] = field(default_factory=lambda: asyncio.Queue(maxsize=10000))

    # Components (initialized in start())
    _listener: OptionsQuoteListener | None = field(default=None, init=False)
    _client: PolymarketClient | None = field(default=None, init=False)
    _tasks: list[asyncio.Task] = field(default_factory=list, init=False)
    _running: bool = field(default=False, init=False)

    async def start(self) -> None:
        """Start all orchestrator tasks."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting orchestrator...")
        self._running = True

        # Initialize Polymarket client
        self._client = PolymarketClient(
            api_key=self.config.polymarket_api_key,
            wallet_address=self.config.polymarket_wallet,
        )

        # Initialize listener
        self._listener = OptionsQuoteListener(
            event_queue=self.quote_queue,
            tickers=self.config.symbols,
        )

        # Initial Polymarket refresh
        logger.info("Initial Polymarket event refresh...")
        await asyncio.to_thread(self.event_store.refresh)

        # Start all tasks
        self._tasks = [
            asyncio.create_task(self._listener_task(), name="listener"),
            asyncio.create_task(self._state_builder_task(), name="state_builder"),
            asyncio.create_task(self._polymarket_refresh_task(), name="polymarket_refresh"),
            asyncio.create_task(self._trade_task(), name="trade"),
        ]

        logger.info("Orchestrator started with %d tasks", len(self._tasks))

    async def stop(self) -> None:
        """Stop all orchestrator tasks gracefully."""
        if not self._running:
            return

        logger.info("Stopping orchestrator...")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close listener
        if self._listener:
            await self._listener.close()

        self._tasks.clear()
        logger.info("Orchestrator stopped")

    async def run_forever(self) -> None:
        """Run the orchestrator until interrupted."""
        await self.start()
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        finally:
            await self.stop()

    # -------------------------------------------------------------------------
    # Task 1: Options Listener
    # -------------------------------------------------------------------------

    async def _listener_task(self) -> None:
        """
        Options listener task.

        Connects to WebSocket and pushes OptionQuoteEvent into the queue.
        """
        logger.info("Starting options listener task")
        try:
            if self._listener:
                await self._listener.run()
        except asyncio.CancelledError:
            logger.info("Listener task cancelled")
        except Exception:
            logger.exception("Listener task failed")
            raise

    # -------------------------------------------------------------------------
    # Task 2: State Builder (consumes queue, writes to store)
    # -------------------------------------------------------------------------

    async def _state_builder_task(self) -> None:
        """
        State builder task (single writer to OptionStore).

        Consumes OptionQuoteEvent from queue and applies to OptionStore.
        This is the only task that writes to the store.
        """
        logger.info("Starting state builder task")
        processed = 0
        try:
            while self._running:
                try:
                    quote = await asyncio.wait_for(self.quote_queue.get(), timeout=1.0)
                    self.option_store.apply_quote(quote)
                    processed += 1
                    if processed % 1000 == 0:
                        logger.info(
                            "Processed %d quotes | store size: %d | queue: %d",
                            processed,
                            self.option_store.count(),
                            self.quote_queue.qsize(),
                        )
                        write_heartbeat(OPTION_STORE_HEARTBEAT)
                    self.quote_queue.task_done()
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info("State builder task cancelled | processed: %d", processed)

    # -------------------------------------------------------------------------
    # Task 3: Polymarket Refresh Loop
    # -------------------------------------------------------------------------

    async def _polymarket_refresh_task(self) -> None:
        """
        Polymarket refresh task.

        Periodically refreshes EventStore with latest Polymarket events.
        Uses asyncio.to_thread to avoid blocking the event loop.
        """
        logger.info("Starting Polymarket refresh task (interval: %ds)", self.config.polymarket_refresh_interval_s)
        try:
            while self._running:
                await asyncio.sleep(self.config.polymarket_refresh_interval_s)
                try:
                    count = await asyncio.to_thread(self.event_store.refresh)
                    logger.info("Polymarket refresh complete: %d events", count)
                    write_heartbeat(EVENT_STORE_HEARTBEAT)
                except Exception:
                    logger.exception("Polymarket refresh failed")
        except asyncio.CancelledError:
            logger.info("Polymarket refresh task cancelled")

    # -------------------------------------------------------------------------
    # Task 4: Model / Decision Loop
    # -------------------------------------------------------------------------

    async def _trade_task(self) -> None:
        """
        Model and decision task.

        Wakes up every N milliseconds to:
        1. Build snapshots for each symbol/expiry
        2. Evaluate probability models
        3. Compare to Polymarket prices
        4. Trigger trades when edge is found

        Sleeps until market open when market is closed.
        """
        interval_s = self.config.model_loop_interval_ms / 1000.0
        logger.info("Starting model/decision task (interval: %dms)", self.config.model_loop_interval_ms)

        try:
            while self._running:
                # Sleep until market open if closed
                if not is_market_open():
                    await wait_for_market_open()
                    continue

                # Wait for fresh option data after market open
                if not has_recent_option_data(self.option_store):
                    logger.info("Waiting for fresh option data...")
                    await asyncio.sleep(5)
                    continue

                loop_start = datetime.now(UTC)

                if self._client is None:
                    logger.warning("Polymarket client not initialized")
                    await asyncio.sleep(5)
                    continue

                try:
                    await asyncio.gather(
                        *[
                            trade_symbol(
                                symbol=symbol,
                                config=self.config,
                                client=self._client,
                                event_store=self.event_store,
                                option_store=self.option_store,
                            )
                            for symbol in self.config.symbols
                        ],
                        return_exceptions=True,
                    )
                except Exception:
                    logger.exception("Model evaluation failed")

                # Sleep for remaining interval
                elapsed = (datetime.now(UTC) - loop_start).total_seconds()
                sleep_time = max(0, interval_s - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Model/decision task cancelled")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


async def main() -> None:
    """Main entry point for the trading bot."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = Config.load()

    orchestrator = Orchestrator(config=config)

    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(orchestrator.stop()))

    await orchestrator.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
