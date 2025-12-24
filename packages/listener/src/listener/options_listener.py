"""
Minimal Options Quote Listener (Polygon / Massive)

Purpose:
- Fetch options contracts for underlying tickers
- Subscribe to option quotes
- Update shared state store with latest quotes
- Optionally push events to an asyncio.Queue
"""

import asyncio
import logging
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

from domain.models import OptionQuoteEvent
from domain.secrets import load_required_secret
from domain.types import SYMBOLS
from dotenv import load_dotenv
from massive import RESTClient, WebSocketClient
from massive.websocket.models import EquityQuote, Feed, Market, WebSocketMessage
from store.option_store import OptionStore
from urllib3 import HTTPResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("options_listener")


def fetch_options_symbols(
    api_key: str,
    tickers: Iterable[str],
    *,
    expiration_date_gte: date | None = None,
    expiration_date_lte: date | None = None,
    limit_per_ticker: int = 100,
) -> list[str]:
    """
    Fetch options contract symbols for given underlying tickers.

    Args:
        api_key: Massive API key
        tickers: Underlying stock tickers (e.g., ["AAPL", "MSFT"])
        expiration_date_gte: Min expiration date (default: today)
        expiration_date_lte: Max expiration date (default: 30 days out)
        limit_per_ticker: Max contracts per ticker

    Returns:
        List of options symbols (e.g., ["O:AAPL250117C00180000", ...])
    """
    client = RESTClient(api_key=api_key)

    if expiration_date_gte is None:
        expiration_date_gte = datetime.now(UTC).date()
    if expiration_date_lte is None:
        expiration_date_lte = datetime.now(UTC).date() + timedelta(days=30)

    symbols: list[str] = []

    for ticker in tickers:
        try:
            result = client.list_options_contracts(
                underlying_ticker=ticker,
                expiration_date_gte=expiration_date_gte,
                expiration_date_lte=expiration_date_lte,
                expired=False,
                limit=limit_per_ticker,
            )

            # list_options_contracts returns HTTPResponse when raw=True,
            # Iterator[OptionsContract] when raw=False (default)
            if isinstance(result, HTTPResponse):
                logger.error("Unexpected raw response for %s", ticker)
                continue

            count = 0
            for contract in result:
                ticker = getattr(contract, "ticker", None)
                if ticker:
                    symbols.append(ticker)
                    count += 1
                    if count >= limit_per_ticker:
                        break

            logger.info("Fetched %d contracts for %s", count, ticker)

        except Exception:
            logger.exception("Failed to fetch contracts for %s", ticker)

    logger.info("Total options symbols: %d", len(symbols))
    return symbols


class OptionsQuoteListener:
    def __init__(
        self,
        *,
        state_store: OptionStore,
        event_queue: asyncio.Queue[OptionQuoteEvent] | None = None,
        option_symbols: Iterable[str] | None = None,
        tickers: Iterable[str] | None = None,
    ):
        """
        Initialize the options quote listener.

        Args:
            state_store: Store to update with latest quotes (required).
            event_queue: Optional queue to push quote events to.
            option_symbols: Explicit list of option symbols to subscribe to.
                           If not provided, fetches from tickers.
            tickers: Underlying tickers to fetch options for.
                    Defaults to TICKERS from domain.constants.
        """
        self.api_key = load_required_secret("MASSIVE_API_KEY")
        self.state_store = state_store
        self.event_queue = event_queue

        if option_symbols:
            self.option_symbols = list(option_symbols)
        else:
            tickers = list(tickers) if tickers else SYMBOLS
            self.option_symbols = fetch_options_symbols(self.api_key, tickers)

        self._client: WebSocketClient | None = None
        self._msg_count = 0

        logger.info(
            "Initialized listener | symbols=%d",
            len(self.option_symbols),
        )

    def _build_subscriptions(self) -> list[str]:
        return [f"Q.{symbol}" for symbol in self.option_symbols]

    async def _handle_message(self, messages: list[WebSocketMessage]) -> None:
        for msg in messages:
            # Options quotes come as EquityQuote messages
            if not isinstance(msg, EquityQuote):
                continue

            if msg.event_type != "Q":
                continue

            self._msg_count += 1

            try:
                symbol = msg.symbol
                if symbol is None:
                    continue

                bid = msg.bid_price or 0.0
                ask = msg.ask_price or 0.0
                timestamp = msg.timestamp
                if timestamp is None:
                    continue

                ts = datetime.fromtimestamp(timestamp / 1000, tz=UTC)

                # Create quote event
                quote = OptionQuoteEvent(
                    occ_symbol=symbol,
                    bid=bid,
                    ask=ask,
                    ts=ts,
                )

                # Let the store handle state building (single-writer)
                self.state_store.apply_quote(quote)

                # Optionally push to queue for downstream consumers
                if self.event_queue is not None:
                    self.event_queue.put_nowait(quote)

            except Exception:
                logger.exception("Failed to process quote: %s", msg)

    async def run(self) -> None:
        subs = self._build_subscriptions()
        if not subs:
            raise RuntimeError("No option symbols to subscribe to")

        logger.info("Starting WebSocket | subs=%d", len(subs))

        client = WebSocketClient(
            api_key=self.api_key,
            feed=Feed.RealTime,
            market=Market.Options,
            subscriptions=subs,
        )
        self._client = client

        try:
            await client.connect(processor=self._handle_message)
        except asyncio.CancelledError:
            logger.info("Listener cancelled")
        finally:
            logger.info(
                "Listener stopped | messages=%d | states=%d",
                self._msg_count,
                self.state_store.count(),
            )

    async def close(self) -> None:
        if self._client:
            await self._client.close()
