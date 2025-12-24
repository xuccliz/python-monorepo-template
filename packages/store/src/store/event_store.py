"""
In-memory store for Polymarket stock price events.

Single-writer architecture:
- refresh() fetches and updates all events from Gamma API
- Other components get read-only views

No locks needed — asyncio is single-threaded.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from domain.models import EventMetadata, MarketMetadata
from domain.types import Symbol

from store.fetch_events import fetch_stock_events

logger = logging.getLogger(__name__)


class EventReader(Protocol):
    """Read-only view of the event store."""

    def get_by_symbol(self, symbol: Symbol) -> list[EventMetadata]: ...
    def get_all(self) -> list[EventMetadata]: ...
    def last_refresh(self) -> datetime | None: ...
    def get_polymarket_prob(
        self, symbol: Symbol, end_date: str, strike_price: float, direction: Literal["above", "below"] | None = None
    ) -> float | None: ...
    def get_market(self, symbol: Symbol, end_date: str, strike_price: float) -> MarketMetadata | None: ...


@dataclass
class EventStore:
    """
    In-memory store for Polymarket stock events.

    Use refresh() to fetch latest events from Gamma API.
    Read methods provide access to stored events by symbol.
    """

    _events: dict[str, list[EventMetadata]] = field(default_factory=dict)
    _last_refresh: datetime | None = None
    _refresh_interval_seconds: int = 300  # 5 minutes default

    def refresh(self) -> int:
        """Fetch latest events from Gamma API and update store. Returns count."""
        events = fetch_stock_events()

        new_events: dict[str, list[EventMetadata]] = {}
        for event in events:
            if event.symbol:
                if event.symbol not in new_events:
                    new_events[event.symbol] = []
                new_events[event.symbol].append(event)

        self._events = new_events

        total_events = sum(len(v) for v in self._events.values())
        self._last_refresh = datetime.now(UTC)
        logger.info("Refreshed event store: %d events for %d symbols", total_events, len(self._events))
        return total_events

    async def start_refresh_loop(self) -> None:
        """Start background refresh loop (non-blocking)."""
        while True:
            try:
                await asyncio.to_thread(self.refresh)
            except Exception:
                logger.exception("Failed to refresh events")
            await asyncio.sleep(self._refresh_interval_seconds)

    def clear(self) -> None:
        """Clear all stored events."""
        self._events.clear()
        self._last_refresh = None

    # --- Read interface (EventReader) ---

    def get_polymarket_prob(
        self, symbol: Symbol, end_date: str, strike_price: float, direction: Literal["above", "below"] | None = None
    ) -> float | None:
        """Get Polymarket probability for a specific stock symbol, end date, and strike price."""
        market = self.get_market(symbol, end_date, strike_price)
        if not market:
            return None

        if direction == "above":
            return market.yes_price
        else:
            return market.no_price

    def get_market(self, symbol: Symbol, end_date: str, strike_price: float) -> MarketMetadata | None:
        """Get market for a specific stock symbol, end date, and strike price."""
        events = self._events.get(symbol, [])
        for event in events:
            if event.end_date == end_date and event.markets:
                return next(
                    (market for market in event.markets if market.strike_price == strike_price),
                    None,
                )
        return None

    def get_by_symbol(self, symbol: Symbol) -> list[EventMetadata]:
        """Get all events for a specific stock symbol."""
        return self._events.get(symbol, [])

    def get_all(self) -> list[EventMetadata]:
        """Get all stored events."""
        return [event for events in self._events.values() for event in events]

    def last_refresh(self) -> datetime | None:
        """Get timestamp of last successful refresh."""
        return self._last_refresh
