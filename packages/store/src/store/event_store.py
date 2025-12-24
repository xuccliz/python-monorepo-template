"""
In-memory store for Polymarket stock price events.

Single-writer architecture:
- refresh() fetches and updates all events from Gamma API
- Atomic swap on refresh
- Other components get read-only views
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from domain.models import EventMetadata, MarketMetadata
from domain.types import Symbol

from store.fetch_events import fetch_stock_events

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read interface
# ---------------------------------------------------------------------------


class EventReader(Protocol):
    """Read-only view of the event store."""

    def get_by_symbol(self, symbol: Symbol) -> list[EventMetadata]: ...
    def get_by_symbol_and_end_date(self, symbol: Symbol, end_date: datetime) -> EventMetadata | None: ...
    def get_all(self) -> list[EventMetadata]: ...
    def get_expiration_dates(self) -> set[datetime]: ...
    def last_refresh(self) -> datetime | None: ...

    def get_market(self, symbol: Symbol, end_date: datetime, strike_price: float) -> MarketMetadata | None: ...

    def get_polymarket_prob(
        self,
        symbol: Symbol,
        end_date: datetime,
        strike_price: float,
        direction: Literal["above", "below"],
    ) -> float | None: ...


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class EventStore:
    """
    In-memory store for Polymarket stock events.

    Primary storage:
        symbol -> list[EventMetadata]

    Secondary index (trader-optimized):
        (symbol, end_date, strike_price) -> MarketMetadata

    Designed for small–medium reference data (~100–1000 events).
    """

    _events: dict[Symbol, list[EventMetadata]] = field(default_factory=dict)
    _event_index: dict[tuple[Symbol, datetime], EventMetadata] = field(default_factory=dict)
    _market_index: dict[tuple[Symbol, datetime, float], MarketMetadata] = field(default_factory=dict)
    _expiration_dates: set[datetime] = field(default_factory=set)
    _last_refresh: datetime | None = None

    # ---------------------------------------------------------------------
    # Write path (single writer)
    # ---------------------------------------------------------------------

    def refresh(self) -> int:
        """
        Fetch latest events from Gamma API and update store.

        Returns total number of events.
        """
        events = fetch_stock_events()

        new_events: dict[Symbol, list[EventMetadata]] = {}
        new_event_index: dict[tuple[Symbol, datetime], EventMetadata] = {}
        new_index: dict[tuple[Symbol, datetime, float], MarketMetadata] = {}
        new_expirations: set[datetime] = set()

        for event in events:
            if not event.symbol or not event.end_date:
                continue

            new_events.setdefault(event.symbol, []).append(event)
            new_event_index[(event.symbol, event.end_date)] = event
            new_expirations.add(event.end_date)

            if not event.markets:
                continue

            for market in event.markets:
                if market.strike_price is None:
                    continue
                key = (event.symbol, event.end_date, market.strike_price)
                new_index[key] = market

        # Atomic swap
        self._events = new_events
        self._event_index = new_event_index
        self._market_index = new_index
        self._expiration_dates = new_expirations
        self._last_refresh = datetime.now(UTC)

        total_events = sum(len(v) for v in self._events.values())
        logger.info(
            "Refreshed EventStore: %d events | %d symbols | %d expirations",
            total_events,
            len(self._events),
            len(self._expiration_dates),
        )
        return total_events

    def clear(self) -> None:
        """Clear all stored events."""
        self._events.clear()
        self._event_index.clear()
        self._market_index.clear()
        self._expiration_dates.clear()
        self._last_refresh = None

    # ---------------------------------------------------------------------
    # Read interface
    # ---------------------------------------------------------------------

    def get_by_symbol(self, symbol: Symbol) -> list[EventMetadata]:
        """Get all events for a symbol."""
        return self._events.get(symbol, [])

    def get_by_symbol_and_end_date(self, symbol: Symbol, end_date: datetime) -> EventMetadata | None:
        """Get event for symbol + end_date (O(1) lookup)."""
        return self._event_index.get((symbol, end_date))

    def get_all(self) -> list[EventMetadata]:
        """Get all events across all symbols."""
        return [e for events in self._events.values() for e in events]

    def get_expiration_dates(self) -> set[datetime]:
        """Get all Polymarket expiration dates (datetimes)."""
        return set(self._expiration_dates)

    def get_market(
        self,
        symbol: Symbol,
        end_date: datetime,
        strike_price: float,
    ) -> MarketMetadata | None:
        """Get market metadata for symbol + expiration + strike."""
        return self._market_index.get((symbol, end_date, strike_price))

    def get_polymarket_prob(
        self,
        symbol: Symbol,
        end_date: datetime,
        strike_price: float,
        direction: Literal["above", "below"],
    ) -> float | None:
        """
        Get Polymarket-implied probability.

        direction="above" -> YES price
        direction="below" -> NO price
        """
        market = self.get_market(symbol, end_date, strike_price)
        if not market:
            return None

        return market.yes_price if direction == "above" else market.no_price

    def last_refresh(self) -> datetime | None:
        """Timestamp of last successful refresh."""
        return self._last_refresh
