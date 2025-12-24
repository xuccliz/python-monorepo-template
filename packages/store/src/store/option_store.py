"""
Read-only state store for options data with trader-optimized indexing.

Single-writer architecture:
- Only the store owner writes via apply_quote()
- Store owns merge logic and indexing
- Other components get read-only views

Design:
- Primary store: flat dict keyed by OCC symbol (source of truth)
- Secondary index: symbol -> expiry -> strike -> {call, put}
- Expiration index: fast access to all expiries
- No locks needed — asyncio is single-threaded
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from domain.models import OptionQuoteEvent, OptionState
from domain.types import Symbol

from .parse_occ import parse_occ_symbol

# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class StateReader(Protocol):
    """Read-only view of the option state."""

    def get(self, occ_symbol: str) -> OptionState | None: ...
    def get_by_symbol(self, symbol: Symbol) -> list[OptionState]: ...
    def get_expiries(self, symbol: Symbol) -> list[datetime]: ...
    def get_expiration_dates(self) -> set[datetime]: ...
    def get_strikes(self, symbol: Symbol, expiry: datetime) -> list[float]: ...
    def get_pair(
        self, symbol: Symbol, expiry: datetime, strike: float
    ) -> tuple[OptionState | None, OptionState | None]: ...
    def count(self) -> int: ...


class StateWriter(Protocol):
    """Write interface — only the store owner should use this."""

    def apply_quote(self, quote: OptionQuoteEvent) -> OptionState | None: ...
    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class OptionStore:
    """
    In-memory option state store with trader-optimized indexing.

    Primary store (_states):
        OCC symbol -> OptionState

    Secondary index (_index):
        symbol -> expiry -> strike -> {"call": state, "put": state}

    Expiration index (_expirations):
        set of all expiries seen (across symbols)

    Single writer, many readers.
    """

    _states: dict[str, OptionState] = field(default_factory=dict)

    _index: dict[
        Symbol,
        dict[
            datetime,
            dict[
                float,
                dict[str, OptionState],
            ],
        ],
    ] = field(default_factory=dict)

    _expirations: set[datetime] = field(default_factory=set)

    _last_write: datetime | None = field(default=None)

    # ---------------------------------------------------------------------
    # Write interface
    # ---------------------------------------------------------------------

    def apply_quote(self, quote: OptionQuoteEvent) -> OptionState | None:
        """
        Apply a raw quote event and update internal state.

        - Validates quote
        - Parses OCC symbol
        - Computes mid and spread
        - Updates primary store and secondary index

        Returns updated OptionState, or None if invalid.
        """
        # Basic quote sanity checks
        if quote.bid < 0 or quote.ask < 0 or quote.bid > quote.ask:
            return None

        parsed_occ = parse_occ_symbol(quote.occ_symbol)
        if parsed_occ is None:
            return None

        mid = 0.5 * (quote.bid + quote.ask)
        spread = quote.ask - quote.bid

        state = OptionState(
            occ_symbol=quote.occ_symbol,
            symbol=parsed_occ.symbol,
            expiration_date=parsed_occ.expiration_date,
            strike_price=parsed_occ.strike_price,
            option_type=parsed_occ.option_type,
            bid=quote.bid,
            ask=quote.ask,
            mid=mid,
            spread=spread,
            last_updated=quote.ts,
        )

        # --- update primary store ---
        self._states[quote.occ_symbol] = state

        # --- update secondary index ---
        by_symbol = self._index.setdefault(parsed_occ.symbol, {})
        by_expiry = by_symbol.setdefault(parsed_occ.expiration_date, {})
        by_strike = by_expiry.setdefault(parsed_occ.strike_price, {})
        by_strike[parsed_occ.option_type] = state

        # --- update expiration index ---
        self._expirations.add(parsed_occ.expiration_date)

        # --- update last write timestamp ---
        self._last_write = datetime.now(UTC)

        return state

    def clear(self) -> None:
        """Clear all stored state."""
        self._states.clear()
        self._index.clear()
        self._expirations.clear()
        self._last_write = None

    def last_write(self) -> datetime | None:
        """Timestamp of last write to the store."""
        return self._last_write

    # ---------------------------------------------------------------------
    # Read interface (trader / modeler)
    # ---------------------------------------------------------------------

    def get(self, occ_symbol: str) -> OptionState | None:
        """Get state by OCC symbol."""
        return self._states.get(occ_symbol)

    def get_by_symbol(self, symbol: Symbol) -> list[OptionState]:
        """Get all option states for a symbol."""
        out: list[OptionState] = []
        by_symbol = self._index.get(symbol)
        if not by_symbol:
            return out

        for by_expiry in by_symbol.values():
            for by_strike in by_expiry.values():
                out.extend(by_strike.values())

        return out

    def get_expiries(self, symbol: Symbol) -> list[datetime]:
        """Get all expiries for a specific symbol."""
        return sorted(self._index.get(symbol, {}).keys())

    def get_expiration_dates(self) -> set[datetime]:
        """
        Get all expiration dates across all symbols.
        """
        return set(self._expirations)

    def get_strikes(self, symbol: Symbol, expiry: datetime) -> list[float]:
        """Get all strikes for a symbol and expiry."""
        return sorted(self._index.get(symbol, {}).get(expiry, {}).keys())

    def get_pair(
        self,
        symbol: Symbol,
        expiry: datetime,
        strike: float,
    ) -> tuple[OptionState | None, OptionState | None]:
        """
        Get (call, put) OptionState pair for a symbol/expiry/strike.
        """
        pair = self._index.get(symbol, {}).get(expiry, {}).get(strike)
        if not pair:
            return None, None

        return pair.get("call"), pair.get("put")

    def count(self) -> int:
        """Total number of tracked option contracts."""
        return len(self._states)
