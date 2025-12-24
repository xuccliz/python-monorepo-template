"""
Read-only state store for options data.

Single-writer architecture:
- Only the store owner (modeler) writes via apply_quote()
- Store owns the merge logic — callers pass raw quotes, not state
- Other components get read-only views

No locks needed — asyncio is single-threaded.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from domain.models import OptionQuoteEvent, OptionState
from domain.types import Symbol, is_symbol, make_expiry_datetime

# OCC symbol pattern: O:NVDA260117C00140000
# Symbol: NVDA, Date: 260117 (YYMMDD), Type: C/P, Strike: 00140000 (price * 1000)
_OCC_PATTERN = re.compile(
    r"^O:(?P<symbol>[A-Z]+)"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<type>[CP])"
    r"(?P<strike>\d{8})$"
)


def parse_occ_symbol(occ_symbol: str) -> tuple[Symbol, datetime, str, float] | None:
    """
    Parse OCC option symbol.

    Returns (symbol, expiration_date, option_type, strike) or None if invalid.
    """
    match = _OCC_PATTERN.match(occ_symbol)
    if not match:
        return None

    symbol_str = match.group("symbol")
    if not is_symbol(symbol_str):
        return None

    yy, mm, dd = match.group("yy"), match.group("mm"), match.group("dd")
    expiration_date = make_expiry_datetime(f"20{yy}-{mm}-{dd}")
    option_type = "call" if match.group("type") == "C" else "put"
    strike = int(match.group("strike")) / 1000.0

    return symbol_str, expiration_date, option_type, strike


class StateReader(Protocol):
    """Read-only view of the state store."""

    def get(self, occ_symbol: str) -> OptionState | None: ...
    def get_all(self) -> dict[str, OptionState]: ...
    def get_by_symbol(self, symbol: Symbol) -> list[OptionState]: ...
    def get_by_strike(self, symbol: Symbol, strike: float) -> list[OptionState]: ...
    def get_strikes(self, symbol: Symbol) -> list[float]: ...
    def count(self) -> int: ...


class StateWriter(Protocol):
    """Write interface — only the store owner should use this."""

    def apply_quote(self, quote: OptionQuoteEvent) -> OptionState | None: ...
    def clear(self) -> None: ...


@dataclass
class OptionStore:
    """
    In-memory state store with separated read/write interfaces.

    The store owns merge logic:
    - apply_quote() accepts raw quotes and updates internal state
    - Handles symbol parsing, mid/spread calculation
    - Returns the updated state (or None if symbol is invalid)

    No locks — designed for single-threaded asyncio.
    """

    _states: dict[str, OptionState] = field(default_factory=dict)

    def apply_quote(self, quote: OptionQuoteEvent) -> OptionState | None:
        """
        Apply a raw quote event and return the updated state.

        The store owns the merge logic:
        - Parses OCC symbol to extract symbol/strike/expiration_date/type
        - Calculates mid and spread
        - Merges with existing state (or creates new)

        Returns None if the symbol is invalid.
        """
        # Garbage check: reject invalid quotes
        if quote.bid < 0 or quote.ask < 0 or quote.bid > quote.ask:
            return None

        parsed = parse_occ_symbol(quote.occ_symbol)
        if parsed is None:
            return None

        symbol, expiration_date, option_type, strike = parsed

        mid = (quote.bid + quote.ask) / 2
        spread = quote.ask - quote.bid

        state = OptionState(
            occ_symbol=quote.occ_symbol,
            symbol=symbol,
            strike_price=strike,
            expiration_date=expiration_date,
            option_type=option_type,
            bid=quote.bid,
            ask=quote.ask,
            mid=mid,
            spread=spread,
            last_updated=quote.ts,
        )

        self._states[quote.occ_symbol] = state
        return state

    def clear(self) -> None:
        """Clear all states."""
        self._states.clear()

    # --- Read interface (StateReader) ---

    def get(self, occ_symbol: str) -> OptionState | None:
        """Get state for a specific OCC symbol."""
        return self._states.get(occ_symbol)

    def get_all(self) -> dict[str, OptionState]:
        """Get all states (returns a shallow copy)."""
        return dict(self._states)

    def get_by_symbol(self, symbol: Symbol) -> list[OptionState]:
        """Get all states for a stock symbol."""
        return [s for s in self._states.values() if s.symbol == symbol]

    def get_by_strike(self, symbol: Symbol, strike: float) -> list[OptionState]:
        """Get call and put for a specific symbol + strike."""
        return [s for s in self._states.values() if s.symbol == symbol and s.strike_price == strike]

    def get_strikes(self, symbol: Symbol) -> list[float]:
        """Get all unique strikes for a symbol."""
        strikes = {s.strike_price for s in self._states.values() if s.symbol == symbol}
        return sorted(strikes)

    def count(self) -> int:
        """Get total number of tracked options."""
        return len(self._states)
