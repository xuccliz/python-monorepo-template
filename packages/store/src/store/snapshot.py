"""
Expiry-level option surface snapshot builder.

Builds immutable option surface snapshots from OptionState data.
Used as input to implied probability models.
"""

from collections.abc import Iterable
from datetime import datetime

from domain.models import OptionPoint, OptionState, OptionSurfaceSnapshot
from domain.types import Symbol


def build_surface_snapshot(
    *,
    states: Iterable[OptionState],
    symbol: Symbol,
    expiration_date: datetime,
    max_spread: float | None = None,
) -> OptionSurfaceSnapshot:
    """
    Build an option surface snapshot for one symbol + expiry.

    Parameters
    ----------
    states:
        Iterable of OptionState (typically from OptionStore)
    symbol:
        Stock symbol (e.g. "AAPL")
    expiration_date:
        Expiry datetime (21:00 UTC = 4PM ET market close)
    max_spread:
        Optional filter to drop illiquid quotes

    Returns
    -------
    OptionSurfaceSnapshot
    """

    calls: list[OptionPoint] = []
    puts: list[OptionPoint] = []

    for s in states:
        if s.symbol != symbol or s.expiration_date != expiration_date:
            continue

        if max_spread is not None and s.spread > max_spread:
            continue

        point = OptionPoint(
            strike=s.strike_price,
            option_type=s.option_type,
            bid=s.bid,
            ask=s.ask,
            mid=s.mid,
            spread=s.spread,
        )

        if s.option_type == "call":
            calls.append(point)
        else:
            puts.append(point)

    # Sort by strike (ascending)
    calls.sort(key=lambda p: p.strike)
    puts.sort(key=lambda p: p.strike)

    return OptionSurfaceSnapshot(
        symbol=symbol,
        expiration_date=expiration_date,
        calls=tuple(calls),
        puts=tuple(puts),
    )
