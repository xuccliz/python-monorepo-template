"""
Expiry-level option surface snapshot builder.

Builds immutable option surface snapshots from OptionStore.
Optimized for trader hot-path using indexed access.
"""

from datetime import datetime

from domain.models import OptionPoint, OptionSurfaceSnapshot
from domain.types import Symbol

from store.option_store import OptionStore


def build_surface_snapshot(
    *,
    store: OptionStore,
    symbol: Symbol,
    expiration_date: datetime,
    max_spread: float | None = None,
) -> OptionSurfaceSnapshot:
    """
    Build an option surface snapshot for one symbol + expiry.

    Uses OptionStore's secondary index for O(1) access.
    """

    calls: list[OptionPoint] = []
    puts: list[OptionPoint] = []

    strikes = store.get_strikes(symbol, expiration_date)
    if not strikes:
        return OptionSurfaceSnapshot(
            symbol=symbol,
            expiration_date=expiration_date,
            calls=(),
            puts=(),
        )

    for strike in strikes:
        call, put = store.get_pair(symbol, expiration_date, strike)

        if call is not None:
            if max_spread is None or call.spread <= max_spread:
                calls.append(
                    OptionPoint(
                        strike_price=call.strike_price,
                        option_type="call",
                        bid=call.bid,
                        ask=call.ask,
                        mid=call.mid,
                        spread=call.spread,
                    )
                )

        if put is not None:
            if max_spread is None or put.spread <= max_spread:
                puts.append(
                    OptionPoint(
                        strike_price=put.strike_price,
                        option_type="put",
                        bid=put.bid,
                        ask=put.ask,
                        mid=put.mid,
                        spread=put.spread,
                    )
                )

    return OptionSurfaceSnapshot(
        symbol=symbol,
        expiration_date=expiration_date,
        calls=tuple(calls),
        puts=tuple(puts),
    )
