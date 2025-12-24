"""
Historical data fetcher for model evaluation.

Uses local SQLite cache for historical options and stock data.
Run 'task fetch:historical' to populate the cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from domain.models import OptionPoint, OptionSurfaceSnapshot
from domain.types import Symbol

from .database import (
    get_cached_contracts,
    get_cached_expirations,
    get_cached_option_agg,
    get_cached_stock_close,
)


@dataclass(frozen=True, slots=True)
class HistoricalOptionsData:
    """Historical options data for a symbol on a specific date."""

    symbol: Symbol
    data_date: date
    expiration_date: date
    calls: list[OptionPoint]
    puts: list[OptionPoint]


class DataNotCachedError(Exception):
    """Raised when required data is not in the local cache."""

    pass


def fetch_stock_close(symbol: str, target_date: date) -> float:
    """
    Get the closing price of a stock on a specific date from cache.

    Args:
        symbol: Stock ticker (e.g., "NVDA")
        target_date: The date to get closing price for

    Returns:
        Closing price

    Raises:
        DataNotCachedError: If data is not in cache
    """
    cached = get_cached_stock_close(symbol, target_date)
    if cached is not None:
        return cached

    raise DataNotCachedError(
        f"Stock data for {symbol} on {target_date} not in cache. "
        f"Run 'task fetch:historical -- --symbols {symbol}' to populate."
    )


class CachedAgg:
    """Simple object to hold cached OHLC data."""

    def __init__(self, data: dict) -> None:
        self.open = data.get("open")
        self.high = data.get("high")
        self.low = data.get("low")
        self.close = data.get("close")
        self.volume = data.get("volume")


def fetch_option_aggs(
    occ_symbol: str,
    target_date: date,
) -> CachedAgg:
    """
    Get aggregated OHLC data for an option contract on a specific date from cache.

    Args:
        occ_symbol: OCC option symbol (e.g., "O:NVDA260117C00140000")
        target_date: The date to get data for

    Returns:
        CachedAgg object with OHLC data

    Raises:
        DataNotCachedError: If data is not in cache
    """
    cached = get_cached_option_agg(occ_symbol, target_date)
    if cached is not None:
        return CachedAgg(cached)

    raise DataNotCachedError(
        f"Option data for {occ_symbol} on {target_date} not in cache. "
        "Run 'task fetch:historical -- --options' to populate."
    )


def fetch_options_contracts_for_date(
    symbol: Symbol,
    expiration_date: date,
) -> list[dict]:
    """
    Get options contracts for a symbol and expiration from cache.

    Args:
        symbol: Stock ticker
        expiration_date: Option expiration date

    Returns:
        List of contract dicts with occ_symbol, strike, type
    """
    contracts = get_cached_contracts(symbol, expiration_date)
    if not contracts:
        raise DataNotCachedError(
            f"Option contracts for {symbol} exp={expiration_date} not in cache. "
            f"Run 'task fetch:historical -- --symbols {symbol} --options' to populate."
        )
    return contracts


def build_historical_snapshot(
    symbol: Symbol,
    expiration_date: date,
    data_date: date,
    max_spread: float | None = None,
) -> OptionSurfaceSnapshot | None:
    """
    Build an OptionSurfaceSnapshot from historical data.

    This reconstructs what the option surface looked like on a past date.

    Args:
        symbol: Stock ticker
        expiration_date: Option expiration date
        data_date: Historical date to reconstruct snapshot for
        max_spread: Optional max spread filter

    Returns:
        OptionSurfaceSnapshot or None if insufficient data
    """
    # Get contracts for this expiration
    contracts = fetch_options_contracts_for_date(symbol, expiration_date)

    if not contracts:
        return None

    calls: list[OptionPoint] = []
    puts: list[OptionPoint] = []

    for contract in contracts:
        occ_symbol = contract["occ_symbol"]
        strike = contract["strike"]
        opt_type = contract["type"]

        # Fetch historical OHLC for this contract
        try:
            agg = fetch_option_aggs(occ_symbol, data_date)
        except DataNotCachedError:
            continue

        if agg is None:
            continue

        # Bug #6 note: Use high/low as proxy for bid/ask
        # WARNING: Daily high/low capture intraday extremes, not bid/ask at any moment.
        # This systematically overestimates spreads (daily range > typical spread).
        # We use close price as a more stable mid estimate, and a fraction
        # of (high-low) as the spread proxy.
        high = float(agg.high) if agg.high else 0.0
        low = float(agg.low) if agg.low else 0.0
        close_price = float(agg.close) if agg.close else 0.0

        if high <= 0 or low <= 0 or close_price <= 0:
            continue

        # Use close as mid (more stable than (high+low)/2)
        mid = close_price
        # Use a fraction of daily range as spread proxy (typical spread << daily range)
        # Factor of 0.3 is heuristic based on liquid equity options
        spread = (high - low) * 0.3
        bid = mid - spread / 2
        ask = mid + spread / 2

        if max_spread is not None and spread > max_spread:
            continue

        point = OptionPoint(
            strike_price=strike,
            option_type=opt_type,
            bid=bid,
            ask=ask,
            mid=mid,
            spread=spread,
        )

        if opt_type == "call":
            calls.append(point)
        else:
            puts.append(point)

    if not calls or not puts:
        return None

    # Sort by strike
    calls = sorted(calls, key=lambda p: p.strike_price)
    puts = sorted(puts, key=lambda p: p.strike_price)

    return OptionSurfaceSnapshot(
        symbol=symbol,
        expiration_date=datetime.combine(expiration_date, datetime.min.time()).replace(tzinfo=UTC),
        calls=tuple(calls),
        puts=tuple(puts),
    )


def get_historical_expirations(
    symbol: Symbol,
    from_date: date,
    to_date: date,
) -> list[date]:
    """
    Get option expiration dates from cache.

    Args:
        symbol: Stock ticker
        from_date: Start of historical period
        to_date: End of historical period

    Returns:
        List of expiration dates
    """
    expirations = get_cached_expirations(symbol, from_date, to_date)
    if not expirations:
        raise DataNotCachedError(
            f"No expiration data for {symbol} between {from_date} and {to_date} in cache. "
            f"Run 'task fetch:historical -- --symbols {symbol} --options' to populate."
        )
    return expirations


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """A single evaluation case for backtesting."""

    symbol: Symbol
    prediction_date: date  # Date when prediction is made
    expiration_date: date  # Option expiration date
    strike_price: float  # Strike to evaluate


def generate_evaluation_cases(
    symbol: Symbol,
    prediction_date: date,
    days_to_expiry: int = 7,
    n_strikes: int = 5,
) -> list[EvaluationCase]:
    """
    Generate evaluation cases for a symbol on a prediction date.

    Args:
        symbol: Stock ticker
        prediction_date: Date to make predictions from
        days_to_expiry: Target days to expiration
        n_strikes: Number of strikes around ATM to include

    Returns:
        List of EvaluationCase objects
    """
    from .database import get_expirations_with_daily_data

    # Find expiration close to target days that has daily data
    target_expiry = prediction_date + timedelta(days=days_to_expiry)

    expirations = get_expirations_with_daily_data(
        symbol,
        data_date=prediction_date,
        from_date=prediction_date,
        to_date=prediction_date + timedelta(days=days_to_expiry + 14),
    )

    if not expirations:
        # Fall back to contract-only expirations
        expirations = get_historical_expirations(
            symbol,
            prediction_date,
            prediction_date + timedelta(days=days_to_expiry + 7),
        )

    if not expirations:
        return []

    # Find closest expiration to target
    expiration = min(expirations, key=lambda d: abs((d - target_expiry).days))

    # Get stock price on prediction date to find ATM strikes
    stock_price = fetch_stock_close(symbol, prediction_date)
    if stock_price is None:
        return []

    # Get available strikes
    contracts = fetch_options_contracts_for_date(symbol, expiration)
    strikes = sorted({c["strike"] for c in contracts})

    if not strikes:
        return []

    # Find strikes around ATM
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - stock_price))

    start_idx = max(0, atm_idx - n_strikes // 2)
    end_idx = min(len(strikes), start_idx + n_strikes)

    selected_strikes = strikes[start_idx:end_idx]

    return [
        EvaluationCase(
            symbol=symbol,
            prediction_date=prediction_date,
            expiration_date=expiration,
            strike_price=strike,
        )
        for strike in selected_strikes
    ]


def generate_synthetic_cases(
    symbol: Symbol,
    prediction_date: date,
    days_to_expiry: int = 7,
    n_strikes: int = 5,
) -> list[EvaluationCase]:
    """
    Generate synthetic evaluation cases based on stock price (when no option data available).

    Generates strikes +/- 10% around the stock price.
    """
    # Bug #5 fix: Guard against n_strikes < 2 which would cause division by zero
    if n_strikes < 2:
        n_strikes = 2

    # Bug #4 fix: Use a separate variable for the actual data date
    # Don't mutate the input prediction_date
    actual_data_date = prediction_date
    stock_price = None

    # Try to get stock price, looking back up to 5 days if valid trading day not found
    for i in range(5):
        try:
            d = prediction_date - timedelta(days=i)
            stock_price = fetch_stock_close(symbol, d)
            if stock_price is not None and stock_price > 0:
                actual_data_date = d  # Track which date we got data for
                break
        except DataNotCachedError:
            continue

    if stock_price is None or stock_price <= 0:
        return []

    # Generate synthetic expiration (use original prediction_date for expiry calc)
    expiration = prediction_date + timedelta(days=days_to_expiry)

    # Generate strikes +/- 10% around ATM
    min_strike = stock_price * 0.9
    max_strike = stock_price * 1.1
    step = (max_strike - min_strike) / (n_strikes - 1)

    strikes = [min_strike + i * step for i in range(n_strikes)]

    return [
        EvaluationCase(
            symbol=symbol,
            prediction_date=actual_data_date,  # Use the date we actually got data for
            expiration_date=expiration,
            strike_price=strike,
        )
        for strike in strikes
    ]
