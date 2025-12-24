"""
Helper functions for fetching and exploring options data in notebooks.
"""

from datetime import UTC, date, datetime, timedelta
from typing import get_args

from domain.secrets import load_required_secret
from domain.types import Symbol
from dotenv import load_dotenv
from massive import RESTClient

load_dotenv()
SYMBOLS = list(get_args(Symbol))


def get_client() -> RESTClient:
    """Get an authenticated Massive REST client."""
    return RESTClient(api_key=load_required_secret("MASSIVE_API_KEY"))


def fetch_options_occ_symbols(
    expiration_gte: date | None = None,
    expiration_lte: date | None = None,
    contract_type: str | None = None,  # "call" or "put"
    limit_per_symbol: int = 100,
) -> list[str]:
    """
    Fetch options contract OCC symbols for given stock symbols.

    Args:
        expiration_gte: Min expiration date (default: today)
        expiration_lte: Max expiration date (default: 30 days out)
        contract_type: Filter by "call" or "put" (default: both)
        limit_per_symbol: Max contracts per symbol

    Returns:
        List of OCC symbols (e.g., ["O:AAPL250117C00180000", ...])

    Example:
        >>> occ_symbols = fetch_options_occ_symbols(limit_per_symbol=10)
        >>> occ_symbols[:3]
        ['O:AAPL260102C00110000', 'O:AAPL260102C00120000', ...]
    """
    client = get_client()

    if expiration_gte is None:
        expiration_gte = datetime.now(UTC).date()
    if expiration_lte is None:
        expiration_lte = datetime.now(UTC).date() + timedelta(days=30)

    occ_symbols: list[str] = []

    for symbol in SYMBOLS:
        try:
            contracts = client.list_options_contracts(
                underlying_ticker=symbol,
                expiration_date_gte=expiration_gte,
                expiration_date_lte=expiration_lte,
                contract_type=contract_type,
                expired=False,
                limit=limit_per_symbol,
            )

            count = 0
            for contract in contracts:
                if contract.ticker:
                    occ_symbols.append(contract.ticker)
                    count += 1
                    if count >= limit_per_symbol:
                        break

            print(f"  {symbol}: {count} contracts")

        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")

    print(f"\nTotal: {len(occ_symbols)} option OCC symbols")
    return occ_symbols


def fetch_options_contracts(
    expiration_gte: date | None = None,
    expiration_lte: date | None = None,
    contract_type: str | None = None,
    limit_per_symbol: int = 100,
) -> list[dict]:
    """
    Fetch full options contract details (not just OCC symbols).

    Returns list of dicts with: occ_symbol, symbol, strike_price,
    expiration_date, contract_type, etc.

    Example:
        >>> contracts = fetch_options_contracts(limit_per_symbol=5)
        >>> contracts[0]
        {'occ_symbol': 'O:AAPL260102C00110000', 'symbol': 'AAPL', ...}
    """
    client = get_client()

    if expiration_gte is None:
        expiration_gte = datetime.now(UTC).date()
    if expiration_lte is None:
        expiration_lte = datetime.now(UTC).date() + timedelta(days=30)

    results: list[dict] = []

    for symbol in SYMBOLS:
        try:
            contracts = client.list_options_contracts(
                underlying_ticker=symbol,
                expiration_date_gte=expiration_gte,
                expiration_date_lte=expiration_lte,
                contract_type=contract_type,
                expired=False,
                limit=limit_per_symbol,
            )

            count = 0
            for c in contracts:
                results.append(
                    {
                        "occ_symbol": c.ticker,
                        "symbol": c.underlying_ticker,
                        "strike_price": c.strike_price,
                        "expiration_date": c.expiration_date,
                        "contract_type": c.contract_type,
                        "shares_per_contract": c.shares_per_contract,
                        "exercise_style": c.exercise_style,
                    }
                )
                count += 1
                if count >= limit_per_symbol:
                    break

            print(f"  {symbol}: {count} contracts")

        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")

    print(f"\nTotal: {len(results)} contracts")
    return results


def get_expiration_dates(symbol: str, days_ahead: int = 60) -> list[date]:
    """
    Get unique expiration dates for a symbol.

    Example:
        >>> dates = get_expiration_dates("AAPL")
        >>> dates[:5]
        [datetime.date(2026, 1, 2), datetime.date(2026, 1, 3), ...]
    """
    client = get_client()

    contracts = client.list_options_contracts(
        underlying_ticker=symbol,
        expiration_date_gte=datetime.now(UTC).date(),
        expiration_date_lte=datetime.now(UTC).date() + timedelta(days=days_ahead),
        expired=False,
        limit=1000,
    )

    dates = set()
    for c in contracts:
        if c.expiration_date:
            dates.add(c.expiration_date)

    return sorted(dates)


def get_strikes(
    symbol: str,
    expiration_date: date,
    contract_type: str | None = None,
) -> list[float]:
    """
    Get available strike prices for a symbol and expiration.

    Example:
        >>> strikes = get_strikes("AAPL", date(2026, 1, 17))
        >>> strikes[:5]
        [100.0, 105.0, 110.0, 115.0, 120.0]
    """
    client = get_client()

    contracts = client.list_options_contracts(
        underlying_ticker=symbol,
        expiration_date=expiration_date,
        contract_type=contract_type,
        expired=False,
        limit=1000,
    )

    strikes = set()
    for c in contracts:
        if c.strike_price:
            strikes.add(c.strike_price)

    return sorted(strikes)


def get_options_chain(
    symbol: str,
    *,
    min_volume: int = 0,
    min_open_interest: int = 0,
    max_spread_pct: float | None = None,
) -> list[dict]:
    """
    Fetch options chain snapshot with liquidity metrics.

    Returns options with: bid, ask, spread, volume, open_interest, IV, greeks.

    Args:
        symbol: Stock symbol (e.g., "NVDA")
        min_volume: Filter to contracts with at least this volume
        min_open_interest: Filter to contracts with at least this OI
        max_spread_pct: Filter to contracts with spread <= this % of mid price

    Returns:
        List of dicts with full option snapshot data

    Example:
        >>> chain = get_options_chain("NVDA", min_volume=100, max_spread_pct=0.05)
        >>> len(chain)  # Only liquid options
        42
    """
    client = get_client()

    results = []
    count = 0

    for snap in client.list_snapshot_options_chain(symbol):
        count += 1

        # Extract data safely
        details = snap.details or {}
        day = snap.day or {}
        last_quote = snap.last_quote or {}
        greeks = snap.greeks or {}

        bid = getattr(last_quote, "bid", 0) or 0
        ask = getattr(last_quote, "ask", 0) or 0
        mid = (bid + ask) / 2 if bid and ask else 0
        spread = ask - bid if bid and ask else 0
        spread_pct = spread / mid if mid > 0 else float("inf")

        volume = getattr(day, "volume", 0) or 0
        open_interest = getattr(snap, "open_interest", 0) or 0

        # Apply filters
        if volume < min_volume:
            continue
        if open_interest < min_open_interest:
            continue
        if max_spread_pct is not None and spread_pct > max_spread_pct:
            continue

        results.append(
            {
                "occ_symbol": getattr(details, "ticker", None) or snap.ticker,
                "symbol": symbol,
                "strike": getattr(details, "strike_price", None),
                "expiration_date": getattr(details, "expiration_date", None),
                "type": getattr(details, "contract_type", None),
                "bid": bid,
                "ask": ask,
                "mid": round(mid, 2),
                "spread": round(spread, 2),
                "spread_pct": round(spread_pct * 100, 2),  # as percentage
                "volume": volume,
                "open_interest": open_interest,
                "iv": getattr(snap, "implied_volatility", None),
                "delta": getattr(greeks, "delta", None),
                "gamma": getattr(greeks, "gamma", None),
                "theta": getattr(greeks, "theta", None),
                "vega": getattr(greeks, "vega", None),
            }
        )

    print(f"Fetched {count} contracts, {len(results)} passed filters")
    return results


def get_liquid_options(
    symbol: Symbol,
    *,
    min_volume: int = 100,
    min_open_interest: int = 500,
    max_spread_pct: float = 0.10,  # 10%
) -> list[str]:
    """
    Get OCC symbols that meet liquidity criteria.

    Default filters:
    - Volume >= 100
    - Open Interest >= 500
    - Spread <= 10% of mid price

    Returns:
        List of liquid OCC symbols
    """
    chain = get_options_chain(
        symbol,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        max_spread_pct=max_spread_pct,
    )
    return [c["occ_symbol"] for c in chain if c["occ_symbol"]]
