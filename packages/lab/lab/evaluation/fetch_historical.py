"""
Fetch historical data and populate the local cache.

Fetches 15 years of stock and options data from Polygon API.

Usage:
    python -m evaluation.fetch_historical --symbols NVDA AAPL --years 15
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any, get_args

from domain.secrets import load_required_secret
from domain.types import Symbol
from dotenv import load_dotenv
from massive import RESTClient

from .database import (
    cache_option_contracts,
    cache_option_daily,
    cache_stock_daily,
    get_database_stats,
    get_stock_date_range,
    init_database,
)

load_dotenv()

# Rate limit delay (seconds between API calls)
RATE_LIMIT_DELAY = 0.25  # 4 calls/second to stay under limits


def _get_client() -> RESTClient:
    """Get an authenticated Massive REST client."""
    return RESTClient(api_key=load_required_secret("MASSIVE_API_KEY"))


def _parse_agg_data(aggs: Any) -> list[dict[str, Any]]:
    """Parse aggregate data from Polygon API response."""
    if not isinstance(aggs, list):
        return []
    data = []
    for agg in aggs:
        ts = getattr(agg, "timestamp", None)
        if not ts:
            continue
        dt = datetime.fromtimestamp(ts / 1000, tz=UTC).date()
        data.append(
            {
                "date": dt.isoformat(),
                "open": getattr(agg, "open", None),
                "high": getattr(agg, "high", None),
                "low": getattr(agg, "low", None),
                "close": getattr(agg, "close", None),
                "volume": getattr(agg, "volume", None),
            }
        )
    return data


def fetch_and_cache_stock_daily(
    symbol: str,
    from_date: date,
    to_date: date,
    verbose: bool = True,
) -> int:
    """
    Fetch stock daily data and cache it.

    Args:
        symbol: Stock ticker
        from_date: Start date
        to_date: End date
        verbose: Print progress

    Returns:
        Number of rows cached
    """
    client = _get_client()

    if verbose:
        print(f"  Fetching {symbol} stock data {from_date} to {to_date}...")

    try:
        aggs = client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=from_date,
            to=to_date,
            limit=50000,
        )

        if not aggs:
            if verbose:
                print(f"    No data returned for {symbol}")
            return 0

        data = _parse_agg_data(aggs)
        count = cache_stock_daily(symbol, data)
        if verbose:
            print(f"    Cached {count} rows for {symbol}")
        return count

    except Exception as e:
        if verbose:
            print(f"    Error fetching {symbol}: {e}")
        return 0


def fetch_and_cache_option_contracts(
    symbol: str,
    from_date: date,
    to_date: date,
    verbose: bool = True,
) -> int:
    """
    Fetch option contracts and cache metadata.

    Args:
        symbol: Stock ticker
        from_date: Min expiration date
        to_date: Max expiration date
        verbose: Print progress

    Returns:
        Number of contracts cached
    """
    client = _get_client()

    if verbose:
        print(f"  Fetching {symbol} option contracts...")

    try:
        contracts = client.list_options_contracts(
            underlying_ticker=symbol,
            expiration_date_gte=from_date,
            expiration_date_lte=to_date,
            expired=True,
            limit=1000,
        )

        data = []
        for c in contracts:
            ticker = getattr(c, "ticker", None)
            strike = getattr(c, "strike_price", None)
            ctype = getattr(c, "contract_type", None)
            exp = getattr(c, "expiration_date", None)

            if not ticker or not strike or not ctype or not exp:
                continue

            exp_str = exp.isoformat() if hasattr(exp, "isoformat") else str(exp)
            data.append(
                {
                    "occ_symbol": ticker,
                    "symbol": symbol,
                    "strike": strike,
                    "type": ctype,
                    "expiration_date": exp_str,
                }
            )

        count = cache_option_contracts(data)
        if verbose:
            print(f"    Cached {count} contracts for {symbol}")
        return count

    except Exception as e:
        if verbose:
            print(f"    Error fetching contracts for {symbol}: {e}")
        return 0


def fetch_and_cache_option_daily(
    occ_symbol: str,
    from_date: date,
    to_date: date,
) -> int:
    """
    Fetch option daily OHLC data and cache it.

    Returns:
        Number of rows cached
    """
    client = _get_client()

    try:
        aggs = client.get_aggs(
            ticker=occ_symbol,
            multiplier=1,
            timespan="day",
            from_=from_date,
            to=to_date,
            limit=50000,
        )

        if not aggs:
            return 0

        data = _parse_agg_data(aggs)
        return cache_option_daily(occ_symbol, data)

    except Exception:
        return 0


def fetch_all_historical(
    symbols: list[str] | None = None,
    years: int = 15,
    fetch_options: bool = True,
    monthly_only: bool = False,
    verbose: bool = True,
) -> None:
    """
    Fetch all historical data for given symbols.

    Args:
        symbols: List of symbols (default: all configured symbols)
        years: Number of years of history to fetch
        fetch_options: Whether to fetch option data (slower)
        monthly_only: Only fetch monthly expirations (faster)
        verbose: Print progress
    """
    if symbols is None:
        symbols = list(get_args(Symbol))

    init_database()

    today = datetime.now(UTC).date()
    from_date = today - timedelta(days=years * 365)

    print(f"Fetching {years} years of data ({from_date} to {today})")
    print(f"Symbols: {', '.join(symbols)}")
    print()

    # Fetch stock data
    print("=" * 60)
    print("FETCHING STOCK DATA")
    print("=" * 60)

    for symbol in symbols:
        # Check existing data range
        min_date, max_date = get_stock_date_range(symbol)
        if min_date and max_date:
            print(f"  {symbol}: Existing data {min_date} to {max_date}")
            # Only fetch missing ranges
            if min_date > from_date:
                fetch_and_cache_stock_daily(symbol, from_date, min_date - timedelta(days=1), verbose)
            if max_date < today:
                fetch_and_cache_stock_daily(symbol, max_date + timedelta(days=1), today, verbose)
        else:
            fetch_and_cache_stock_daily(symbol, from_date, today, verbose)

        time.sleep(RATE_LIMIT_DELAY)

    if not fetch_options:
        print("\nSkipping options data (use --options to include)")
        _print_stats()
        return

    # Fetch option contracts
    print()
    print("=" * 60)
    print("FETCHING OPTION CONTRACTS")
    print("=" * 60)

    for symbol in symbols:
        fetch_and_cache_option_contracts(symbol, from_date, today, verbose)
        time.sleep(RATE_LIMIT_DELAY)

    # Fetch option daily data (this is the slow part)
    print()
    print("=" * 60)
    print("FETCHING OPTION DAILY DATA")
    print("=" * 60)
    print("(This may take a while due to API rate limits)")

    from .database import get_cached_contracts

    total_contracts = 0
    total_rows = 0

    for symbol in symbols:
        contracts = get_cached_contracts(symbol)
        # Filter contracts within the date range
        valid_contracts = [
            c for c in contracts if from_date <= datetime.strptime(c["expiration_date"], "%Y-%m-%d").date() <= today
        ]
        print(f"  {symbol}: {len(valid_contracts)} contracts to fetch")

        for i, c in enumerate(valid_contracts):
            occ_symbol = c["occ_symbol"]
            exp_str = c["expiration_date"]
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()

            # Fetch data from 30 days before expiry to expiry
            from_fetch = max(from_date, exp_date - timedelta(days=30))
            to_fetch = min(today, exp_date)
            rows = fetch_and_cache_option_daily(occ_symbol, from_fetch, to_fetch)
            total_rows += rows
            total_contracts += 1

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(valid_contracts)} contracts ({total_rows} rows)")

            time.sleep(RATE_LIMIT_DELAY)

    print(f"\nTotal: {total_contracts} contracts, {total_rows} daily rows")

    _print_stats()


def _print_stats() -> None:
    """Print database statistics."""
    print()
    print("=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    stats = get_database_stats()
    print(f"  Stock daily rows: {stats['stock_daily_rows']:,}")
    print(f"  Option daily rows: {stats['option_daily_rows']:,}")
    print(f"  Option contracts: {stats['option_contracts']:,}")
    print(f"  Symbols: {', '.join(stats['symbols'])}")
    print(f"  Database: {stats['db_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical data into local cache")
    parser.add_argument("--symbols", nargs="+", help="Symbols to fetch (default: all)")
    parser.add_argument("--years", type=int, default=15, help="Years of history (default: 15)")
    parser.add_argument("--options", action="store_true", help="Include options data (slower)")
    parser.add_argument("--monthly-only", action="store_true", help="Only fetch monthly expirations (faster)")
    parser.add_argument("--stats", action="store_true", help="Just print database stats")

    args = parser.parse_args()

    if args.stats:
        init_database()
        _print_stats()
        return

    fetch_all_historical(
        symbols=args.symbols,
        years=args.years,
        fetch_options=args.options,
        monthly_only=args.monthly_only,
        verbose=True,
    )


if __name__ == "__main__":
    main()
