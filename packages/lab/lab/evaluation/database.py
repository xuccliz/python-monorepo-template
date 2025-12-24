"""
SQLite database for caching historical options and stock data.

Stores data locally to avoid repeated API calls.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Database location
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "historical.db"


def _ensure_data_dir() -> None:
    """Ensure data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Get a database connection with row factory."""
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database() -> None:
    """Initialize database schema."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Stock daily OHLC data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_daily (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Option daily OHLC data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS option_daily (
                occ_symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (occ_symbol, date)
            )
        """)

        # Option contract metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS option_contracts (
                occ_symbol TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                strike REAL NOT NULL,
                contract_type TEXT NOT NULL,
                expiration_date TEXT NOT NULL
            )
        """)

        # Create indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_daily_symbol
            ON stock_daily(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_option_daily_occ
            ON option_daily(occ_symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_option_contracts_symbol
            ON option_contracts(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_option_contracts_expiration
            ON option_contracts(expiration_date)
        """)

        conn.commit()


# =============================================================================
# Stock Data Cache
# =============================================================================


def get_cached_stock_close(symbol: str, target_date: date) -> float | None:
    """
    Get cached stock closing price.

    Returns None if not in cache.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT close FROM stock_daily WHERE symbol = ? AND date = ?",
            (symbol, target_date.isoformat()),
        )
        row = cursor.fetchone()
        if row and row["close"] is not None:
            return float(row["close"])
    return None


def cache_stock_daily(
    symbol: str,
    data: list[dict],
) -> int:
    """
    Cache stock daily data.

    Args:
        symbol: Stock ticker
        data: List of dicts with date, open, high, low, close, volume

    Returns:
        Number of rows inserted
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        count = 0
        for row in data:
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO stock_daily
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        row["date"],
                        row.get("open"),
                        row.get("high"),
                        row.get("low"),
                        row.get("close"),
                        row.get("volume"),
                    ),
                )
                count += 1
            except Exception as e:
                # Bug #10 fix: Log database errors instead of silent swallow
                logger.warning(f"Failed to cache stock daily for {symbol} on {row.get('date')}: {e}")
        conn.commit()
        return count


def get_stock_date_range(symbol: str) -> tuple[date | None, date | None]:
    """Get the date range of cached stock data for a symbol."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MIN(date) as min_date, MAX(date) as max_date FROM stock_daily WHERE symbol = ?",
            (symbol,),
        )
        row = cursor.fetchone()
        if row and row["min_date"] and row["max_date"]:
            return (
                datetime.strptime(row["min_date"], "%Y-%m-%d").date(),
                datetime.strptime(row["max_date"], "%Y-%m-%d").date(),
            )
    return None, None


# =============================================================================
# Option Data Cache
# =============================================================================


def get_cached_option_agg(occ_symbol: str, target_date: date) -> dict | None:
    """
    Get cached option OHLC data.

    Returns dict with open, high, low, close, volume or None if not in cache.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT open, high, low, close, volume FROM option_daily WHERE occ_symbol = ? AND date = ?",
            (occ_symbol, target_date.isoformat()),
        )
        row = cursor.fetchone()
        if row:
            return {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
    return None


def cache_option_daily(
    occ_symbol: str,
    data: list[dict],
) -> int:
    """
    Cache option daily data.

    Args:
        occ_symbol: OCC option symbol
        data: List of dicts with date, open, high, low, close, volume

    Returns:
        Number of rows inserted
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        count = 0
        for row in data:
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO option_daily
                    (occ_symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        occ_symbol,
                        row["date"],
                        row.get("open"),
                        row.get("high"),
                        row.get("low"),
                        row.get("close"),
                        row.get("volume"),
                    ),
                )
                count += 1
            except Exception as e:
                # Bug #10 fix: Log database errors instead of silent swallow
                logger.warning(f"Failed to cache option daily for {occ_symbol} on {row.get('date')}: {e}")
        conn.commit()
        return count


# =============================================================================
# Option Contract Cache
# =============================================================================


def get_cached_contracts(
    symbol: str,
    expiration_date: date | None = None,
) -> list[dict]:
    """
    Get cached option contracts.

    Returns list of dicts with occ_symbol, strike, type, expiration_date.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        if expiration_date:
            cursor.execute(
                """
                SELECT occ_symbol, strike, contract_type, expiration_date
                FROM option_contracts
                WHERE symbol = ? AND expiration_date = ?
                """,
                (symbol, expiration_date.isoformat()),
            )
        else:
            cursor.execute(
                """
                SELECT occ_symbol, strike, contract_type, expiration_date
                FROM option_contracts
                WHERE symbol = ?
                """,
                (symbol,),
            )

        return [
            {
                "occ_symbol": row["occ_symbol"],
                "strike": row["strike"],
                "type": row["contract_type"],
                "expiration_date": row["expiration_date"],
            }
            for row in cursor.fetchall()
        ]


def cache_option_contracts(contracts: list[dict]) -> int:
    """
    Cache option contract metadata.

    Args:
        contracts: List of dicts with occ_symbol, symbol, strike, type, expiration_date

    Returns:
        Number of rows inserted
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        count = 0
        for c in contracts:
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO option_contracts
                    (occ_symbol, symbol, strike, contract_type, expiration_date)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        c["occ_symbol"],
                        c["symbol"],
                        c["strike"],
                        c["type"],
                        c["expiration_date"],
                    ),
                )
                count += 1
            except Exception as e:
                # Bug #10 fix: Log database errors instead of silent swallow
                logger.warning(f"Failed to cache option contract {c.get('occ_symbol')}: {e}")
        conn.commit()
        return count


def get_cached_expirations(
    symbol: str,
    from_date: date | None = None,
    to_date: date | None = None,
) -> list[date]:
    """
    Get cached expiration dates for a symbol.

    Returns sorted list of expiration dates.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        query = "SELECT DISTINCT expiration_date FROM option_contracts WHERE symbol = ?"
        params: list = [symbol]

        if from_date:
            query += " AND expiration_date >= ?"
            params.append(from_date.isoformat())
        if to_date:
            query += " AND expiration_date <= ?"
            params.append(to_date.isoformat())

        cursor.execute(query, params)

        return sorted(datetime.strptime(row["expiration_date"], "%Y-%m-%d").date() for row in cursor.fetchall())


def get_expirations_with_daily_data(
    symbol: str,
    data_date: date,
    from_date: date | None = None,
    to_date: date | None = None,
) -> list[date]:
    """
    Get expiration dates that have daily data for a specific prediction date.

    Only returns expirations where contracts have OHLC data on data_date.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT oc.expiration_date
            FROM option_contracts oc
            JOIN option_daily od ON oc.occ_symbol = od.occ_symbol
            WHERE oc.symbol = ?
            AND od.date = ?
        """
        params: list = [symbol, data_date.isoformat()]

        if from_date:
            query += " AND oc.expiration_date >= ?"
            params.append(from_date.isoformat())
        if to_date:
            query += " AND oc.expiration_date <= ?"
            params.append(to_date.isoformat())

        cursor.execute(query, params)

        return sorted(datetime.strptime(row["expiration_date"], "%Y-%m-%d").date() for row in cursor.fetchall())


def get_database_stats() -> dict:
    """Get statistics about cached data."""
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM stock_daily")
        stock_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM option_daily")
        option_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM option_contracts")
        contract_count = cursor.fetchone()["count"]

        cursor.execute("SELECT DISTINCT symbol FROM stock_daily")
        symbols = [row["symbol"] for row in cursor.fetchall()]

        return {
            "stock_daily_rows": stock_count,
            "option_daily_rows": option_count,
            "option_contracts": contract_count,
            "symbols": symbols,
            "db_path": str(DB_PATH),
        }


def get_option_daily_date_range(symbol: str) -> tuple[date | None, date | None]:
    """Get the date range of cached option daily data for a symbol."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # Extract symbol from OCC format (e.g., O:AAPL251205C00100000 -> AAPL)
        cursor.execute(
            """
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM option_daily
            WHERE occ_symbol LIKE ?
            """,
            (f"O:{symbol}%",),
        )
        row = cursor.fetchone()
        if row and row["min_date"] and row["max_date"]:
            return (
                datetime.strptime(row["min_date"], "%Y-%m-%d").date(),
                datetime.strptime(row["max_date"], "%Y-%m-%d").date(),
            )
    return None, None


def get_option_daily_dates(symbol: str) -> list[date]:
    """Get all dates that have option daily data for a symbol."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT date
            FROM option_daily
            WHERE occ_symbol LIKE ?
            ORDER BY date
            """,
            (f"O:{symbol}%",),
        )
        return [datetime.strptime(row["date"], "%Y-%m-%d").date() for row in cursor.fetchall()]
