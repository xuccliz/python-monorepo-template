"""Domain utility functions."""

from datetime import UTC, datetime

import exchange_calendars as xcals

SECONDS_PER_YEAR = 365.25 * 24 * 3600

# NYSE calendar covers regular trading hours for US equities and options
_NYSE_CALENDAR = xcals.get_calendar("XNYS")


def make_expiry_datetime(date_str: str) -> datetime:
    """Create expiry datetime from YYYY-MM-DD string with 21:00 UTC (4PM ET market close)."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    return datetime(d.year, d.month, d.day, 21, 0, 0, tzinfo=UTC)


def time_to_expiry_years(expiration_date: datetime, now: datetime | None = None) -> float:
    """Calculate time to expiry in years."""
    if now is None:
        now = datetime.now(UTC)
    return (expiration_date - now).total_seconds() / SECONDS_PER_YEAR


def is_market_open(dt: datetime | None = None) -> bool:
    """
    Check if the US stock market is currently open.

    Args:
        dt: Datetime to check (default: current UTC time)

    Returns:
        True if the market is open for regular trading, False otherwise
    """
    if dt is None:
        dt = datetime.now(UTC)

    try:
        return _NYSE_CALENDAR.is_open_on_minute(dt)
    except Exception:
        return False


def get_next_market_open(dt: datetime | None = None) -> datetime | None:
    """
    Get the next market open time.

    Args:
        dt: Reference datetime (default: current UTC time)

    Returns:
        Next market open as datetime, or None if unavailable
    """
    if dt is None:
        dt = datetime.now(UTC)

    try:
        next_open = _NYSE_CALENDAR.next_open(dt)
        return next_open.to_pydatetime()
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration (e.g., '2h 15m 30s')."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"
