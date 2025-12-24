"""
Utility functions for the trading bot orchestrator.
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from domain.models import OrderResult
from domain.paths import HEALTH_DIR
from domain.types import Symbol
from domain.utils import format_duration, get_next_market_open
from store import EventStore, OptionStore

logger = logging.getLogger(__name__)


def write_heartbeat(heartbeat_file: Path) -> None:
    """Write heartbeat file for health checks."""
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    heartbeat_file.write_text(datetime.now(UTC).isoformat())


def has_recent_option_data(option_store: OptionStore, max_age_seconds: int = 30) -> bool:
    """Check if option store has been updated recently."""
    last_write = option_store.last_write()
    if last_write is None:
        return False
    age = (datetime.now(UTC) - last_write).total_seconds()
    return age < max_age_seconds


async def wait_for_market_open() -> None:
    """Sleep until 30 seconds before market open."""
    import asyncio

    next_open = get_next_market_open()
    if next_open is None:
        logger.warning("Could not determine next market open, sleeping 5 minutes")
        await asyncio.sleep(300)
        return

    now = datetime.now(UTC)
    seconds_until_open = (next_open - now).total_seconds() - 30  # Wake up 30s early

    if seconds_until_open > 0:
        logger.info("Market closed, sleeping until %s (%s)", next_open, format_duration(seconds_until_open))
        await asyncio.sleep(seconds_until_open)


def get_unique_expiries(event_store: EventStore, option_store: OptionStore) -> list[datetime]:
    """Get sorted expiries in common between EventStore and OptionStore"""
    event_expiries = event_store.get_expiration_dates()
    option_expiries = option_store.get_expiration_dates()
    expiries = sorted(option_expiries & event_expiries)
    return expiries


def filter_expiries(expiries: list[datetime], max_days: int) -> list[datetime]:
    """Filter expiries to only those within max_days"""
    today = datetime.now(UTC).date()
    cutoff = today + timedelta(days=max_days)
    expiries = [e for e in expiries if today <= e.date() <= cutoff]
    return expiries


def log_order_results(results: list[dict], order_meta: list[OrderResult], symbol: Symbol, expiry: datetime) -> None:
    """Log batch order results in a block format."""
    if not results:
        return

    lines = [f"{symbol} {expiry.date()}:"]
    placed = 0
    errors = 0

    for i, result in enumerate(results):
        meta = order_meta[i]
        order_id = result.get("orderID", "")
        error_msg = result.get("errorMsg", "")

        info = (
            f"${meta.strike} {meta.outcome} | size={meta.size} price={meta.price:.3f} "
            f"total=${meta.total:.2f} | prob={meta.prob:.2f} conf={meta.confidence:.2f}"
        )

        if order_id:
            placed += 1
            lines.append(f"  {info} → placed (id={order_id[:16]}...)")
        elif error_msg:
            errors += 1
            lines.append(f"  {info} → {error_msg}")

    lines.append(f"  ─ {placed} placed, {errors} errors")
    logger.info("\n".join(lines))
