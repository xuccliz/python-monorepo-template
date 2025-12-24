from datetime import UTC, datetime, time
from typing import Annotated, Literal, TypeGuard, get_args

from pydantic import BeforeValidator

Outcome = Literal["YES", "NO"]
Side = Literal["BUY", "SELL"]
Symbol = Literal["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "PLTR", "OPEN"]
SYMBOLS = set(get_args(Symbol))
TickSize = Annotated[
    Literal["0.1", "0.01", "0.001", "0.0001"], BeforeValidator(lambda v: str(v) if v is not None else v)
]

# Options expire at 4:00 PM ET = 21:00 UTC
EXPIRY_TIME_UTC = time(21, 0, 0, tzinfo=UTC)


def is_symbol(value: str) -> TypeGuard[Symbol]:
    """Check if a string is a valid Symbol."""
    return value in SYMBOLS


def make_expiry_datetime(date_str: str) -> datetime:
    """Create expiry datetime from YYYY-MM-DD string with 21:00 UTC (4PM ET market close)."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    return datetime.combine(d, EXPIRY_TIME_UTC)
