from typing import Annotated, Literal, TypeGuard, get_args

from pydantic import BeforeValidator, Field

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Outcome = Literal["YES", "NO"]
Side = Literal["BUY", "SELL"]

TickSize = Annotated[
    Literal["0.1", "0.01", "0.001", "0.0001"], BeforeValidator(lambda v: str(v) if v is not None else v)
]
Symbol = Literal["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "PLTR", "OPEN"]
SYMBOLS = set(get_args(Symbol))

ModelName = Literal["simple", "slope", "svi", "spline"]


def is_symbol(value: str) -> TypeGuard[Symbol]:
    """Check if a string is a valid Symbol."""
    return value in SYMBOLS
