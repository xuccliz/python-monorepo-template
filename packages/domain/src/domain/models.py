"""Domain models for options trading and probability estimation."""

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from .types import Symbol, TickSize

# ---------------------------------------------------------------------
# Options quote/state models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class OptionQuoteEvent:
    """Real-time quote event from WebSocket."""

    occ_symbol: str  # e.g., "O:NVDA260117C00140000"
    bid: float
    ask: float
    ts: datetime


@dataclass(slots=True)
class OptionState:
    """Latest state of an option contract."""

    occ_symbol: str  # e.g., "O:NVDA260117C00140000"
    symbol: Symbol  # e.g., "NVDA"
    strike_price: float
    expiration_date: datetime  # Expiry datetime (21:00 UTC = 4PM ET market close)
    option_type: str  # "call" or "put"
    bid: float
    ask: float
    mid: float
    spread: float
    last_updated: datetime


# ---------------------------------------------------------------------
# Option surface snapshot models
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptionPoint:
    """One option quote at a single strike."""

    strike: float
    option_type: str  # "call" or "put"
    bid: float
    ask: float
    mid: float
    spread: float


@dataclass(frozen=True, slots=True)
class OptionSurfaceSnapshot:
    """Immutable snapshot of an option surface for one symbol and expiry."""

    symbol: Symbol
    expiration_date: datetime
    calls: tuple["OptionPoint", ...]
    puts: tuple["OptionPoint", ...]

    @property
    def call_strikes(self) -> list[float]:
        return [p.strike for p in self.calls]

    @property
    def put_strikes(self) -> list[float]:
        return [p.strike for p in self.puts]

    @property
    def all_strikes(self) -> list[float]:
        return sorted(set(self.call_strikes + self.put_strikes))

    def get_call(self, strike: float) -> OptionPoint | None:
        for p in self.calls:
            if p.strike == strike:
                return p
        return None

    def get_put(self, strike: float) -> OptionPoint | None:
        for p in self.puts:
            if p.strike == strike:
                return p
        return None


# ---------------------------------------------------------------------
# Probability estimation models
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrikeProbability:
    """Probability estimate for a single strike."""

    strike_price: float
    prob_above: float

    @property
    def prob_below(self) -> float | None:
        if self.prob_above is None:
            return None
        return 1.0 - self.prob_above


@dataclass(frozen=True, slots=True)
class ConfidenceDiagnostics:
    """Diagnostics for probability estimate confidence."""

    agreement: float
    liquidity: float
    monotonicity: float
    spacing: float


@dataclass
class ModelPrediction:
    """Single model's probability prediction."""

    model_name: str
    prob_above: float | None
    forward: float | None = None
    extra: dict | None = None

    @property
    def prob_below(self) -> float | None:
        if self.prob_above is None:
            return None
        return 1.0 - self.prob_above


@dataclass
class ExpiryPredictions:
    """All model predictions for one expiry."""

    expiration_date: datetime
    tte_days: int
    strike_price: float
    predictions: list[ModelPrediction]


# ---------------------------------------------------------------------
# Polymarket models
# ---------------------------------------------------------------------


class MarketMetadata(BaseModel):
    """Polymarket stock price prediction market metadata."""

    question: str | None = None
    question_id: str | None = None
    strike_price: float | None = None
    yes_token_id: str | None = None
    yes_price: float | None = None
    no_token_id: str | None = None
    no_price: float | None = None
    fee_rate_bps: int | None = None
    tick_size: TickSize | None = None
    neg_risk: bool | None = None


class EventMetadata(BaseModel):
    """Polymarket stock price prediction event metadata."""

    symbol: Symbol | None = None
    question: str | None = None
    question_id: str | None = None
    end_date: str | None = None
    markets: list[MarketMetadata] | None = None
