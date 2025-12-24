"""Domain models for options trading and probability estimation."""

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

from .types import Outcome, Side, Symbol, TickSize

# ---------------------------------------------------------------------
# Options quote/state models
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedOccSymbol:
    """Parsed OCC option symbol components."""

    symbol: Symbol  # e.g., "NVDA"
    expiration_date: datetime  # Expiry datetime (21:00 UTC = 4PM ET market close)
    option_type: str  # "call" or "put"
    strike_price: float


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

    strike_price: float
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
    calls: tuple[OptionPoint, ...]
    puts: tuple[OptionPoint, ...]

    @property
    def call_strikes(self) -> list[float]:
        return [p.strike_price for p in self.calls]

    @property
    def put_strikes(self) -> list[float]:
        return [p.strike_price for p in self.puts]

    @property
    def all_strikes(self) -> list[float]:
        return sorted(set(self.call_strikes + self.put_strikes))

    def get_call(self, strike_price: float) -> OptionPoint | None:
        for p in self.calls:
            if p.strike_price == strike_price:
                return p
        return None

    def get_put(self, strike_price: float) -> OptionPoint | None:
        for p in self.puts:
            if p.strike_price == strike_price:
                return p
        return None


# ---------------------------------------------------------------------
# Probability estimation models
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfidenceDiagnostics:
    """Diagnostics for probability estimate confidence."""

    agreement: float
    liquidity: float
    monotonicity: float
    spacing: float


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Model evaluation result for a strike price."""

    prob_above: float
    confidence_score: float
    diagnostics: ConfidenceDiagnostics

    @property
    def prob_below(self) -> float:
        return 1.0 - self.prob_above


@dataclass
class ModelPrediction:
    """Single model's probability prediction."""

    model_name: str
    prob_above: float | None
    forward: float | None = None
    extra: dict | None = None
    error: str | None = None

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
    confidence_score: float | None = None
    polymarket_bid: float | None = None
    polymarket_ask: float | None = None


# ---------------------------------------------------------------------
# Polymarket models
# ---------------------------------------------------------------------


class MarketMetadata(BaseModel):
    """Polymarket stock price prediction market metadata."""

    question: str
    question_id: str | None = None
    strike_price: float
    yes_token_id: str | None = None
    yes_price: float | None = None
    no_token_id: str | None = None
    no_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    fee_rate_bps: int | None = None
    tick_size: TickSize | None = None
    neg_risk: bool | None = None


class EventMetadata(BaseModel):
    """Polymarket stock price prediction event metadata."""

    symbol: Symbol | None = None
    question: str | None = None
    question_id: str | None = None
    end_date: datetime | None = None
    markets: list[MarketMetadata] | None = None


# ---------------------------------------------------------------------
# Trade models
# ---------------------------------------------------------------------


class TradeDecision(BaseModel):
    """Internal trade decision based on confidence score."""

    should_trade: bool = Field(..., description="Whether a trade should be executed")
    side: Side | None = Field(None, description="Order side: BUY")
    outcome: Outcome | None = Field(None, description="Market outcome: YES or NO")
    size: int | None = Field(None, gt=0, description="Total number of shares")
    price: float | None = Field(None, ge=0, le=1, description="Limit order price")
    total_amount: float | None = Field(None, description="Total amount in dollars")
    reason: str = Field(..., description="Reason for the trade decision")

    @classmethod
    def execute_trade(
        cls,
        side: Side,
        outcome: Outcome,
        size: int,
        price: float,
        total_amount: float,
        reason: str,
    ) -> "TradeDecision":
        """Create a TradeDecision to EXECUTE a trade."""
        return cls(
            should_trade=True,
            side=side,
            outcome=outcome,
            size=size,
            price=price,
            total_amount=total_amount,
            reason=reason,
        )

    @classmethod
    def skip_trade(cls, reason: str) -> "TradeDecision":
        """Create a TradeDecision to SKIP executing a trade."""
        return cls(
            should_trade=False,
            side=None,
            outcome=None,
            size=None,
            price=None,
            total_amount=None,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Metadata for logging order results."""

    strike: float
    outcome: str
    size: int
    price: float
    total: float
    prob: float
    confidence: float
