import os
from dataclasses import dataclass, field

from domain.secrets import load_required_secret
from domain.types import SYMBOLS, ModelName, Symbol


@dataclass
class Config:
    """Configuration for Trader service and orchestrator."""

    # Polymarket credentials
    polymarket_wallet: str
    polymarket_api_key: str

    # Orchestrator settings
    symbols: list[Symbol] = field(default_factory=lambda: list(SYMBOLS))
    model_loop_interval_ms: int = 60000
    polymarket_refresh_interval_s: int = 300
    max_days: int = 14  # max days to expiry for trading

    # Model selection
    models: list[ModelName] = field(default_factory=lambda: ["simple", "slope", "spline"])

    # Trader settings
    max_total_amount: float = 1.0
    min_edge: float = 0.1
    min_confidence_score: float = 0.4
    max_spread: float = 1.0  # max option spread for model inputs

    @classmethod
    def load(cls) -> "Config":
        # Safety: Never use real polymarket key during tests unless explicitly requested
        if os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("RUN_UNSAFE_TESTS"):
            return cls(polymarket_wallet="test_wallet_address", polymarket_api_key="test_api_key_123")

        polymarket_wallet = load_required_secret("POLYMARKET_WALLET")
        polymarket_api_key = load_required_secret("POLYMARKET_API_KEY")

        return cls(polymarket_wallet=polymarket_wallet, polymarket_api_key=polymarket_api_key)
