"""
Simple strike probability estimator.

Estimates P(S_T > K) using call/put mid prices
at the same strike.
"""

from dataclasses import dataclass

from domain.models import OptionSurfaceSnapshot
from domain.types import Probability


@dataclass(frozen=True, slots=True)
class SimpleModel:
    """Simple call/put ratio probability model."""

    snapshot: OptionSurfaceSnapshot
    max_spread: float | None = None

    def prob_above(self, K: float) -> Probability | None:
        """Return P(S_T > K) for strike K."""
        return estimate_probability_simple(
            snapshot=self.snapshot,
            strike_price=K,
            max_spread=self.max_spread,
        )


def build_simple_model(
    snapshot: OptionSurfaceSnapshot,
    *,
    max_spread: float | None = None,
) -> SimpleModel:
    """
    Build a simple probability model.

    Always returns a model (never None) since the model
    can handle missing strikes at query time.
    """
    return SimpleModel(snapshot=snapshot, max_spread=max_spread)


def estimate_probability_simple(
    *,
    snapshot: OptionSurfaceSnapshot,
    strike_price: float,
    max_spread: float | None = None,
) -> Probability | None:
    """
    Estimate probability that the stock price finishes above a strike.

    Uses:
        P(S_T > K) ≈ C(K) / (C(K) + P(K))

    Returns None if required quotes are missing or unreliable.
    """

    call = snapshot.get_call(strike_price)
    put = snapshot.get_put(strike_price)

    if call is None or put is None:
        return None

    if max_spread is not None:
        if call.spread > max_spread or put.spread > max_spread:
            return None

    c = call.mid
    p = put.mid

    if c <= 0 or p <= 0:
        return None

    denom = c + p
    if denom <= 0:
        return None

    prob_above = c / denom

    # Clamp for numerical safety
    return max(0.0, min(1.0, prob_above))
