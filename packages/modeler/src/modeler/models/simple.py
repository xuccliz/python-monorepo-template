"""
Simple strike probability estimator.

Estimates P(S_T > K) using call/put mid prices
at the same strike.
"""

from dataclasses import dataclass

from domain.models import OptionSurfaceSnapshot, StrikeProbability


@dataclass(frozen=True, slots=True)
class SimpleModel:
    """Simple call/put ratio probability model."""

    snapshot: OptionSurfaceSnapshot
    max_spread: float | None = None

    def prob_above(self, K: float) -> float:
        """Return P(S_T > K) for strike K."""
        result = estimate_probability_simple(
            snapshot=self.snapshot,
            strike=K,
            max_spread=self.max_spread,
        )
        if result is None:
            return float("nan")
        return result.prob_above


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
    strike: float,
    max_spread: float | None = None,
) -> StrikeProbability | None:
    """
    Estimate probability that the stock price finishes above a strike.

    Uses:
        P(S_T > K) ≈ C(K) / (C(K) + P(K))

    Returns None if required quotes are missing or unreliable.
    """

    call = snapshot.get_call(strike)
    put = snapshot.get_put(strike)

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
    prob_above = max(0.0, min(1.0, prob_above))

    return StrikeProbability(
        strike_price=strike,
        prob_above=prob_above,
    )
