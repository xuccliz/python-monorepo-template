"""
Strike probability estimator using call-price slope.

Estimates:
    P(S_T > K) = - (1 / D) * dC/dK

where dC/dK is approximated using finite differences
on call mid prices.
"""

from dataclasses import dataclass

from domain.models import OptionSurfaceSnapshot
from domain.types import Probability


@dataclass(frozen=True, slots=True)
class SlopeModel:
    """Slope-based probability model using call price gradients."""

    snapshot: OptionSurfaceSnapshot
    window: int = 1
    discount: float = 1.0
    max_spread: float | None = None

    def prob_above(self, K: float) -> Probability | None:
        """Return P(S_T > K) for strike K."""
        return estimate_probability_slope(
            snapshot=self.snapshot,
            strike_price=K,
            window=self.window,
            discount=self.discount,
            max_spread=self.max_spread,
        )


def build_slope_model(
    snapshot: OptionSurfaceSnapshot,
    *,
    window: int = 1,
    discount: float = 1.0,
    max_spread: float | None = None,
) -> SlopeModel:
    """Build a slope-based probability model."""
    return SlopeModel(snapshot=snapshot, window=window, discount=discount, max_spread=max_spread)


def estimate_probability_slope(
    *,
    snapshot: OptionSurfaceSnapshot,
    strike_price: float,
    window: int = 1,
    discount: float = 1.0,
    max_spread: float | None = None,
) -> Probability | None:
    """
    Estimate probability that the stock price finishes above a strike
    using the slope of call prices with respect to strike price.

    Parameters
    ----------
    snapshot:
        OptionSurfaceSnapshot for one underlying + expiry
    strike_price:
        Target strike K
    window:
        Number of strikes on each side to use (1 = nearest neighbors)
    discount:
        Discount factor D = exp(-rT). Use 1.0 if rates are ignored.
    max_spread:
        Optional filter to exclude illiquid calls

    Returns
    -------
    Probability or None if insufficient data
    """

    calls = snapshot.calls
    if len(calls) < 2 * window + 1:
        return None

    # Extract strikes and mids
    strikes = [c.strike_price for c in calls]
    mids = [c.mid for c in calls]
    spreads = [c.spread for c in calls]

    # Find index closest to target strike
    try:
        i = min(
            range(len(strikes)),
            key=lambda j: abs(strikes[j] - strike_price),
        )
    except ValueError:
        return None

    left = i - window
    right = i + window

    if left < 0 or right >= len(strikes):
        return None

    # Optional liquidity filter
    if max_spread is not None:
        for j in range(left, right + 1):
            if spreads[j] > max_spread:
                return None

    k_left = strikes[left]
    k_right = strikes[right]
    c_left = mids[left]
    c_right = mids[right]

    if k_right == k_left:
        return None

    # Finite-difference slope
    slope = (c_right - c_left) / (k_right - k_left)

    # Probability from slope
    prob_above = -slope / discount

    # Clamp for numerical safety
    return max(0.0, min(1.0, prob_above))
