"""
Implied volatility inversion (European) using Black–Scholes on forwards.

Call price:
    C = D * ( F * N(d1) - K * N(d2) )
Put price:
    P = D * ( K * N(-d2) - F * N(-d1) )

where:
    d1 = (ln(F/K) + 0.5*sigma^2*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

This module inverts sigma from (price, F, K, T, D) via bisection.

Note: US equity options are American; this is a European approximation.
"""

from dataclasses import dataclass
from math import erf, isfinite, log, sqrt


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price_forward(
    *,
    option_type: str,  # "call" or "put"
    F: float,  # forward price
    K: float,  # strike
    T: float,  # time to expiry (years)
    sigma: float,
    discount: float = 1.0,
) -> float:
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0 or discount <= 0:
        return float("nan")

    vol_sqrt = sigma * sqrt(T)
    if vol_sqrt <= 0:
        return float("nan")

    d1 = (log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt

    if option_type == "call":
        return discount * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    if option_type == "put":
        return discount * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))

    raise ValueError("option_type must be 'call' or 'put'")


def _no_arb_bounds_forward(
    *,
    option_type: str,
    F: float,
    K: float,
    discount: float,
) -> tuple[float, float]:
    # Lower bound: discounted intrinsic (European)
    if option_type == "call":
        lb = discount * max(F - K, 0.0)
        ub = discount * F  # call <= D*F
    else:
        lb = discount * max(K - F, 0.0)
        ub = discount * K  # put <= D*K
    return lb, ub


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    sigma: float
    iterations: int
    price_fit: float


def implied_vol_bisect(
    *,
    option_type: str,  # "call" or "put"
    price: float,  # option mid price
    F: float,  # forward price
    K: float,  # strike
    T: float,  # time to expiry (years)
    discount: float = 1.0,
    vol_low: float = 1e-6,
    vol_high: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> ImpliedVolResult | None:
    """
    Invert Black–Scholes implied vol using bisection.

    Returns None if inputs are invalid or price is outside no-arbitrage bounds.
    """

    if not (isfinite(price) and isfinite(F) and isfinite(K) and isfinite(T) and isfinite(discount)):
        return None
    if price <= 0 or F <= 0 or K <= 0 or T <= 0 or discount <= 0:
        return None

    lb, ub = _no_arb_bounds_forward(option_type=option_type, F=F, K=K, discount=discount)

    # Allow small numerical slack
    if price < lb - 1e-10 or price > ub + 1e-10:
        return None

    # Handle near-intrinsic: vol ~ 0
    if abs(price - lb) <= 1e-12:
        return ImpliedVolResult(sigma=vol_low, iterations=0, price_fit=lb)

    lo = vol_low
    hi = vol_high

    # Ensure bracket: price(lo) <= target <= price(hi)
    p_lo = bs_price_forward(option_type=option_type, F=F, K=K, T=T, sigma=lo, discount=discount)
    p_hi = bs_price_forward(option_type=option_type, F=F, K=K, T=T, sigma=hi, discount=discount)

    if not (isfinite(p_lo) and isfinite(p_hi)):
        return None

    # If not bracketed, expand hi a bit (bounded)
    if p_hi < price:
        for _ in range(10):
            hi *= 1.5
            if hi > 10.0:
                break
            p_hi = bs_price_forward(option_type=option_type, F=F, K=K, T=T, sigma=hi, discount=discount)
            if p_hi >= price:
                break

    if p_lo > price or p_hi < price:
        return None

    it = 0
    mid = 0.0
    p_mid = 0.0

    while it < max_iter:
        it += 1
        mid = 0.5 * (lo + hi)
        p_mid = bs_price_forward(option_type=option_type, F=F, K=K, T=T, sigma=mid, discount=discount)

        if not isfinite(p_mid):
            return None

        diff = p_mid - price
        if abs(diff) <= tol:
            return ImpliedVolResult(sigma=mid, iterations=it, price_fit=p_mid)

        if diff < 0:
            lo = mid
        else:
            hi = mid

    return ImpliedVolResult(sigma=mid, iterations=it, price_fit=p_mid)
