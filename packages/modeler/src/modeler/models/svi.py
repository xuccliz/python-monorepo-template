"""
SVI smile model (raw SVI) for P(S_T > K).

Pipeline:
- snapshot -> forward F (put–call parity)
- select OTM options (puts for K<F, calls for K>=F)
- invert implied vols at strikes
- fit SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
- compute prob_above(K) = 1 - N(d2) using sigma(K)

Notes:
- Uses European BS on forwards (standard approximation for US equity options).
- Filters illiquid quotes; returns None if insufficient data.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt

import numpy as np
from domain.models import OptionSurfaceSnapshot
from scipy.optimize import minimize

from modeler.forward_estimator import estimate_forward_put_call_parity
from modeler.implied_vol import implied_vol_bisect

# ----------------------------
# Utilities
# ----------------------------


def _norm_cdf(x: float) -> float:
    # stable enough for our use; avoids scipy.stats import
    from math import erf

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _bs_prob_above(*, F: float, K: float, T: float, sigma: float) -> float:
    """Risk-neutral probability that S_T > K."""
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    vol_sqrt = sigma * sqrt(T)
    d2 = (log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    # P(S_T > K) = N(d2) in risk-neutral measure
    return _norm_cdf(d2)


# ----------------------------
# SVI definition
# ----------------------------


def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sig: float) -> np.ndarray:
    # w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sig^2))
    km = k - m
    return a + b * (rho * km + np.sqrt(km * km + sig * sig))


@dataclass(frozen=True, slots=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sig: float


@dataclass(frozen=True, slots=True)
class SVIFitResult:
    params: SVIParams
    forward: float
    n_points: int


# ----------------------------
# Data extraction: OTM IVs
# ----------------------------


def _extract_otm_iv_points(
    *,
    snapshot: OptionSurfaceSnapshot,
    F: float,
    T: float,
    discount: float,
    max_spread: float | None,
    min_mid: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Returns arrays (k, w, weights) where:
      k = ln(K/F)
      w = sigma^2 * T
    Uses OTM option per strike:
      - put if K < F
      - call if K >= F
    """
    strikes = sorted(set(snapshot.call_strikes).intersection(snapshot.put_strikes))
    if len(strikes) < 8:
        return None

    ks: list[float] = []
    ws: list[float] = []
    weights: list[float] = []

    for K in strikes:
        call = snapshot.get_call(K)
        put = snapshot.get_put(K)
        if call is None or put is None:
            continue

        if max_spread is not None and (call.spread > max_spread or put.spread > max_spread):
            continue

        # pick OTM
        if K < F:
            opt_type = "put"
            price = put.mid
            spread = put.spread
        else:
            opt_type = "call"
            price = call.mid
            spread = call.spread

        if price <= min_mid or spread < 0:
            continue

        iv = implied_vol_bisect(
            option_type=opt_type,
            price=price,
            F=F,
            K=K,
            T=T,
            discount=discount,
        )
        if iv is None:
            continue

        sigma = iv.sigma
        w = sigma * sigma * T
        if not isfinite(w) or w <= 0:
            continue

        k = log(K / F)
        ks.append(k)
        ws.append(w)

        # weight tighter spreads higher (in IV-space, spread is a proxy)
        weights.append(1.0 / max(spread, 1e-6))

    if len(ks) < 8:
        return None

    return np.array(ks), np.array(ws), np.array(weights)


# ----------------------------
# SVI fit
# ----------------------------


def fit_svi(
    *,
    k: np.ndarray,
    w: np.ndarray,
    weights: np.ndarray | None = None,
) -> SVIParams | None:
    """
    Weighted least squares fit for raw SVI with simple constraints.
    Constraints enforced via bounds + penalties:
      b > 0, sig > 0, rho in (-1,1)
      w(k) >= 0 via penalty
    """

    if weights is None:
        weights = np.ones_like(w)

    # Initial guesses (robust-ish)
    a0 = max(1e-8, float(np.min(w)) * 0.5)
    b0 = max(1e-6, float(np.std(w)) + 1e-3)
    rho0 = 0.0
    m0 = float(np.median(k))
    sig0 = max(1e-3, float(np.std(k)) + 1e-3)

    x0 = np.array([a0, b0, rho0, m0, sig0], dtype=float)

    bounds = [
        (0.0, None),  # a
        (1e-10, None),  # b
        (-0.999, 0.999),  # rho
        (None, None),  # m
        (1e-10, None),  # sig
    ]

    wgt = weights / max(float(np.mean(weights)), 1e-12)

    def objective(x: np.ndarray) -> float:
        a, b, rho, m, sig = x
        w_hat = svi_total_variance(k, a, b, rho, m, sig)
        resid = w_hat - w
        loss = float(np.sum(wgt * resid * resid))

        # Penalty: discourage negative total variance
        neg = np.minimum(w_hat, 0.0)
        loss += 1e6 * float(np.sum(neg * neg))

        # Soft regularization to avoid absurd parameters
        loss += 1e-3 * float((m * m) + (sig * sig))

        return loss

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    if not res.success:
        return None

    a, b, rho, m, sig = (float(v) for v in res.x)
    return SVIParams(a=a, b=b, rho=rho, m=m, sig=sig)


# ----------------------------
# Public model API
# ----------------------------


@dataclass(frozen=True, slots=True)
class SVIModel:
    fit: SVIFitResult
    T: float
    discount: float

    def implied_vol(self, K: float) -> float:
        F = self.fit.forward
        k = log(K / F)
        p = self.fit.params
        w = float(svi_total_variance(np.array([k]), p.a, p.b, p.rho, p.m, p.sig)[0])
        w = max(w, 1e-12)
        return sqrt(w / self.T)

    def prob_above(self, K: float) -> float:
        sigma = self.implied_vol(K)
        p = _bs_prob_above(F=self.fit.forward, K=K, T=self.T, sigma=sigma)
        return max(0.0, min(1.0, float(p)))


def build_svi_model(
    *,
    snapshot: OptionSurfaceSnapshot,
    T: float,  # years to expiry
    discount: float = 1.0,
    max_spread: float | None = None,
    trim_pct: float = 0.02,
) -> SVIModel | None:
    """
    Fit SVI for one snapshot. Returns None if insufficient data.
    """

    f_est = estimate_forward_put_call_parity(
        snapshot=snapshot,
        discount=discount,
        max_spread=max_spread,
        trim_pct=trim_pct,
    )
    if f_est is None:
        return None
    F = f_est.forward

    pts = _extract_otm_iv_points(
        snapshot=snapshot,
        F=F,
        T=T,
        discount=discount,
        max_spread=max_spread,
    )
    if pts is None:
        return None

    k, w, weights = pts
    params = fit_svi(k=k, w=w, weights=weights)
    if params is None:
        return None

    return SVIModel(
        fit=SVIFitResult(params=params, forward=F, n_points=int(len(k))),
        T=T,
        discount=discount,
    )
