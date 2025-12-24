"""
Spline smile model (smoothing spline) for P(S_T > K).

Pipeline:
- snapshot -> forward F (put–call parity)
- select OTM options (puts for K<F, calls for K>=F)
- invert implied vols at strikes
- fit smoothing spline to total variance w(k)
- compute prob_above(K) = 1 - N(d2) using sigma(K)

Notes:
- Smoothing splines are very accurate in practice, especially with good liquidity.
- This does not strictly guarantee global no-arbitrage; we enforce positivity and
  use smoothing + filtering to keep it stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt

import numpy as np
from domain.models import OptionSurfaceSnapshot
from scipy.interpolate import UnivariateSpline

from modeler.forward_estimator import estimate_forward_put_call_parity
from modeler.implied_vol import implied_vol_bisect


def _norm_cdf(x: float) -> float:
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


@dataclass(frozen=True, slots=True)
class SplineFitResult:
    forward: float
    n_points: int


@dataclass(frozen=True, slots=True)
class SplineModel:
    fit: SplineFitResult
    spline_w: UnivariateSpline
    T: float
    discount: float
    k_min: float
    k_max: float

    def total_variance(self, K: float) -> float:
        F = self.fit.forward
        k = log(K / F)

        # mild extrapolation control: clamp to observed range
        k = min(max(k, self.k_min), self.k_max)

        result = np.asarray(self.spline_w(k))
        w: float = float(result.item()) if result.ndim == 0 else float(result[0])
        # enforce positivity
        return max(w, 1e-12)

    def implied_vol(self, K: float) -> float:
        w = self.total_variance(K)
        return sqrt(w / self.T)

    def prob_above(self, K: float) -> float:
        sigma = self.implied_vol(K)
        p = _bs_prob_above(F=self.fit.forward, K=K, T=self.T, sigma=sigma)
        return max(0.0, min(1.0, float(p)))


def _extract_otm_iv_points(
    *,
    snapshot: OptionSurfaceSnapshot,
    F: float,
    T: float,
    discount: float,
    max_spread: float | None,
    min_mid: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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
        weights.append(1.0 / max(spread, 1e-6))

    if len(ks) < 8:
        return None

    k_arr = np.array(ks)
    w_arr = np.array(ws)
    wgt = np.array(weights)

    # sort by k (required for spline stability)
    idx = np.argsort(k_arr)
    return k_arr[idx], w_arr[idx], wgt[idx]


def build_spline_model(
    *,
    snapshot: OptionSurfaceSnapshot,
    T: float,  # years to expiry
    discount: float = 1.0,
    max_spread: float | None = None,
    trim_pct: float = 0.02,
    smoothing: float | None = None,  # if None, auto based on noise
) -> SplineModel | None:
    """
    Fit a smoothing spline to total variance w(k).
    Returns None if insufficient data.
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

    # Choose smoothing if not provided: proportional to noise and point count
    # Larger smoothing => smoother curve; smaller => closer fit
    if smoothing is None:
        # Heuristic: smooth more when fewer points / more noise
        w_std = float(np.std(w))
        smoothing = max(1e-8, 0.5 * w_std * len(w))

    # UnivariateSpline uses weights as 1/sigma_y; we already use liquidity proxy
    spline = UnivariateSpline(k, w, w=weights, s=float(smoothing), k=3)

    return SplineModel(
        fit=SplineFitResult(forward=F, n_points=int(len(k))),
        spline_w=spline,
        T=T,
        discount=discount,
        k_min=float(k.min()),
        k_max=float(k.max()),
    )
