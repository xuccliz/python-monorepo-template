"""
Forward price estimator from put–call parity.

Estimates the market-implied forward F for a given expiry using:
    C(K) - P(K) = D * (F - K)  =>  F(K) = K + (C(K) - P(K)) / D

Uses multiple strikes and robust aggregation (weighted mean with trimming).
"""

from dataclasses import dataclass
from math import isfinite

from domain.models import OptionSurfaceSnapshot


@dataclass(frozen=True, slots=True)
class ForwardEstimate:
    forward: float
    n_used: int
    median: float
    min_f: float
    max_f: float


def estimate_forward_put_call_parity(
    *,
    snapshot: OptionSurfaceSnapshot,
    discount: float = 1.0,  # D = exp(-rT)
    max_spread: float | None = None,  # absolute spread filter
    trim_pct: float = 0.02,  # keep strikes within ±2% of median forward
    min_mid: float = 1e-6,
) -> ForwardEstimate | None:
    """
    Robust forward estimator for one underlying + expiry.

    Parameters
    ----------
    snapshot:
        OptionSurfaceSnapshot (same underlying + expiry)
    discount:
        Discount factor D. Use 1.0 initially if you ignore rates.
    max_spread:
        Drop call/put quotes with spread above this absolute threshold.
    trim_pct:
        After computing forwards across strikes, trim outliers relative to median.
    min_mid:
        Drop quotes with mid <= min_mid.

    Returns
    -------
    ForwardEstimate or None if insufficient data.
    """

    if discount <= 0 or not isfinite(discount):
        raise ValueError("discount must be positive and finite")

    # Build common strike set where both call and put exist
    strikes = sorted(set(snapshot.call_strikes).intersection(snapshot.put_strikes))
    if not strikes:
        return None

    candidates: list[tuple[float, float]] = []  # (F_i, weight)

    for k in strikes:
        call = snapshot.get_call(k)
        put = snapshot.get_put(k)
        if call is None or put is None:
            continue

        # Basic quote sanity
        if call.mid <= min_mid or put.mid <= min_mid:
            continue
        if call.bid < 0 or call.ask < 0 or call.bid > call.ask:
            continue
        if put.bid < 0 or put.ask < 0 or put.bid > put.ask:
            continue

        if max_spread is not None:
            if call.spread > max_spread or put.spread > max_spread:
                continue

        # Put–call parity forward per strike
        f_i = k + (call.mid - put.mid) / discount
        if not isfinite(f_i) or f_i <= 0:
            continue

        # Weight by liquidity (tighter spreads = higher weight)
        w = 1.0 / max(call.spread + put.spread, 1e-9)
        candidates.append((f_i, w))

    if len(candidates) < 3:
        return None

    # Median forward (robust center)
    fs_sorted = sorted(f for f, _ in candidates)
    mid_idx = len(fs_sorted) // 2
    median = fs_sorted[mid_idx] if len(fs_sorted) % 2 == 1 else 0.5 * (fs_sorted[mid_idx - 1] + fs_sorted[mid_idx])

    # Trim outliers around median
    lo = median * (1.0 - trim_pct)
    hi = median * (1.0 + trim_pct)
    trimmed = [(f, w) for f, w in candidates if lo <= f <= hi]

    if len(trimmed) < 3:
        # If trimming was too aggressive, fall back to untrimmed
        trimmed = candidates

    # Weighted mean on trimmed set
    w_sum = sum(w for _, w in trimmed)
    if w_sum <= 0:
        return None

    forward = sum(f * w for f, w in trimmed) / w_sum
    f_vals = [f for f, _ in trimmed]

    return ForwardEstimate(
        forward=forward,
        n_used=len(trimmed),
        median=median,
        min_f=min(f_vals),
        max_f=max(f_vals),
    )
