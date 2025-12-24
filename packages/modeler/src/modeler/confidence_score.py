"""
Confidence score for strike-level probability estimates.

The score measures DATA QUALITY and ESTIMATOR STABILITY,
not correctness of the market-implied probability.

Output:
    confidence in [0, 1]

High confidence means:
- liquid quotes
- stable local surface
- agreement between estimators
"""

from math import exp

from domain.models import ConfidenceDiagnostics, OptionSurfaceSnapshot, StrikeProbability


def compute_confidence(
    *,
    snapshot: OptionSurfaceSnapshot,
    strike: float,
    prob_simple: StrikeProbability | None,
    prob_slope: StrikeProbability | None,
    max_relative_spread: float = 0.5,
) -> tuple[float, ConfidenceDiagnostics]:
    """
    Compute a confidence score for P(S_T > K).

    Parameters
    ----------
    snapshot:
        OptionSurfaceSnapshot
    strike:
        Target strike K
    prob_simple:
        Result from simple estimator (call/put ratio)
    prob_slope:
        Result from slope-based estimator
    max_relative_spread:
        Relative spread above which confidence collapses

    Returns
    -------
    (confidence, diagnostics)
    """

    # ------------------------------------------------------------
    # 1. Estimator agreement (very strong signal)
    # ------------------------------------------------------------

    if prob_simple is None or prob_slope is None:
        agreement_score = 0.0
    else:
        delta = abs(prob_simple.prob_above - prob_slope.prob_above)
        # exponential decay: small differences are fine, large ones kill confidence
        agreement_score = exp(-5.0 * delta)

    # ------------------------------------------------------------
    # 2. Liquidity / spread quality
    # ------------------------------------------------------------

    call = snapshot.get_call(strike)
    put = snapshot.get_put(strike)

    if call is None or put is None:
        liquidity_score = 0.0
    else:
        # relative spread vs mid
        rel_call = call.spread / max(call.mid, 1e-6)
        rel_put = put.spread / max(put.mid, 1e-6)
        rel_spread = max(rel_call, rel_put)

        if rel_spread >= max_relative_spread:
            liquidity_score = 0.0
        else:
            liquidity_score = 1.0 - (rel_spread / max_relative_spread)

    # ------------------------------------------------------------
    # 3. Local monotonicity sanity (calls ↓ as strike ↑)
    # ------------------------------------------------------------

    calls = snapshot.calls
    strikes = [c.strike for c in calls]
    mids = [c.mid for c in calls]

    i: int | None = None
    try:
        i = min(range(len(strikes)), key=lambda j: abs(strikes[j] - strike))
    except ValueError:
        pass

    if i is None:
        monotonicity_score = 0.0
    else:
        ok = True
        if i > 0 and mids[i] > mids[i - 1]:
            ok = False
        if i < len(mids) - 1 and mids[i] < mids[i + 1]:
            ok = False

        monotonicity_score = 1.0 if ok else 0.0

    # ------------------------------------------------------------
    # 4. Strike spacing sanity (avoid extrapolation)
    # ------------------------------------------------------------

    if i is None or i <= 0 or i >= len(strikes) - 1:
        spacing_score = 0.0
    else:
        dk_left = abs(strikes[i] - strikes[i - 1])
        dk_right = abs(strikes[i + 1] - strikes[i])
        spacing = max(dk_left, dk_right)

        # heuristic: tighter spacing is better
        spacing_score = exp(-0.1 * spacing)

    # ------------------------------------------------------------
    # Final weighted confidence
    # ------------------------------------------------------------

    confidence = 0.40 * agreement_score + 0.30 * liquidity_score + 0.20 * monotonicity_score + 0.10 * spacing_score

    confidence = max(0.0, min(1.0, confidence))

    diagnostics = ConfidenceDiagnostics(
        agreement=agreement_score,
        liquidity=liquidity_score,
        monotonicity=monotonicity_score,
        spacing=spacing_score,
    )

    return confidence, diagnostics
