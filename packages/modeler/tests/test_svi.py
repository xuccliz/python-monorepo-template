from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from domain.models import OptionPoint, OptionSurfaceSnapshot
from modeler.models.svi import build_svi_model


def make_point(
    strike: float,
    option_type: str,
    price: float,
    spread: float = 0.1,
) -> OptionPoint:
    """Helper to create an OptionPoint."""
    return OptionPoint(
        strike_price=strike,
        option_type=option_type,
        bid=price - spread / 2,
        ask=price + spread / 2,
        mid=price,
        spread=spread,
    )


@pytest.fixture
def snapshot() -> OptionSurfaceSnapshot:
    """Create a synthetic option surface snapshot."""
    # Assume S=100.
    # Generate calls and puts around 100 with Black-Scholes-like prices
    # For simplicity, we just put in some reasonable numbers.

    # Increase number of points (min 8 required for SVI/Spline)
    strikes = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130]
    # Intrinsic value approximation
    calls = []
    puts = []

    for k in strikes:
        # Call price ~ max(0, 100-k) + time value
        # Put price ~ max(0, k-100) + time value
        time_value = 2.0 * np.exp(-0.005 * (abs(100 - k)) ** 2)  # Fake bell curve, slightly wider

        c_price = max(0, 100 - k) + time_value
        p_price = max(0, k - 100) + time_value

        calls.append(make_point(k, "call", float(c_price)))
        puts.append(make_point(k, "put", float(p_price)))

    return OptionSurfaceSnapshot(
        symbol="NVDA",
        expiration_date=datetime.now(UTC) + timedelta(days=30),
        calls=tuple(calls),
        puts=tuple(puts),
    )


def test_svi_smoke(snapshot: OptionSurfaceSnapshot) -> None:
    """Test that SVI model builds and runs."""
    T = 30 / 365.0
    model = build_svi_model(snapshot=snapshot, T=T)
    assert model is not None
    assert model.fit.n_points >= 4  # Should have enough points from the 7 strikes


def test_svi_prob_above(snapshot: OptionSurfaceSnapshot) -> None:
    """Test probability estimation."""
    T = 30 / 365.0
    model = build_svi_model(snapshot=snapshot, T=T)
    assert model is not None

    # ITM call / OTM put -> Low strike -> Prob > 0.5 (since S=100, K=80)
    p80 = model.prob_above(80)
    assert 0.8 < p80 < 1.0

    # ATM -> Prob ~ 0.5
    p100 = model.prob_above(100)
    assert 0.3 < p100 < 0.7

    # OTM call / ITM put -> High strike -> Prob < 0.5
    p120 = model.prob_above(120)
    assert 0.0 < p120 < 0.2


def test_svi_insufficient_data() -> None:
    """Test graceful failure with insufficient data."""
    snapshot = OptionSurfaceSnapshot(
        symbol="NVDA",
        expiration_date=datetime.now(UTC) + timedelta(days=30),
        calls=(),
        puts=(),
    )
    T = 30 / 365.0
    model = build_svi_model(snapshot=snapshot, T=T)
    assert model is None
