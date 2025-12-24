"""
Probability estimation models.

All models expose a uniform interface:

    def build(snapshot, **kwargs) -> Model | None

    class Model(Protocol):
        def prob_above(self, K: float) -> float: ...
"""

from typing import Protocol

from modeler.models.simple import SimpleModel, build_simple_model
from modeler.models.slope import SlopeModel, build_slope_model
from modeler.models.spline import SplineModel, build_spline_model
from modeler.models.svi import SVIModel, build_svi_model


class Model(Protocol):
    """Uniform interface for probability models."""

    def prob_above(self, K: float) -> float:
        """Return P(S_T > K) for strike K."""
        ...


__all__ = [
    "Model",
    "SimpleModel",
    "SlopeModel",
    "SplineModel",
    "SVIModel",
    "build_simple_model",
    "build_slope_model",
    "build_spline_model",
    "build_svi_model",
]
