"""Shared test fixtures and configuration for plotting tests.

This module centralizes test-time configuration and data for plotting tests.
It defines:
- A model registry (list of ``ModelInfo``) with each model's callable,
  fixed parameters, default thresholds, recommended test ranges, and a
  numeric tolerance used when asserting curve accuracy.
- Autouse fixtures to enforce reproducibility and clean up figures.

Rationales for notable numeric defaults:
- Random seed = 0: matches common testing practices (also used by seaborn)
  so any stochastic helpers remain deterministic across runs.
- Tolerance = 0.1 (absolute): pragmatic allowance for numerical solving,
  grid discretization, and smoothing/rounding where applicable.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass

# Use a non-GUI backend to avoid opening windows in CI/headless environments
matplotlib.use("Agg")

# Import the main thermal comfort / heat risk models exercised by T-RH plots
from pythermalcomfort.models import (
    pmv_ppd_ashrae, pmv_ppd_iso, set_tmp, utci, heat_index_rothfusz
)
from pythermalcomfort.plots.matplotlib import ranges_tdb_rh
from pythermalcomfort.plots.utils import make_metric_eval, mapper_tdb_rh


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(0)
    yield


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def sample_model_result():
    """Create a mock model result object for metric extraction tests."""

    class MockResult:
        def __init__(self, pmv=0.0, ppd=5.0, set=25.0):
            self.pmv = pmv
            self.ppd = ppd
            self.set = set

    return MockResult


@pytest.fixture
def simple_linear_model():
    """Simple linear model used for utility/solver unit tests."""

    def model(tdb, rh, **kwargs):
        a = kwargs.get("a", 1.0)
        b = kwargs.get("b", 0.1)

        class Result:
            def __init__(self):
                self.metric = a * tdb + b * rh
                self.pmv = a * tdb + b * rh

        return Result()

    return model


@pytest.fixture
def fixed_params_pmv():
    """Standard fixed parameters for PMV-like tests."""
    return {
        "tr": 25.0,
        "met": 1.2,
        "clo": 0.5,
        "vr": 0.1,
        "wme": 0.0,
    }


@dataclass
class ModelInfo:
    """Model information bundle consumed by tests.

    - ``thresholds`` are the default values used when a test passes
      ``thresholds=None``. They reflect common bands used by each metric.
    - ``test_ranges`` define (tdb_min, tdb_max) and (rh_min, rh_max) windows
      where solving is practical and meaningful for the given model.
    - ``tolerance`` is the absolute tolerance applied when checking that a
      point on a plotted curve indeed satisfies ``metric == threshold`` within
      numerical limits.
    """
    name: str
    func: Callable
    fixed_params: Dict[str, Any]
    thresholds: List[float]
    metric_attr: Optional[str]
    test_ranges: Tuple[Tuple[float, float], Tuple[float, float]]  # (t_range, rh_range)
    tolerance: float = 0.1

# List of model configurations used by the T-RH plotting tests
MODEL_CONFIGS = [
    {
        "name": "pmv_ppd_iso",
        "func": pmv_ppd_iso,
        "fixed_params": {"tr": 25.0, "met": 1.2, "clo": 0.5, "vr": 0.1},
        "thresholds": [-0.5, 0.5],
        "metric_attr": "pmv",
        "test_ranges": ((20.0, 26.0), (30.0, 70.0)),
    },
    {
        "name": "pmv_ppd_ashrae",
        "func": pmv_ppd_ashrae,
        "fixed_params": {"tr": 25.0, "met": 1.2, "clo": 0.5, "vr": 0.1},
        "thresholds": [-0.5, 0.5],
        "metric_attr": "pmv",
        "test_ranges": ((20.0, 26.0), (30.0, 70.0)),
    },
    {
        "name": "set_tmp",
        "func": set_tmp,
        "fixed_params": {"tr": 25.0, "met": 1.2, "clo": 0.5, "v": 0.1, "body_surface_area": 1.8258},
        "thresholds": [22.0, 24.0, 26.0, 28.0, 32.0],
        "metric_attr": "set",
        "test_ranges": ((20.0, 30.0), (30.0, 70.0)),
    },
    {
        "name": "utci",
        "func": utci,
        "fixed_params": {"tr": 25.0, "v": 0.1},
        "thresholds": [-40.0, -27.0, -13.0, -1.0, 9.0, 26.0, 32.0, 38.0, 46.0],
        "metric_attr": "utci",
        "test_ranges": ((20.0, 30.0), (30.0, 70.0)),
    },
    {
        "name": "heat_index_rothfusz",
        "func": heat_index_rothfusz,
        "fixed_params": {},
        "thresholds": [30.0, 35.0, 40.0, 55.0],
        "metric_attr": "hi",
        "test_ranges": ((25.0, 35.0), (40.0, 80.0)),
    },
]

# Build model info objects from the configuration list
ALL_MODELS = [
    ModelInfo(
        name=cfg["name"],
        func=cfg["func"],
        fixed_params=cfg["fixed_params"],
        thresholds=cfg["thresholds"],
        metric_attr=cfg["metric_attr"],
        test_ranges=cfg["test_ranges"],
        tolerance=0.1,
    )
    for cfg in MODEL_CONFIGS
]

# All configured models have defaults; expose a convenience alias
MODELS_WITH_DEFAULTS = ALL_MODELS


@pytest.fixture
def all_models():
    """Provide all registered models."""
    return ALL_MODELS

@pytest.fixture
def models_with_defaults():
    """Provide models with default thresholds."""
    return MODELS_WITH_DEFAULTS