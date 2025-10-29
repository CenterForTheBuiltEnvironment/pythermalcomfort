"""Shared test fixtures and configuration for plotting tests."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Use non-GUI backend for tests
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility.
    
    Using seed=0 like seaborn for consistency.
    """
    np.random.seed(0)
    yield


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def sample_model_result():
    """Create a mock model result object for testing."""

    class MockResult:
        def __init__(self, pmv=0.0, ppd=5.0, set=25.0):
            self.pmv = pmv
            self.ppd = ppd
            self.set = set

    return MockResult


@pytest.fixture
def simple_linear_model():
    """Simple linear model: metric = a*tdb + b*rh for testing."""

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
    """Standard fixed parameters for PMV model testing."""
    return {
        "tr": 25.0,
        "met": 1.2,
        "clo": 0.5,
        "vr": 0.1,
        "wme": 0.0,
    }