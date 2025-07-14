from __future__ import annotations

import numpy as np
import pytest

from pythermalcomfort.classes_return import WorkCapacity
from pythermalcomfort.models.work_capacity_iso import work_capacity_iso


def _expected_capacity(
    wbgt: float | list[float],
    met: float | list[float],
) -> np.ndarray:
    """Calculate the expected work capacity based on WBGT and metabolic rate."""
    wbgt_arr = np.array(wbgt)
    met_arr = np.array(met)
    met_rest = 117.0
    wbgt_lim = 34.9 - met_arr / 46.0
    wbgt_lim_rest = 34.9 - met_rest / 46.0
    cap = ((wbgt_lim_rest - wbgt_arr) / (wbgt_lim_rest - wbgt_lim)) * 100.0
    return np.clip(cap, 0, 100)


def test_scalar_typical() -> None:
    """Test that the function returns a WorkCapacity object with correct capacity."""
    wbgt = 30.0
    met = 200.0
    result = work_capacity_iso(wbgt, met)
    assert isinstance(result, WorkCapacity)
    assert isinstance(result.capacity, float)
    expected = _expected_capacity(wbgt, met)
    assert pytest.approx(expected, rel=1e-3) == result.capacity


def test_list_input_pairwise() -> None:
    """Test that the function handles list inputs correctly."""
    wbgts = [20.0, 30.0, 40.0]
    mets = [100.0, 200.0, 300.0]
    result = work_capacity_iso(wbgts, mets)
    assert isinstance(result.capacity, np.ndarray)
    expected = _expected_capacity(wbgts, mets)
    for got, exp in zip(result.capacity, expected, strict=False):
        assert pytest.approx(exp, rel=1e-3) == got


def test_exact_wbgt_lim_full_capacity() -> None:
    """Test that the function returns full capacity when WBGT limit is met."""
    met = 250.0
    wbgt_lim = 34.9 - met / 46.0
    result = work_capacity_iso(wbgt_lim, met)
    assert result.capacity == pytest.approx(100.0, abs=1e-6)


def test_resting_limit_zero_capacity() -> None:
    """Test that the function returns zero capacity when WBGT limit for resting is met."""
    met = 250.0
    wbgt_lim_rest = 34.9 - 117.0 / 46.0
    result = work_capacity_iso(wbgt_lim_rest, met)
    assert result.capacity == pytest.approx(0.0, abs=1e-6)


def test_low_wbgt_clamped_to_100() -> None:
    """Test that the function clamps low WBGT values to 100% capacity."""
    result = work_capacity_iso(0.0, 500.0)
    assert result.capacity == pytest.approx(100.0, abs=1e-6)


def test_high_wbgt_clamped_to_0() -> None:
    """Test that the function clamps high WBGT values to 0% capacity."""
    result = work_capacity_iso(100.0, 500.0)
    assert result.capacity == pytest.approx(0.0, abs=1e-6)


def test_negative_met_raises() -> None:
    """Test that the function raises ValueError for negative met."""
    with pytest.raises(ValueError):
        work_capacity_iso(25.0, -10.0)


def test_met_above_max_raises() -> None:
    """Test that the function raises ValueError for met above maximum."""
    with pytest.raises(ValueError):
        work_capacity_iso(25.0, 3000.0)
