import numpy as np
import pytest

from pythermalcomfort.classes_input import WorkIntensity
from pythermalcomfort.classes_return import WorkCapacity
from pythermalcomfort.models import work_capacity_hothaps


# helper to compute expected capacity
def _expected_capacity(wbgt, divisor, exponent):
    wbgt = np.asarray(wbgt)
    cap = 100 * (0.1 + 0.9 / (1 + (wbgt / divisor) ** exponent))
    return cap.clip(0, 100)


def test_scalar_heavy():
    result = work_capacity_hothaps(30.0, intensity=WorkIntensity.HEAVY.value)
    assert isinstance(result, WorkCapacity)
    assert isinstance(result.capacity, float)
    exp = _expected_capacity(30.0, divisor=30.94, exponent=16.64)
    assert pytest.approx(exp, rel=1e-3) == result.capacity


def test_scalar_moderate():
    result = work_capacity_hothaps(30.0, intensity=WorkIntensity.MODERATE.value)
    assert isinstance(result.capacity, float)
    exp = _expected_capacity(30.0, divisor=32.93, exponent=17.81)
    assert pytest.approx(exp, rel=1e-3) == result.capacity


def test_scalar_light():
    result = work_capacity_hothaps(30.0, intensity=WorkIntensity.LIGHT.value)
    assert isinstance(result.capacity, float)
    exp = _expected_capacity(30.0, divisor=34.64, exponent=22.72)
    assert pytest.approx(exp, rel=1e-3) == result.capacity


def test_list_input():
    wbgts = [20.0, 40.0]
    result = work_capacity_hothaps(wbgts, intensity="heavy")
    assert isinstance(result.capacity, np.ndarray)
    exp_list = _expected_capacity(wbgts, divisor=30.94, exponent=16.64).tolist()
    assert all(
        pytest.approx(e, rel=1e-3) == r for e, r in zip(exp_list, result.capacity)
    )


def test_low_wbgt_clamped_to_100():
    result = work_capacity_hothaps(0.0, intensity="light")
    assert result.capacity == pytest.approx(100.0, rel=1e-6)


def test_high_wbgt_approaches_10_percent():
    result = work_capacity_hothaps(100.0, intensity="moderate")
    assert result.capacity == pytest.approx(10.0, rel=1e-3)


def test_invalid_intensity_raises():
    with pytest.raises(ValueError):
        work_capacity_hothaps(30.0, intensity="invalid")
