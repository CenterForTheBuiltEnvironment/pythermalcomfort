import numpy as np
import pytest

from pythermalcomfort.classes_return import WorkCapacity
from pythermalcomfort.models import work_capacity_dunne


def test_heavy_scalar_at_25():
    wc = work_capacity_dunne(25, "heavy")
    assert isinstance(wc, WorkCapacity)
    assert wc.capacity == pytest.approx(100)


def test_heavy_array_and_clipping():
    wbgt = [20, 25, 28, 30, 35]
    wc = work_capacity_dunne(wbgt, "heavy")
    expected_base = np.clip(
        100 - 25 * np.maximum(0, np.array(wbgt) - 25) ** (2 / 3), 0, 100
    )
    assert isinstance(wc, WorkCapacity)
    assert np.allclose(wc.capacity, expected_base)


def test_moderate_and_light_intensity():
    wbgt = 30
    base = np.clip(100 - 25 * (5) ** (2 / 3), 0, 100)
    wc_med = work_capacity_dunne(wbgt, "moderate")
    wc_light = work_capacity_dunne(wbgt, "light")
    assert wc_med.capacity == pytest.approx(np.clip(base * 2, 0, 100))
    assert wc_light.capacity == pytest.approx(np.clip(base * 4, 0, 100))


def test_lower_bound_clipping():
    wc = work_capacity_dunne(100, "heavy")
    assert wc.capacity == pytest.approx(0)


def test_invalid_intensity_raises():
    with pytest.raises(ValueError):
        work_capacity_dunne(25, "invalid")
