import numpy as np
import pytest

from pythermalcomfort.classes_return import THI
from pythermalcomfort.models import thi
from tests.conftest import is_equal


@pytest.mark.parametrize(
    "tdb, rh, expected",
    [
        (30.0, 70.0, 81.4),
        (20.0, 50.0, 65.2),
    ],
)
def test_scalar_rounding_default(tdb, rh, expected):
    result = thi(tdb, rh)
    assert isinstance(result, THI)
    assert isinstance(result.thi, float)
    assert is_equal(result.thi, expected)


def test_scalar_no_rounding():
    tdb, rh = 30.0, 70.0
    expected = 1.8 * tdb + 32 - 0.55 * (1 - 0.01 * rh) * (1.8 * tdb - 26)
    result = thi(tdb, rh, round_output=False)
    assert is_equal(result.thi, expected)


def test_list_input():
    tdb_list = [30, 20]
    rh_list = [70, 50]
    result = thi(tdb_list, rh_list)
    expected = [81.4, 65.2]

    assert isinstance(result.thi, np.ndarray)
    assert is_equal(result.thi, expected)


@pytest.mark.parametrize(
    "tdb, rh, expected_error",
    [
        (25.0, -5.0, ValueError),
        (25.0, 150.0, ValueError),  # humidity should be between 0 and 100
        ("hot", "humid", TypeError),  # invalid types
    ],
)
def test_invalid_inputs_raise_specific(tdb, rh, expected_error):
    with pytest.raises(expected_error):
        thi(tdb, rh)
