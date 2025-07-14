import numpy as np
import pytest

from pythermalcomfort.classes_return import GaggeTwoNodesSleep
from pythermalcomfort.models import two_nodes_gagge_sleep


def test_two_nodes_gagge_sleep_single_input() -> None:
    """Test the two_nodes_gagge_sleep function with scalar inputs."""
    result = two_nodes_gagge_sleep(18, 18, 0.05, 50, 1.4, thickness_quilt=1.76)

    # expected outputs
    expected = {
        "set": np.array([28.62412397]),
        "t_core": np.array([37.0322359]),
        "t_skin": np.array([33.71180903]),
        "wet": np.array([0.38381668]),
        "t_sens": np.array([1.15402501]),
        "disc": np.array([2.29254673]),
        "e_skin": np.array([27.10243546]),
        "met_shivering": np.array([0.0]),
        "alfa": np.array([0.13731683]),
        "skin_blood_flow": np.array([7.21402727]),
    }

    # compare each field with a reasonable tolerance
    for field, exp in expected.items():
        actual = getattr(result, field)
        np.testing.assert_allclose(actual, exp, rtol=1e-6, atol=1e-8)


def test_two_nodes_gagge_sleep_long_duration() -> None:
    """Test the two_nodes_gagge_sleep function with a longer duration input."""
    duration = 481
    ta = np.repeat(18, duration)
    tr = np.repeat(18, duration)
    vel = np.repeat(0.05, duration)
    rh = np.repeat(50, duration)
    clo_a = np.repeat(1.4, duration)
    thickness1 = np.repeat(1.76, duration)

    result = two_nodes_gagge_sleep(ta, tr, vel, rh, clo_a, thickness1)

    # Assert return type and shape
    assert isinstance(result, GaggeTwoNodesSleep)
    assert result.set.shape == (duration,)
    assert result.t_core.shape == (duration,)
    assert result.t_skin.shape == (duration,)

    first_row_expected = [
        28.624124,
        37.032236,
        33.711809,
        0.383817,
        1.154025,
        2.292547,
        27.102435,
        0.0,
        0.137317,
        7.214027,
    ]
    last_row_expected = [
        24.908947,
        36.235312,
        32.397438,
        0.060000,
        -0.469718,
        -0.469718,
        3.799124,
        0.0,
        0.191773,
        4.382500,
    ]

    fields = [
        "set",
        "t_core",
        "t_skin",
        "wet",
        "t_sens",
        "disc",
        "e_skin",
        "met_shivering",
        "alfa",
        "skin_blood_flow",
    ]

    # check first row
    for field, exp in zip(fields, first_row_expected, strict=False):
        np.testing.assert_allclose(
            getattr(result, field)[0],
            exp,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"first {field} mismatch",
        )

    # check last row
    for field, exp in zip(fields, last_row_expected, strict=False):
        np.testing.assert_allclose(
            getattr(result, field)[-1],
            exp,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"last {field} mismatch",
        )


def test_length_mismatch_raises_value_error() -> None:
    """Test that length mismatch in input lists raises ValueError."""
    with pytest.raises(ValueError) as exc:
        two_nodes_gagge_sleep(
            [18, 18],
            [18],  # length mismatch
            [0.05, 0.05],
            [50, 50],
            [1.4, 1.4],
            [1.76, 1.76],
        )

    assert "must have the same length" in str(exc.value)


def test_unexpected_kwargs_raises_type_error() -> None:
    """Test that unexpected keyword arguments raise TypeError."""
    with pytest.raises(TypeError) as exc:
        two_nodes_gagge_sleep(18, 18, 0.05, 50, 1.4, 1.76, foo=123)
    assert "Unexpected keyword arguments" in str(exc.value)


def test_invalid_kwarg_type_raises_type_error() -> None:
    """Test that a non-numeric type for tdb raises TypeError."""
    with pytest.raises(TypeError) as exc:
        two_nodes_gagge_sleep("string", 18, 0.05, 50, 1.4, 1.76)
    msg = str(exc.value)
    assert "tdb" in msg


def test_tickness_quilt_negative() -> None:
    """Test that a negative thickness_quilt raises ValueError."""
    with pytest.raises(ValueError):
        two_nodes_gagge_sleep(18, 18, 0.05, 50, 1.4, thickness_quilt=-1.76)
