import numpy as np
import pytest

from pythermalcomfort.utils import scale_wind_speed_log


def test_compare_results_wind_profile_calculator() -> None:
    """Compare results with Wind Profile Calculator online tool.

    Reference:
    https://wind-data.ch/tools/profile.php?h=2&v=10&z0=0.01&abfrage=Refresh
    """
    v10 = 6.52  # m/s
    z1 = 10  # m
    z2 = 2.0  # m
    z0 = 0.01  # m
    expected = 5  # m/s from Wind Profile Calculator
    result = scale_wind_speed_log(v_z1=v10, z2=z2, z1=z1, z0=z0, round_output=False)
    print(result)
    assert np.allclose(result.v_z2, expected, rtol=1e-2)

    v10 = 7.69  # m/s
    z2 = 2.0  # m
    z0 = 0.1  # m
    expected = 5  # m/s from Wind Profile Calculator
    result = scale_wind_speed_log(v_z1=v10, z2=z2, z1=10, z0=z0, round_output=False)
    print(result)
    assert np.allclose(result.v_z2, expected, rtol=1e-2)

    v_z1 = 5.0  # m/s
    z2 = 90.0  # m
    z0 = 0.1  # m
    expected = 7.39  # m/s from Wind Profile Calculator
    result = scale_wind_speed_log(v_z1=v_z1, z2=z2, z1=10, z0=z0, round_output=False)
    print(result)
    assert np.allclose(result.v_z2, expected, rtol=1e-2)


def test_scale_winds_speed_scalar() -> None:
    """Test scaling wind speed from 10m to 2m (scalar inputs)."""
    v10 = 5.0
    z2 = 2.0
    expected = v10 * np.log((z2 - 0.0) / 0.01) / np.log((10.0 - 0.0) / 0.01)
    result = scale_wind_speed_log(v10, z2, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-5)


def test_scale_winds_speed_array() -> None:
    """Test scaling wind speed for array inputs."""
    v10 = np.asarray([3.0, 5.0])
    z2 = np.asarray([1.5, 2.5])
    expected = v10 * np.log((z2 - 0.0) / 0.01) / np.log((10.0 - 0.0) / 0.01)
    result = scale_wind_speed_log(v10, z2, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-5)


def test_scale_wind_speed_broadcasting() -> None:
    """Test broadcasting with different z0 for each measurement."""
    v10 = [3.0, 5.0]
    z2 = [1.5, 2.5]
    z0 = [0.01, 0.1]
    expected = np.asarray(
        [
            3.0 * np.log((1.5 - 0.0) / 0.01) / np.log((10.0 - 0.0) / 0.01),
            5.0 * np.log((2.5 - 0.0) / 0.1) / np.log((10.0 - 0.0) / 0.1),
        ]
    )
    result = scale_wind_speed_log(v10, z2, z0=z0, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-5)


def test_scale_winds_speed_with_displacement() -> None:
    """Test with nonzero displacement height d."""
    v10 = 5.0
    z2 = 2.0
    z1 = 10.0
    z0 = 0.1
    d = 0.5
    expected = v10 * np.log((z2 - d) / z0) / np.log((z1 - d) / z0)
    result = scale_wind_speed_log(v10, z2, z1=z1, z0=z0, d=d, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-5)


def test_invalid_types() -> None:
    """Test that invalid types raise TypeError."""
    with pytest.raises(TypeError):
        scale_wind_speed_log("bad", 2.0)
    with pytest.raises(TypeError):
        scale_wind_speed_log(5.0, "bad")
    with pytest.raises(TypeError):
        scale_wind_speed_log(5.0, 2.0, z0="bad")


def test_negative_and_zero_values() -> None:
    """Test that negative and zero values raise ValueError."""
    # Negative wind speed
    with pytest.raises(ValueError):
        scale_wind_speed_log(-1.0, 2.0, round_output=False)
    # Negative z2
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, -2.0, round_output=False)
    # z0 <= 0
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 2.0, z0=0.0, round_output=False)
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 2.0, z0=-0.1, round_output=False)
    # z2 <= d
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 2.0, d=2.0, round_output=False)
    # z1 <= d
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 2.0, z1=1.0, d=1.0, round_output=False)
    # z2 <= z0
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 0.01, z0=0.01, round_output=False)
    # z1 <= z0
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 2.0, z1=0.01, z0=0.01, round_output=False)


def test_edge_case_z2_less_than_z1() -> None:
    """Test scaling when z2 < z1 (should still work if all constraints are met)."""
    v10 = 5.0
    z2 = 2.0
    z1 = 10.0
    result = scale_wind_speed_log(v10, z2, z1=z1, round_output=False)
    assert result.v_z2 < v10


def test_large_and_small_z0() -> None:
    """Test with very large and very small roughness lengths."""
    v10 = 5.0
    z2 = 2.0
    # Very small z0
    result_small = scale_wind_speed_log(v10, z2, z0=1e-6, round_output=False)
    assert result_small.v_z2 > 0
    # Very large z0 (should be close to zero wind speed)
    result_large = scale_wind_speed_log(v10, z2, z0=1.0, round_output=False)
    assert result_large.v_z2 > 0
