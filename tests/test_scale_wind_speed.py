import numpy as np
import pytest

from pythermalcomfort.utilities import scale_wind_speed


def test_scale_wind_speed() -> None:
    """Test the scale_wind_speed function for wind profile scaling."""
    # Test case when height is 10 m should return the same wind speed
    assert np.isclose(scale_wind_speed(va=5.0, h=10.0), 5.0)
    assert np.allclose(scale_wind_speed(va=[3.0, 7.5], h=10.0), [3.0, 7.5])

    # Test expected scaling at h = 2 m
    v10 = 5.0
    h = 2.0
    expected = v10 * (np.log10(h / 0.01) / np.log10(10 / 0.01))
    assert np.isclose(scale_wind_speed(v10, h), expected, atol=1e-12)

    # Test case with array input
    v10 = [5.0, 8.0, 12.0]
    h = [2.0, 5.0, 10.0]
    z0 = [0.01, 0.03, 0.1]
    assert np.allclose(
        scale_wind_speed(v10, h, z0),
        v10
        * (np.log10(np.asarray(h) / np.asarray(z0)) / np.log10(10 / np.asarray(z0))),
        atol=1e-12,
    )

    # Broadcasting: scalar va with vector h
    v10_scalar = 6.0
    h_vec = [1.5, 10.0, 20.0]
    assert np.allclose(
        scale_wind_speed(v10_scalar, h_vec),
        v10_scalar * (np.log10(np.asarray(h_vec) / 0.01) / np.log10(10 / 0.01)),
        atol=1e-12,
    )

    #  --- Invalid: not broadcastable shapes ---
    with pytest.raises(ValueError):
        scale_wind_speed(va=[3.0, 4.0], h=[2.0, 5.0, 10.0])

    # --- Invalid: non-finite inputs ---
    with pytest.raises(ValueError):
        scale_wind_speed(np.nan, 2.0)
    with pytest.raises(ValueError):
        scale_wind_speed(5.0, np.inf)
    with pytest.raises(ValueError):
        scale_wind_speed(5.0, 2.0, z0=np.nan)

    # --- Invalid: negative va ---
    with pytest.raises(ValueError):
        scale_wind_speed(va=-0.1, h=2.0)

    # --- Invalid: z0 bounds (0 < z0 < 10) ---
    with pytest.raises(ValueError):
        scale_wind_speed(va=3.0, h=2.0, z0=0.0)
    with pytest.raises(ValueError):
        scale_wind_speed(va=3.0, h=2.0, z0=10.0)
    with pytest.raises(ValueError):
        scale_wind_speed(va=3.0, h=2.0, z0=11.0)

    # --- Invalid: h <= z0 ---
    with pytest.raises(ValueError):
        scale_wind_speed(va=3.0, h=0.01, z0=0.01)
    with pytest.raises(ValueError):
        scale_wind_speed(va=[3.0, 4.0], h=[0.02, 0.01], z0=[0.02, 0.01])
