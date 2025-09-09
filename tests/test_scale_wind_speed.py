import numpy as np
import pytest

from pythermalcomfort.utilities import scale_wind_speed


def test_scale_wind_speed() -> None:
    """Test the scale_wind_speed function for wind profile scaling."""
    # Test case when height is 10 m should return the same wind speed
    assert np.isclose(scale_wind_speed(va=5.0, h=10.0), 5.0)
    assert np.allclose(scale_wind_speed(va=[3.0, 7.5], h=10.0), [3.0, 7.5])

    # Test case working is expected
    v10 = 5.0
    h = 2.0
    expected = v10 * (np.log10(h / 0.01) / np.log10(10 / 0.01))
    assert np.isclose(scale_wind_speed(v10, h), expected, atol=1e-12)

    # Test case with array input
    v10 = np.array([5.0, 8.0, 12.0])
    h = np.array([2.0, 5.0, 10.0])
    assert np.allclose(
        scale_wind_speed(v10, h),
        v10 * (np.log10(h / 0.01) / np.log10(10 / 0.01)),
        atol=1e-12,
    )

    # Test invalid height
    with pytest.raises(ValueError):
        scale_wind_speed(va=3.0, h=0.01)
    with pytest.raises(ValueError):
        scale_wind_speed(va=[3.0, 4.0], h=[0.02, 0.0])
