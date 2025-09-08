import numpy as np

from pythermalcomfort.utilities import scale_windspeed


def test_scale_windspeed_identity_at_10m():
    # at h = 10 m, result must equal va
    assert scale_windspeed(va=5.0, h=10.0, round_output=False) == 5.0


def test_scale_windspeed_lower_height_matches_reference_math():
    # With z0=0.01 and base-10 log, factor at 2 m is log10(200)/3
    expected = 6.0 * (np.log10(200.0) / 3.0)
    got = scale_windspeed(va=6.0, h=2.0, round_output=False)
    assert np.isclose(got, expected, rtol=0, atol=1e-12)


def test_scale_windspeed_vectorized_inputs():
    va = np.array([5.0, 10.0])
    h = 2.0
    expected = va * (np.log10(200.0) / 3.0)
    got = scale_windspeed(va=va, h=h, round_output=False)
    assert np.allclose(got, expected, rtol=0, atol=1e-12)


def test_scale_windspeed_invalid_height_raises():
    import pytest

    with pytest.raises(ValueError):
        scale_windspeed(va=3.0, h=0.01)  # h must be > z0
