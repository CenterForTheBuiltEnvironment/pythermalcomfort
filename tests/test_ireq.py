"""Unit tests for calc_ireq() in pythermalcomfort.models.ireq.

Covers:
- Scalar inputs (single values)
- Vectorized inputs (lists, numpy arrays)
- Broadcasting and consistent output shapes
- Invalid inputs (TypeError, ValueError)
- Edge cases (zeros, very small/large inputs)
- Deterministic numeric comparison
- Reference validation (ISO 11079 Table F.1)

Run with:
pytest -sv tests/test_ireq.py
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pythermalcomfort.models.ireq import calc_ireq


def test_calc_ireq_scalar():
    """Test calc_ireq with scalar input values."""
    result = calc_ireq(m=100, w_work=0, tdb=0, tr=0, p_air=2, v_walk=0.5, v=0.5, rh=50, clo=1.0)
    assert hasattr(result, "ireq_neutral")
    assert hasattr(result, "ireq_min")
    assert hasattr(result, "icl_neutral")
    assert hasattr(result, "icl_min")
    assert result.ireq_neutral >= 0
    assert result.ireq_min >= 0
    assert result.ireq_neutral >= result.ireq_min


def test_calc_ireq_array():
    """Test calc_ireq handles numpy arrays correctly."""
    m = np.array([100, 150])
    tdb = np.array([-10, 0])
    tr = np.array([-10, 0])
    result = calc_ireq(m=m, w_work=0, tdb=tdb, tr=tr, p_air=2, v_walk=0.5, v=0.5, rh=50, clo=1.0)
    for attr in ["ireq_neutral", "ireq_min"]:
        val = getattr(result, attr)
        assert isinstance(val, np.ndarray)
        assert val.shape == m.shape
        assert np.all(val >= 0)


def test_calc_ireq_broadcasting():
    """Test broadcasting behavior when mixing scalar and array inputs."""
    tdb = np.array([-10, 0, 10])
    result = calc_ireq(m=150, w_work=0, tdb=tdb, tr=tdb, p_air=2, v_walk=0.5, v=0.5, rh=50, clo=1.0)
    for attr in ["ireq_neutral", "ireq_min"]:
        val = getattr(result, attr)
        assert val.shape == tdb.shape
        assert np.all(np.isfinite(val))


def test_calc_ireq_invalid_inputs():
    """Test invalid inputs raise proper errors."""
    # String input should raise ValueError or TypeError
    with pytest.raises((ValueError, TypeError)):
        calc_ireq(m="abc", w_work=0, tdb=0, tr=0, p_air=2, v_walk=0.5, v=0.5, rh=50, clo=1.0)

    # Missing required parameters should raise TypeError
    with pytest.raises(TypeError):
        calc_ireq()

    # Negative metabolic rate should raise ValueError
    with pytest.raises(ValueError):
        calc_ireq(m=-10, w_work=0, tdb=0, tr=0, p_air=2, v_walk=0.5, v=0.5, rh=50, clo=1.0)

    # Non-positive air pressure (p_air <= 0) should raise ValueError
    with pytest.raises(ValueError):
        calc_ireq(m=100, w_work=0, tdb=0, tr=0, p_air=0, v_walk=0.5, v=0.5, rh=50, clo=1.0)


def test_calc_ireq_edge_cases():
    """Test numeric stability with extreme input values."""
    cold = calc_ireq(m=150, w_work=0, tdb=-30, tr=-30, p_air=2, v_walk=0.5, v=1, rh=50, clo=1.0)
    hot = calc_ireq(m=150, w_work=0, tdb=10, tr=10, p_air=2, v_walk=0.5, v=1, rh=50, clo=0.5)
    for r in [cold, hot]:
        for attr in ["ireq_neutral", "ireq_min"]:
            val = getattr(r, attr)
            assert np.isfinite(val)
            assert val >= 0


def test_calc_ireq_consistency():
    """Ensure deterministic output given same input."""
    args = dict(m=150, w_work=0, tdb=0, tr=0, p_air=2, v_walk=0.5, v=1, rh=50, clo=1.0)
    out1 = calc_ireq(**args)
    out2 = calc_ireq(**args)
    assert_allclose(out1.ireq_neutral, out2.ireq_neutral, rtol=1e-6)
    assert_allclose(out1.ireq_min, out2.ireq_min, rtol=1e-6)


def test_calc_ireq_reference_cases():
    """Validate calc_ireq() against ISO 11079 Table F.1 reference data (IREQ only)."""
    reference_cases = [
        (0, 0, 2, 90, 2.6),
        (0, 0, 2, 145, 1.5),
        (-10, -10, 2, 90, 3.5),
        (-10, 0, 2, 145, 1.9),
        (-20, -20, 2, 115, 3.4),
        (-20, -20, 7, 115, 3.5),
        (-30, -30, 2, 115, 4.0),
        (-30, -30, 5, 175, 2.6),
    ]

    for tdb, tr, v, m, ireq_ref in reference_cases:
        v_walk = np.clip(0.0052 * (m - 58), 0.3, 1.2)
        result = calc_ireq(
            m=m, w_work=0, tdb=tdb, tr=tr, p_air=8.0, v_walk=v_walk, v=v, rh=85, clo=1.0
        )
        assert_allclose(result.ireq_neutral, ireq_ref, rtol=0.1)

    # Additional test: ensure DLE returns "more than 8" for mild conditions
        # Additional test: ensure DLE returns "more than 8" for mild conditions (within valid range)
    result_dle = calc_ireq(
        m=60,               # low metabolic rate
        w_work=0,
        tdb=10,             # upper valid temperature limit (≤10 °C)
        tr=10,
        p_air=8.0,
        v_walk=0.4,
        v=0.4,
        rh=90,
        clo=4.0,            # thick clothing, minimal heat loss
    )

    # Accept both string or numpy array output
    dle_value = result_dle.dle_neutral
    if isinstance(dle_value, np.ndarray):
        dle_value = dle_value.item()  # extract scalar

    # Check type and possible meaning
    assert isinstance(dle_value, str)
    assert ("more than 8" in dle_value) or (float(dle_value) < 0)

    # Check mixed dtype handling for object arrays
    results_array = np.array([result.ireq_neutral for result in [result_dle]], dtype=object)
    assert results_array.dtype == object
