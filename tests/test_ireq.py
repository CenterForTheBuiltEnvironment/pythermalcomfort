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
    result = calc_ireq(
        m=100,
        w_work=0,
        tdb=0,
        tr=0,
        p_air=2,
        v_walk=0.5,
        v=0.5,
        rh=50,
        clo=1.0,
    )

    # Expect dataclass-like output
    assert hasattr(result, "ireq_neutral")
    assert hasattr(result, "ireq_min")
    assert hasattr(result, "icl_neutral")
    assert hasattr(result, "icl_min")

    # Physically valid outputs
    assert result.ireq_neutral >= 0
    assert result.ireq_min >= 0
    assert result.ireq_neutral >= result.ireq_min


def test_calc_ireq_array():
    """Test calc_ireq handles numpy arrays correctly."""
    m = np.array([100, 150])
    tdb = np.array([-10, 0])
    tr = np.array([-10, 0])
    result = calc_ireq(
        m=m, w_work=0, tdb=tdb, tr=tr, p_air=2, v_walk=0.5, v=0.5, rh=50, clo=1.0
    )

    for attr in ["ireq_neutral", "ireq_min"]:
        val = getattr(result, attr)
        assert isinstance(val, np.ndarray)
        assert val.shape == m.shape
        assert np.all(val >= 0)


def test_calc_ireq_broadcasting():
    """Test broadcasting behavior when mixing scalar and array inputs."""
    tdb = np.array([-10, 0, 10])
    result = calc_ireq(
        m=150, w_work=0, tdb=tdb, tr=tdb, p_air=2, v_walk=0.5, v=0.5, rh=50, clo=1.0
    )

    for attr in ["ireq_neutral", "ireq_min"]:
        val = getattr(result, attr)
        assert val.shape == tdb.shape
        assert np.all(np.isfinite(val))


def test_calc_ireq_invalid_inputs():
    """Test invalid inputs raise proper errors."""
    # String input should raise a ValueError
    with pytest.raises(ValueError):
        calc_ireq(
            m="abc", w_work=0, tdb=0, tr=0, p_air=2, v_walk=0.5, v=0.5, rh=50, clo=1.0
        )

    # Missing required parameters should raise a TypeError
    with pytest.raises(TypeError):
        calc_ireq()

    # ValueError should be raised when array lengths do not match
    with pytest.raises(ValueError):
        calc_ireq(
            m=[100, 120],
            w_work=0,
            tdb=[0],
            tr=[0],
            p_air=2,
            v_walk=0.5,
            v=0.5,
            rh=50,
            clo=1.0,
        )


def test_calc_ireq_edge_cases():
    """Test numeric stability with extreme input values."""
    cold = calc_ireq(
        m=150, w_work=0, tdb=-30, tr=-30, p_air=2, v_walk=0.5, v=1, rh=50, clo=1.0
    )
    hot = calc_ireq(
        m=150, w_work=0, tdb=10, tr=10, p_air=2, v_walk=0.5, v=1, rh=50, clo=0.5
    )

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
        # tdb, tr, v, m, expected_ireq_neutral
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
        result = calc_ireq(
            m=m,
            w_work=0,
            tdb=tdb,
            tr=tr,
            p_air=8.0,
            v_walk=v,
            v=v,
            rh=85,
            clo=1.0,  # CLO doesn't affect IREQ_neutral directly in comparison
        )

        assert_allclose(result.ireq_neutral, ireq_ref, rtol=0.1)
