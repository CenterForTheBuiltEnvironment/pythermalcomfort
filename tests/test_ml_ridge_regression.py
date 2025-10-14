import numpy as np
import pytest

from pythermalcomfort.utilities import Sex
from pythermalcomfort.models.ml_ridge_regression import (
    _inverse_scale_output,
    _scale_features,
    ml_ridge_regression,
)


def test_ml_ridge_regression_scale_features():
    """Test the feature scaling function."""
    # Create a sample feature array: [sex, age, height, mass, temp, humidity, tre, mtsk]
    features = np.array([[0, 30, 180, 75, 25, 50, 37.0, 34.0]])
    scaled = _scale_features(features)
    assert scaled.shape == (1, 8)
    # Check if the scaling is applied correctly by comparing with pre-calculated values
    expected = np.array(
        [
            [
                0.0,
                0.18032787,
                0.72916667,
                0.34182563,
                0.16666667,
                1.0,
                0.49972339,
                0.6549385,
            ]
        ]
    )
    np.testing.assert_allclose(scaled, expected, rtol=1e-6)


def test_ml_ridge_regression_inverse_scale_output():
    """Test the inverse output scaling function."""
    # Create a sample scaled output array: [scaled_tre, scaled_mtsk]
    scaled_output = np.array([[0.54714297, 0.51841453]])
    inversed = _inverse_scale_output(scaled_output)
    assert inversed.shape == (1, 2)
    # Check if inverse scaling is applied correctly
    expected = np.array([[37.15, 32.15]])
    np.testing.assert_allclose(inversed, expected, rtol=1e-6)


def test_ml_ridge_regression_scalar():
    """Test the model with scalar inputs."""
    result = ml_ridge_regression(
        sex=Sex.male.value,
        age=60,
        height=1.80,
        weight=75,
        tdb=35,
        rh=60,
        duration=540,
    )
    assert isinstance(result.t_re, float)
    assert isinstance(result.t_sk, float)
    # Check against known values from the docstring example
    assert result.t_re == pytest.approx(38.15, abs=1e-2)
    assert result.t_sk == pytest.approx(37.02, abs=1e-2)


def test_ml_ridge_regression_vectorized():
    """Test the model with array inputs for vectorization."""
    sex = [Sex.male.value, Sex.female.value]
    age = [60, 65]
    height = [1.80, 1.65]
    weight = [75, 60]
    tdb = [35, 40]
    rh = [60, 50]

    result = ml_ridge_regression(
        sex=sex,
        age=age,
        height=height,
        weight=weight,
        tdb=tdb,
        rh=rh,
        duration=540,
    )
    assert isinstance(result.t_re, np.ndarray)
    assert result.t_re.shape == (2,)
    assert isinstance(result.t_sk, np.ndarray)
    assert result.t_sk.shape == (2,)

    # Check results for each case against docstring examples
    assert result.t_re[0] == pytest.approx(38.15, abs=1e-2)
    assert result.t_sk[0] == pytest.approx(37.02, abs=1e-2)
    assert result.t_re[1] == pytest.approx(38.53, abs=1e-2)
    assert result.t_sk[1] == pytest.approx(38.04, abs=1e-2)


def test_ml_ridge_regression_broadcasting():
    """Test NumPy broadcasting with mixed scalar and array inputs."""
    sex = [Sex.male.value, Sex.female.value]
    age = 75  # scalar
    height = 1.70  # scalar
    weight = 70  # scalar
    tdb = [30, 35]
    rh = 50  # scalar

    result = ml_ridge_regression(
        sex=sex,
        age=age,
        height=height,
        weight=weight,
        tdb=tdb,
        rh=rh,
        duration=540,
    )
    assert result.t_re.shape == (2,)
    assert result.t_sk.shape == (2,)

    # Check against pre-calculated values for this specific broadcast scenario
    assert result.t_re[0] == pytest.approx(37.84, abs=1e-2)
    assert result.t_sk[0] == pytest.approx(35.92, abs=1e-2)
    assert result.t_re[1] == pytest.approx(38.26, abs=1e-2)
    assert result.t_sk[1] == pytest.approx(37.02, abs=1e-2)

    # Ensure it raises error for incompatible shapes
    with pytest.raises(ValueError):
        ml_ridge_regression(
            sex=[Sex.male.value, Sex.female.value],
            age=[30, 40, 50],  # Incompatible shape
            height=1.70,
            weight=70,
            tdb=30,
            rh=50,
            duration=540,
        )

def test_ml_ridge_regression_initial_body_temp():
    """Test the model with scalar inputs."""
    result = ml_ridge_regression(
        sex=Sex.male.value,
        age=70,
        height=1.80,
        weight=75,
        tdb=35,
        rh=60,
        duration=60,
        initial_t_re=37.0,
        initial_t_sk=32.0,
    )
    assert isinstance(result.t_re, float)
    assert isinstance(result.t_sk, float)
    # Check against known values from the docstring example
    assert result.t_re == pytest.approx(37.33, abs=1e-2)
    assert result.t_sk == pytest.approx(37.00, abs=1e-2)
