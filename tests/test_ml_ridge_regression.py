import numpy as np
import pytest

from pythermalcomfort.models.ml_ridge_regression import (
    Sex,
    _inverse_scale_output,
    _scale_features,
    ridge_regression_predictor,
)


def test_ridge_regression_scale_features():
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


def test_ridge_regression_inverse_scale_output():
    """Test the inverse output scaling function."""
    # Create a sample scaled output array: [scaled_tre, scaled_mtsk]
    scaled_output = np.array([[0.54714297, 0.51841453]])
    inversed = _inverse_scale_output(scaled_output)
    assert inversed.shape == (1, 2)
    # Check if inverse scaling is applied correctly
    expected = np.array([[37.15, 32.15]])
    np.testing.assert_allclose(inversed, expected, rtol=1e-6)


def test_ridge_regression_predictor_scalar():
    """Test the model with scalar inputs."""
    result = ridge_regression_predictor(
        sex=Sex.MALE.value,
        age=30,
        height_cm=180,
        mass_kg=75,
        ambient_temp=35,
        humidity=60,
        duration_minutes=540,
    )
    assert isinstance(result.rectal_temp, float)
    assert isinstance(result.skin_temp, float)
    # Check against known values from the docstring example
    assert result.rectal_temp == pytest.approx(37.98, abs=1e-2)
    assert result.skin_temp == pytest.approx(37.02, abs=1e-2)


def test_ridge_regression_predictor_vectorized():
    """Test the model with array inputs for vectorization."""
    sex = [Sex.MALE.value, Sex.FEMALE.value]
    age = [30, 45]
    height_cm = [180, 165]
    mass_kg = [75, 60]
    ambient_temp = [35, 40]
    humidity = [60, 50]

    result = ridge_regression_predictor(
        sex=sex,
        age=age,
        height_cm=height_cm,
        mass_kg=mass_kg,
        ambient_temp=ambient_temp,
        humidity=humidity,
        duration_minutes=540,
    )
    assert isinstance(result.rectal_temp, np.ndarray)
    assert result.rectal_temp.shape == (2,)
    assert isinstance(result.skin_temp, np.ndarray)
    assert result.skin_temp.shape == (2,)

    # Check results for each case against docstring examples
    assert result.rectal_temp[0] == pytest.approx(37.98, abs=1e-2)
    assert result.skin_temp[0] == pytest.approx(37.02, abs=1e-2)
    assert result.rectal_temp[1] == pytest.approx(38.42, abs=1e-2)
    assert result.skin_temp[1] == pytest.approx(38.04, abs=1e-2)


def test_ridge_regression_broadcasting():
    """Test NumPy broadcasting with mixed scalar and array inputs."""
    sex = [Sex.MALE.value, Sex.FEMALE.value]
    age = 35  # scalar
    height_cm = 170  # scalar
    mass_kg = 70  # scalar
    ambient_temp = [30, 35]
    humidity = 50  # scalar

    result = ridge_regression_predictor(
        sex=sex,
        age=age,
        height_cm=height_cm,
        mass_kg=mass_kg,
        ambient_temp=ambient_temp,
        humidity=humidity,
        duration_minutes=540,
    )
    assert result.rectal_temp.shape == (2,)
    assert result.skin_temp.shape == (2,)

    # Check against pre-calculated values for this specific broadcast scenario
    assert result.rectal_temp[0] == pytest.approx(37.61, abs=1e-2)
    assert result.skin_temp[0] == pytest.approx(35.92, abs=1e-2)
    assert result.rectal_temp[1] == pytest.approx(38.04, abs=1e-2)
    assert result.skin_temp[1] == pytest.approx(37.02, abs=1e-2)

    # Ensure it raises error for incompatible shapes
    with pytest.raises(ValueError):
        ridge_regression_predictor(
            sex=[0, 1],
            age=[30, 40, 50],  # Incompatible shape
            height_cm=170,
            mass_kg=70,
            ambient_temp=30,
            humidity=50,
            duration_minutes=540,
        )
