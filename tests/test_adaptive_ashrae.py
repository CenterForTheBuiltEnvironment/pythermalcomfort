import numpy as np
import pytest

from pythermalcomfort.models import adaptive_ashrae
from pythermalcomfort.utilities import Units
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_adaptive_ashrae(get_test_url, retrieve_data) -> None:
    """Test that the Adaptive ASHRAE function calculates correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.ADAPTIVE_ASHRAE.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        units = inputs.get("units", Units.SI.value)
        result = adaptive_ashrae(
            inputs["tdb"],
            inputs["tr"],
            inputs["t_running_mean"],
            inputs["v"],
            units,
        )

        validate_result(result, outputs, tolerance)


def test_ashrae_inputs_invalid_units() -> None:
    """Test that the function raises a ValueError for invalid units."""
    with pytest.raises(ValueError):
        adaptive_ashrae(tdb=25, tr=25, t_running_mean=20, v=0.1, units="INVALID")


def test_ashrae_inputs_invalid_tdb_type() -> None:
    """Test that the function raises a TypeError for invalid tdb type."""
    with pytest.raises(TypeError):
        adaptive_ashrae(tdb="invalid", tr=25, t_running_mean=20, v=0.1)


def test_ashrae_inputs_invalid_tr_type() -> None:
    """Test that the function raises a TypeError for invalid tr type."""
    with pytest.raises(TypeError):
        adaptive_ashrae(tdb=25, tr="invalid", t_running_mean=20, v=0.1)


def test_ashrae_inputs_invalid_t_running_mean_type() -> None:
    """Test that the function raises a TypeError for invalid t_running_mean type."""
    with pytest.raises(TypeError):
        adaptive_ashrae(tdb=25, tr=25, t_running_mean="invalid", v=0.1)


def test_ashrae_inputs_invalid_v_type() -> None:
    """Test that the function raises a TypeError for invalid v type."""
    with pytest.raises(TypeError):
        adaptive_ashrae(tdb=25, tr=25, t_running_mean=20, v="invalid")

    # Return nan values when limit_inputs=True and inputs are invalid


def test_nan_values_for_invalid_inputs() -> None:
    """Test that the function returns nan values for invalid inputs when limit_inputs=True."""
    # Test with invalid inputs where limit_inputs=True
    result = adaptive_ashrae(
        tdb=5.0,
        tr=5.0,
        t_running_mean=5.0,
        v=3.0,
        limit_inputs=True,
    )

    # Check that the comfort temperature is nan
    assert np.isnan(result.tmp_cmf)

    # Check that the acceptability flags are False
    assert result.acceptability_80 == False
    assert result.acceptability_90 == False
