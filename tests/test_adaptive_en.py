import pytest

from pythermalcomfort.models import adaptive_en
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_adaptive_en(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the Adaptive Thermal Comfort Model (ASHRAE 55) correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.ADAPTIVE_EN.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = adaptive_en(
            inputs["tdb"],
            inputs["tr"],
            inputs["t_running_mean"],
            inputs["v"],
        )

        validate_result(result, outputs, tolerance)


def test_ashrae_inputs_invalid_units() -> None:
    """Test that the function raises a ValueError for invalid units."""
    with pytest.raises(ValueError):
        adaptive_en(tdb=25, tr=25, t_running_mean=20, v=0.1, units="INVALID")


def test_ashrae_inputs_invalid_tdb_type() -> None:
    """Test that the function raises a TypeError for invalid tdb type."""
    with pytest.raises(TypeError):
        adaptive_en(tdb="invalid", tr=25, t_running_mean=20, v=0.1)


def test_ashrae_inputs_invalid_tr_type() -> None:
    """Test that the function raises a TypeError for invalid tr type."""
    with pytest.raises(TypeError):
        adaptive_en(tdb=25, tr="invalid", t_running_mean=20, v=0.1)


def test_ashrae_inputs_invalid_t_running_mean_type() -> None:
    """Test that the function raises a TypeError for invalid t_running_mean type."""
    with pytest.raises(TypeError):
        adaptive_en(tdb=25, tr=25, t_running_mean="invalid", v=0.1)


def test_ashrae_inputs_invalid_v_type() -> None:
    """Test that the function raises a TypeError for invalid v type."""
    with pytest.raises(TypeError):
        adaptive_en(tdb=25, tr=25, t_running_mean=20, v="invalid")
