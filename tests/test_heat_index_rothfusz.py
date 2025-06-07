import numpy as np
import pytest

from pythermalcomfort.models import heat_index_rothfusz
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_heat_index(get_test_url, retrieve_data) -> None:
    """Test the heat index function with various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.HEAT_INDEX.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = heat_index_rothfusz(**inputs, limit_inputs=False)

        validate_result(result, outputs, tolerance)


def test_single_input_caution() -> None:
    """Test that the function returns a single value with caution category."""
    result = heat_index_rothfusz(tdb=29, rh=50, round_output=True)
    assert result.hi.shape == ()  # zero-dim ndarray
    assert result.hi.item() == pytest.approx(29.7, rel=1e-3)
    assert result.stress_category == "caution"


def test_below_threshold_produces_nan() -> None:
    """Test that the function returns NaN for heat index below threshold."""
    result = heat_index_rothfusz(tdb=25, rh=80)
    assert np.isnan(result.hi).item()
    assert isinstance(result.stress_category, np.str_)


def test_vector_input_no_rounding() -> None:
    """Test that the function handles vector inputs correctly without rounding."""
    tdb = [30.0, 20.0, 28.5]
    rh = [70.0, 90.0, 50.0]
    result = heat_index_rothfusz(tdb=tdb, rh=rh, round_output=False)
    hi = result.hi
    # Shape matches inputs
    assert hi.shape == (3,)
    # First element has multiple decimals and matches expected formula
    assert hi[0] == pytest.approx(35.33, rel=1e-2)
    # Second element below threshold ⇒ NaN
    assert np.isnan(hi[1])
    # Third element above threshold ⇒ finite number
    assert np.isfinite(hi[2])
    # Stress categories array matches hi length
    assert isinstance(result.stress_category, np.ndarray)
    assert result.stress_category.shape == (3,)
    assert result.stress_category[0] == "extreme caution"
