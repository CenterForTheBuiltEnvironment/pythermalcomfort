import pytest

from pythermalcomfort.models import wci
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_calculates_wci(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the wind chill index (WCI) correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.WIND_CHILL.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = wci(**inputs)

        validate_result(result, outputs, tolerance)


class TestWc:
    """Test cases for the wind chill index (WCI) model."""

    #  Raises TypeError if tdb parameter is not provided
    def test_raises_type_error_tdb_not_provided(self) -> None:
        """Test that the function raises TypeError if tdb is not provided."""
        with pytest.raises(TypeError):
            wci(v=0.1)

    #  Raises TypeError if v parameter is not provided
    def test_raises_type_error_v_not_provided(self) -> None:
        """Test that the function raises TypeError if v is not provided."""
        with pytest.raises(TypeError):
            wci(tdb=0)

    #  Raises TypeError if tdb parameter is not a float
    def test_raises_type_error_tdb_not_float(self) -> None:
        """Test that the function raises TypeError if tdb is not a float."""
        with pytest.raises(TypeError):
            wci(tdb="0", v=0.1)
