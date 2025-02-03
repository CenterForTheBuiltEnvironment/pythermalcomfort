import pytest

from pythermalcomfort.models import wci
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_calculates_wci(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.WIND_CHILL.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = wci(**inputs)

        validate_result(result, outputs, tolerance)


class TestWc:
    #  Raises TypeError if tdb parameter is not provided
    def test_raises_type_error_tdb_not_provided(self):
        with pytest.raises(TypeError):
            wci(v=0.1)

    #  Raises TypeError if v parameter is not provided
    def test_raises_type_error_v_not_provided(self):
        with pytest.raises(TypeError):
            wci(tdb=0)

    #  Raises TypeError if tdb parameter is not a float
    def test_raises_type_error_tdb_not_float(self):
        with pytest.raises(TypeError):
            wci(tdb="0", v=0.1)
