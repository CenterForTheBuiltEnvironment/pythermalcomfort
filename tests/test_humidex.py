import pytest

from pythermalcomfort.models import humidex
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_humidex(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.HUMIDEX.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = humidex(**inputs)

        validate_result(result, outputs, tolerance)

    with pytest.raises(TypeError):
        humidex("25", 50)

    with pytest.raises(TypeError):
        humidex(25, "50")

    with pytest.raises(ValueError):
        humidex(tdb=25, rh=110)

    with pytest.raises(ValueError):
        humidex(25, -10)
