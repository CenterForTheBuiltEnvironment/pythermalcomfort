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


def test_humidex_masterson():
    # todo move this to shared test
    # I got these values from
    # https://publications.gc.ca/collections/collection_2018/eccc/En57-23-1-79-eng.pdf
    result = humidex(tdb=21, rh=100, model="masterson")
    assert result.humidex == 29.3
    assert result.discomfort == "Little or no discomfort"

    result = humidex(tdb=34, rh=100, model="masterson")
    assert result.humidex == 58.4

    result = humidex(tdb=43, rh=20, model="masterson")
    assert result.humidex == 47.1

    result = humidex(tdb=30, rh=30, model="masterson")
    assert result.humidex == 31.5

    result = humidex(tdb=31, rh=55, model="masterson")
    assert result.humidex == 39.3
