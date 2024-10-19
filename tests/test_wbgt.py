import pytest

from pythermalcomfort.models import wbgt
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_wbgt_with_url_cases(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.WBGT.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = wbgt(**inputs)

        validate_result(result, outputs, tolerance)


# Test wbgt value error
def test_wbgt():
    with pytest.raises(ValueError):
        wbgt(twb=25, tg=32, with_solar_load=True)


#  Calculate WBGT with twb and tg set to None
def test_calculate_wbgt_with_twb_and_tg_set_to_none():
    with pytest.raises(TypeError):
        wbgt(None, None)
