import pytest
import math

from pythermalcomfort.models import wbgt


def test_wbgt_with_url_cases(get_wbgt_url, retrieve_data, is_equal):

    reference_table = retrieve_data(get_wbgt_url)
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = wbgt(**inputs)
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result, outputs[key])
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise


# Test wbgt value error
def test_wbgt():
    with pytest.raises(ValueError):
        wbgt(twb=25, tg=32, with_solar_load=True)


#  Calculate WBGT with twb and tg set to None
def test_calculate_wbgt_with_twb_and_tg_set_to_none():
    with pytest.raises(TypeError):
        wbgt(None, None)
