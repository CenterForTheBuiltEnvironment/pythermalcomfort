from pythermalcomfort.models import cooling_effect
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_cooling_effect(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.COOLING_EFFECT.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = cooling_effect(**inputs)

        validate_result(result, outputs, tolerance)
