from pythermalcomfort.models import solar_gain
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_solar_gain(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.SOLAR_GAIN.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = solar_gain(**inputs)

        validate_result(result, outputs, tolerance)
