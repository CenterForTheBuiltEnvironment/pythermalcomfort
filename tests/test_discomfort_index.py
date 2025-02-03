from pythermalcomfort.models import discomfort_index
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_discomfort_index(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.DISCOMFORT_INDEX.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = discomfort_index(**inputs)

        validate_result(result, outputs, tolerance)
