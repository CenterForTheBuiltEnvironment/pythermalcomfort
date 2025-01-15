from pythermalcomfort.models import heat_index_rothfusz
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_heat_index(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.HEAT_INDEX.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = heat_index_rothfusz(**inputs)

        validate_result(result, outputs, tolerance)
