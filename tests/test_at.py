from pythermalcomfort.models import at
from tests.conftest import Urls, is_equal, retrieve_reference_table, validate_result


def test_at(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.AT.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = at(**inputs)

        validate_result(result, outputs, tolerance)


def test_at_list_input():
    result = at([25, 25, 25], [30, 30, 30], [0.1, 0.1, 0.1])
    is_equal(result, [24.1, 24.1, 24.1], 0.1)


def test_at_q():
    result = at(25, 30, 0.1, 100)
    is_equal(result, 25.3, 0.1)
