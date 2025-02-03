from pythermalcomfort.models import net
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_net(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.NET.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = net(**inputs)

        validate_result(result, outputs, tolerance)
