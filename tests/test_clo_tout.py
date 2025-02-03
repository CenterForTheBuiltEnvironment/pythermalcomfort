from pythermalcomfort.models import clo_tout
from pythermalcomfort.utilities import Units
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_clo_tout(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.CLO_TOUT.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = clo_tout(
            tout=inputs["tout"], units=inputs.get("units", Units.SI.value)
        )

        validate_result(result, outputs, tolerance)
