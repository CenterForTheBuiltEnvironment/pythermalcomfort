from pythermalcomfort.models import adaptive_ashrae
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_adaptive_ashrae(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.ADAPTIVE_ASHRAE.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        units = inputs.get("units", "SI")
        result = adaptive_ashrae(
            inputs["tdb"], inputs["tr"], inputs["t_running_mean"], inputs["v"], units
        )

        validate_result(result, outputs, tolerance)
