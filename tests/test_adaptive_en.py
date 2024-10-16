from pythermalcomfort.models import adaptive_en
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_adaptive_en(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.ADAPTIVE_EN.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = adaptive_en(
            inputs["tdb"], inputs["tr"], inputs["t_running_mean"], inputs["v"]
        )

        validate_result(result, outputs, tolerance)
