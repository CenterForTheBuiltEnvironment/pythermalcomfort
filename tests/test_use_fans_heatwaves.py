from pythermalcomfort.models import use_fans_heatwaves
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_use_fans_heatwaves(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.USE_FANS_HEATWAVES.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = use_fans_heatwaves(**inputs)

        validate_result(result, outputs, tolerance)
