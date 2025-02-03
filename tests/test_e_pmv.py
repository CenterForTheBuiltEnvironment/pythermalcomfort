from pythermalcomfort.models import pmv_e
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_e_pmv(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.E_PMV.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = pmv_e(**inputs)

        validate_result(result, outputs, tolerance)
