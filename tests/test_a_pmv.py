from pythermalcomfort.models import a_pmv
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_a_pmv(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.A_PMV.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = a_pmv(**inputs)

        validate_result(result, outputs, tolerance)
