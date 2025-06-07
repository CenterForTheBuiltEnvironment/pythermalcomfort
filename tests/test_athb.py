from pythermalcomfort.models import pmv_athb
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_athb(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the Adaptive Thermal Comfort Model (ASHRAE 55) correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.ATHB.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]

        result = pmv_athb(**inputs)

        validate_result(result, outputs, tolerance)
