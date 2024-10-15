import pytest
from pythermalcomfort.models import a_pmv
from tests.conftest import Urls


def test_a_pmv(get_test_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_test_url(Urls.A_PMV.name))

    if reference_table is None:
        pytest.fail(
            f"Failed to retrieve reference table for {Urls.A_PMV.value.lower()}"
        )

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        expected_output = entry["outputs"]["a_pmv"]
        result = a_pmv(
            inputs["tdb"],
            inputs["tr"],
            inputs["vr"],
            inputs["rh"],
            inputs["met"],
            inputs["clo"],
            inputs["a_coefficient"],
        )
        try:
            assert is_equal(result, expected_output)
        except AssertionError as e:
            print(
                f"Assertion failed for {Urls.A_PMV.value.lower()}. Expected {expected_output}, got {result}, inputs={inputs}\nError: {str(e)}"
            )
            raise
