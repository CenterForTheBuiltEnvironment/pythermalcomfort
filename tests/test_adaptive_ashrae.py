import pytest
from pythermalcomfort.models import adaptive_ashrae
from tests.conftest import Urls


def test_adaptive_ashrae(get_test_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_test_url(Urls.ADAPTIVE_ASHRAE.name))

    if reference_table is None:
        pytest.fail(
            f"Failed to retrieve reference table for {Urls.ADAPTIVE_ASHRAE.value.lower()}"
        )

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        units = inputs.get("units", "SI")
        result = adaptive_ashrae(
            inputs["tdb"], inputs["tr"], inputs["t_running_mean"], inputs["v"], units
        )
        for key in outputs:
            try:
                assert is_equal(result[key], outputs[key])
            except AssertionError as e:
                print(
                    f"Assertion failed for {Urls.ADAPTIVE_ASHRAE.value.lower()}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise
