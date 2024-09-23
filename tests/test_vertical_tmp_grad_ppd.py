import pytest
from pythermalcomfort.models import vertical_tmp_grad_ppd


def test_vertical_tmp_grad_ppd(get_vertical_tmp_grad_ppd_url, retrieve_data, is_equal):
    test_data = retrieve_data(get_vertical_tmp_grad_ppd_url)
    tolerance = test_data["tolerance"]

    if test_data is None:
        pytest.skip("Failed to retrieve test data")

    for case in test_data.get("data", []):
        inputs = case["inputs"]
        expected_outputs = case["outputs"]
        result = vertical_tmp_grad_ppd(
            inputs["tdb"],
            inputs["tr"],
            inputs["v"],
            inputs["rh"],
            inputs["met"],
            inputs["clo"],
            inputs["delta_t"],
            units=inputs.get("units", "SI"),
        )

        for key, expected_value in expected_outputs.items():
            try:
                is_equal(result[key], expected_value, tolerance.get(key, 1e-6))
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {expected_value}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise

    # Test for ValueError
    with pytest.raises(ValueError):
        vertical_tmp_grad_ppd(25, 25, 0.3, 50, 1.2, 0.5, 7)
