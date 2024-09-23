import numpy as np

from pythermalcomfort.models import adaptive_ashrae


def test_adaptive_ashrae(get_adaptive_ashrae_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_adaptive_ashrae_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        units = inputs.get("units", "SI")
        result = adaptive_ashrae(
            inputs["tdb"], inputs["tr"], inputs["t_running_mean"], inputs["v"], units
        )
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result[key], outputs[key], tolerance.get(key, 1e-6))
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise
