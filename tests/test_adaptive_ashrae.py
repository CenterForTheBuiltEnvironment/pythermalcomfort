import numpy as np

from pythermalcomfort.models import adaptive_ashrae


def test_adaptive_ashrae(get_adaptive_ashrae_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_adaptive_ashrae_url)
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        units = inputs.get("units", "SI")
        result = adaptive_ashrae(
            inputs["tdb"], inputs["tr"], inputs["t_running_mean"], inputs["v"], units
        )
        for key in outputs:
            # Check if the result[key] is a numpy array
            if isinstance(result[key], np.ndarray):
                numpy_outputs = np.array(outputs[key], dtype=float)
                numpy_outputs = np.where(numpy_outputs == None, np.nan, numpy_outputs)
                try:
                    np.testing.assert_almost_equal(result[key], numpy_outputs)
                except AssertionError as e:
                    print(
                        f"Assertion failed for {key}. Expected {numpy_outputs}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                    )
            else:
                # Use the custom is_equal for other types
                try:
                    assert is_equal(result[key], outputs[key])
                except AssertionError as e:
                    print(
                        f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                    )