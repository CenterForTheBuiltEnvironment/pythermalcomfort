import numpy as np
from pythermalcomfort.models import adaptive_en


def test_adaptive_en(get_adaptive_en_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_adaptive_en_url)
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = adaptive_en(
            inputs["tdb"], inputs["tr"], inputs["t_running_mean"], inputs["v"]
        )
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result[key], outputs[key])
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise
