import numpy as np
from pythermalcomfort.models import (
    set_tmp,
)

def test_phs_url(get_set_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_set_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        inputs["round"] = True
        inputs["limit_inputs"] = False
        result = set_tmp(**inputs)
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result, outputs[key], tolerance.get(key, 1e-6))
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result}, inputs={inputs}\nError: {str(e)}"
                )
                raise



def test_set_npnan():
    np.testing.assert_equal(
        set_tmp(
            [41, 20, 20, 20, 20, 39],
            [20, 41, 20, 20, 20, 39],
            [0.1, 0.1, 2.1, 0.1, 0.1, 0.1],
            50,
            [1.1, 1.1, 1.1, 0.7, 1.1, 3.9],
            [0.5, 0.5, 0.5, 0.5, 2.1, 1.9],
        ),
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    )
