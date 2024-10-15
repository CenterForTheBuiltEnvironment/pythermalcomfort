import numpy as np

from pythermalcomfort.models import (
    pmv,
)


def test_pmv(get_pmv_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_pmv_url)
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = pmv(**inputs)
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result, outputs[key])
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise
