import numpy as np

from pythermalcomfort.models import discomfort_index


def test_discomfort_index(get_discomfort_index_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_discomfort_index_url)
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = discomfort_index(**inputs)
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result[key], outputs[key])
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise
