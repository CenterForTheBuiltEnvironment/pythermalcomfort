import numpy as np

from pythermalcomfort.models import (
    utci,
)
from pythermalcomfort.models.utci import _utci_optimized

def test_utci(get_utci_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_utci_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = utci(**inputs)
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                if isinstance(result, dict):
                    assert is_equal(result[key], outputs[key], tolerance.get(key, 1e-6))
                else:
                    assert is_equal(result, outputs[key], tolerance.get(key, 1e-6))
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result}, inputs={inputs}\nError: {str(e)}"
                )
                raise

def test_utci_optimized():
    np.testing.assert_equal(
        np.around(_utci_optimized([25, 27], 1, 1, 1.5), 2), [24.73, 26.57]
    )
