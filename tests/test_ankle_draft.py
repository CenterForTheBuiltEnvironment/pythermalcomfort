import pytest
from pythermalcomfort.models import ankle_draft


def test_ankle_draft(get_ankle_draft_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_ankle_draft_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = ankle_draft(**inputs)
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result[key], outputs[key], tolerance.get(key, 1e-6))
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise

    # Test for ValueError
    with pytest.raises(ValueError):
        ankle_draft(25, 25, 0.3, 50, 1.2, 0.5, 7)
