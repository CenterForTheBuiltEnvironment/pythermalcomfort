import pytest
from pythermalcomfort.models import vertical_tmp_grad_ppd


def test_vertical_tmp_grad_ppd(get_vertical_tmp_grad_ppd_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_vertical_tmp_grad_ppd_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = vertical_tmp_grad_ppd(**inputs)

        for key in outputs:
            try:
                assert is_equal(result[key], outputs[key], tolerance.get(key, 1e-6))
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise

    # Test for ValueError
    with pytest.raises(ValueError):
        vertical_tmp_grad_ppd(25, 25, 0.3, 50, 1.2, 0.5, 7)
