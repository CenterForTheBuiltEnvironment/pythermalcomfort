import numpy as np
from pythermalcomfort.models import e_pmv


def test_e_pmv(get_e_pmv_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_e_pmv_url)

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        expected_outputs = entry["outputs"]["pmv"]

        result = e_pmv(
            tdb=inputs["tdb"],
            tr=inputs["tr"],
            vr=inputs["vr"],
            rh=inputs["rh"],
            met=inputs["met"],
            clo=inputs["clo"],
            e_coefficient=inputs["e_coefficient"],
        )

        try:
            if isinstance(expected_outputs, list):
                np.testing.assert_equal(result, expected_outputs)
            else:
                assert is_equal(result, expected_outputs)
        except AssertionError as e:
            print(
                f"Assertion failed for e_pmv. Expected {expected_outputs}, got {result}, inputs={inputs}\nError: {str(e)}"
            )
            raise
