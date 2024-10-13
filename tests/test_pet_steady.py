import numpy as np
import pytest
from pythermalcomfort.models import pet_steady


def test_pet_steady(get_pet_steady_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_pet_steady_url)

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        expected_output = entry["outputs"]["PET"]

        result = pet_steady(
            tdb=inputs["tdb"],
            tr=inputs["tr"],
            rh=inputs["rh"],
            v=inputs["v"],
            met=inputs["met"],
            clo=inputs["clo"],
        )

        try:
            if isinstance(expected_output, list):
                np.testing.assert_equal(result, expected_output)
            else:
                assert np.isclose(result, expected_output, atol=0.1)
        except AssertionError as e:
            print(
                f"Assertion failed for pet_steady. Expected {expected_output}, got {result}, inputs={inputs}\nError: {str(e)}"
            )
            raise

PET_TEST_MATRIX = (
    # 'tdb', 'tr', 'rh', 'v', 'met', 'clo', 'exp'
    (20, 20, 50, 0.15, 1.37, 0.5, 18.85),
    (30, 30, 50, 0.15, 1.37, 0.5, 30.6),
    (20, 20, 50, 0.5, 1.37, 0.5, 17.16),
    (21, 21, 50, 0.1, 1.37, 0.9, 21.08),
    (20, 20, 50, 0.1, 1.37, 0.9, 19.92),
    (-5, 40, 2, 0.5, 1.37, 0.9, 7.82),
    (-5, -5, 50, 5.0, 1.37, 0.9, -13.38),
    (30, 60, 80, 1.0, 1.37, 0.9, 44.63),
    (30, 30, 80, 1.0, 1.37, 0.9, 32.21),
)

@pytest.mark.parametrize('shape', ((10, 10), 10, (3, 3, 3)))
@pytest.mark.parametrize(('tdb', 'tr', 'rh', 'v', 'met', 'clo', 'exp'), PET_TEST_MATRIX)
def test_pet_array(shape, tdb, tr, rh, v, met, clo, exp):
    tdb_arr=np.full(shape, tdb)
    tr_arr=np.full(shape, tr)
    rh_arr=np.full(shape, rh)
    v_arr=np.full(shape, v)
    res = pet_steady(tdb=tdb_arr, tr=tr_arr, rh=rh_arr, v=v_arr, met=met, clo=clo)
    np.testing.assert_array_equal(actual=res, desired=np.full(shape, exp))
