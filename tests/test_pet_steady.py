import numpy as np
import pytest

from pythermalcomfort.models import pet_steady
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_pet_steady(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.PET_STEADY.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = pet_steady(**inputs)

        validate_result(result, outputs, tolerance)


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


@pytest.mark.parametrize("shape", ((10, 10), 10, (3, 3, 3)))
@pytest.mark.parametrize(("tdb", "tr", "rh", "v", "met", "clo", "exp"), PET_TEST_MATRIX)
def test_pet_array(shape, tdb, tr, rh, v, met, clo, exp):
    tdb_arr = list(np.full(shape, tdb))
    tr_arr = list(np.full(shape, tr))
    rh_arr = list(np.full(shape, rh))
    v_arr = list(np.full(shape, v))
    res = pet_steady(tdb=tdb_arr, tr=tr_arr, rh=rh_arr, v=v_arr, met=met, clo=clo).pet
    np.testing.assert_array_equal(actual=res, desired=np.full(shape, exp))
