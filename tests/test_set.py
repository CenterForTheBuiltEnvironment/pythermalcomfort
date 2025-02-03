import numpy as np

from pythermalcomfort.models import (
    set_tmp,
)
from tests.conftest import Urls, retrieve_reference_table, validate_result

tdb = []
tr = []
v = []
rh = []
met = []
clo = []
set_exp = []


def test_set_url(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.SET.name
    )
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        inputs["round_output"] = True
        inputs["limit_inputs"] = False
        result = set_tmp(**inputs)

        validate_result(result, outputs, tolerance)


def test_set_npnan():
    np.testing.assert_equal(
        set_tmp(
            [41, 20, 20, 20, 20, 39],
            [20, 41, 20, 20, 20, 39],
            [0.1, 0.1, 2.1, 0.1, 0.1, 0.1],
            50,
            [1.1, 1.1, 1.1, 0.7, 1.1, 3.9],
            [0.5, 0.5, 0.5, 0.5, 2.1, 1.9],
        ).set,
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    )
