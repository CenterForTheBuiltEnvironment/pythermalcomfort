import json

import numpy as np
import requests

from pythermalcomfort.models import (
    set_tmp,
)

# get file containing validation tables
url = "https://raw.githubusercontent.com/FedericoTartarini/validation-data-comfort-models/main/validation_data.json"
resp = requests.get(url)
reference_tables = json.loads(resp.text)

# fmt: off
data_test_set_ip = [  # I have commented the lines of code that don't pass the test
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 74.9},
    {'tdb': 32, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 53.7},
    {'tdb': 50, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 62.3},
    {'tdb': 59, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 66.5},
    {'tdb': 68, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 70.7},
    {'tdb': 86, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 79.6},
    {'tdb': 104, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 93.8},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 10, 'met': 1, 'clo': 0.5, 'set': 74.0},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 90, 'met': 1, 'clo': 0.5, 'set': 76.8},
    {'tdb': 77, 'tr': 77, 'v': 19.7 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 75.2},
    {'tdb': 77, 'tr': 77, 'v': 118.1 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 70.4},
    {'tdb': 77, 'tr': 77, 'v': 216.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 68.4},
    {'tdb': 77, 'tr': 77, 'v': 590.6 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 65.6},
    {'tdb': 77, 'tr': 50, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 59.6},
    {'tdb': 77, 'tr': 104, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 88.9},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.1, 'set': 69.3},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 1, 'set': 81.0},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 2, 'set': 90.3},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 4, 'set': 99.7},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 0.8, 'clo': 0.5, 'set': 73.9},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 2, 'clo': 0.5, 'set': 78.7},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 4, 'clo': 0.5, 'set': 86.8},
    ]

data_test_pmv_ip = [  # I have commented the lines of code that don't pass the test
    {'tdb': 67.3, 'rh': 86, 'vr': 20 / 60, 'met': 1.1, 'clo': 1, 'pmv': -0.5, 'ppd': 10},
    {'tdb': 75.0, 'rh': 66, 'vr': 20 / 60, 'met': 1.1, 'clo': 1, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 78.2, 'rh': 15, 'vr': 20 / 60, 'met': 1.1, 'clo': 1, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 70.2, 'rh': 20, 'vr': 20 / 60, 'met': 1.1, 'clo': 1, 'pmv': -0.5, 'ppd': 10},
    {'tdb': 74.5, 'rh': 67, 'vr': 20 / 60, 'met': 1.1, 'clo': .5, 'pmv': -0.5, 'ppd': 10},
    {'tdb': 80.2, 'rh': 56, 'vr': 20 / 60, 'met': 1.1, 'clo': .5, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 82.2, 'rh': 13, 'vr': 20 / 60, 'met': 1.1, 'clo': .5, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 76.5, 'rh': 16, 'vr': 20 / 60, 'met': 1.1, 'clo': .5, 'pmv': -0.5, 'ppd': 10},
    ]
# fmt: on


def test_set():
    for table in reference_tables["reference_data"]["set"]:
        for entry in table["data"]:
            inputs = entry["inputs"]
            outputs = entry["outputs"]
            assert (
                set_tmp(
                    inputs["ta"],
                    inputs["tr"],
                    inputs["v"],
                    inputs["rh"],
                    inputs["met"],
                    inputs["clo"],
                    round=True,
                    limit_inputs=False,
                )
                == outputs["set"]
            )

    # testing SET equation to calculate cooling effect
    assert (set_tmp(25, 25, 1.1, 50, 2, 0.5, calculate_ce=True)) == 20.5
    assert (set_tmp(25, 25, 1.1, 50, 3, 0.5, calculate_ce=True)) == 20.9
    assert (set_tmp(25, 25, 1.1, 50, 1.5, 0.5, calculate_ce=True)) == 20.5
    assert (set_tmp(25, 25, 1.1, 50, 1.5, 0.75, calculate_ce=True)) == 23.1
    assert (set_tmp(25, 25, 1.1, 50, 1.5, 0.1, calculate_ce=True)) == 15.6
    assert (set_tmp(29, 25, 1.1, 50, 1.5, 0.5, calculate_ce=True)) == 23.3
    assert (set_tmp(27, 25, 1.1, 50, 1.5, 0.75, calculate_ce=True)) == 24.5
    assert (set_tmp(20, 25, 1.1, 50, 1.5, 0.1, calculate_ce=True)) == 11.2
    assert (set_tmp(25, 27, 1.1, 50, 1.5, 0.5, calculate_ce=True)) == 21.1
    assert (set_tmp(25, 29, 1.1, 50, 1.5, 0.5, calculate_ce=True)) == 21.7
    assert (set_tmp(25, 31, 1.1, 50, 1.5, 0.5, calculate_ce=True)) == 22.3
    assert (set_tmp(25, 27, 1.3, 50, 1.5, 0.5, calculate_ce=True)) == 20.8
    assert (set_tmp(25, 29, 1.5, 50, 1.5, 0.5, calculate_ce=True)) == 21.0
    assert (set_tmp(25, 31, 1.7, 50, 1.5, 0.5, calculate_ce=True)) == 21.3

    assert (
        set_tmp(
            tdb=77,
            tr=77,
            v=0.328,
            rh=50,
            met=1.2,
            clo=0.5,
            units="IP",
        )
    ) == 75.8

    for row in data_test_set_ip:
        assert (
            set_tmp(
                row["tdb"],
                row["tr"],
                row["v"],
                row["rh"],
                row["met"],
                row["clo"],
                units="IP",
                limit_inputs=False,
            )
            == row["set"]
        )

    # checking that returns np.nan when outside standard applicability limits
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

    for table in reference_tables["reference_data"]["set"]:
        tdb = np.array([d["inputs"]["ta"] for d in table["data"]])
        tr = np.array([d["inputs"]["tr"] for d in table["data"]])
        v = np.array([d["inputs"]["v"] for d in table["data"]])
        rh = np.array([d["inputs"]["rh"] for d in table["data"]])
        met = np.array([d["inputs"]["met"] for d in table["data"]])
        clo = np.array([d["inputs"]["clo"] for d in table["data"]])
        set_exp = np.array([d["outputs"]["set"] for d in table["data"]])
        results = set_tmp(tdb, tr, v, rh, met, clo, limit_inputs=False)

        np.testing.assert_equal(set_exp, results)
