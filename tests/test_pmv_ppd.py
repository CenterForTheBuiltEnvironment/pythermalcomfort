import json

import numpy as np
import pytest
import requests

from pythermalcomfort.models import (
    pmv_ppd,
)
from pythermalcomfort.models.pmv_ppd import _pmv_ppd_optimized

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


def test_pmv_ppd():
    for table in reference_tables["reference_data"]["pmv_ppd"]:
        for entry in table["data"]:
            standard = "ISO"
            if "ASHRAE" in table["source"]:
                standard = "ASHRAE"
            inputs = entry["inputs"]
            outputs = entry["outputs"]
            r = pmv_ppd(
                inputs["ta"],
                inputs["tr"],
                inputs["v"],
                inputs["rh"],
                inputs["met"],
                inputs["clo"],
                standard=standard,
            )
            # asserting with this strange code otherwise face issues with rounding fund
            assert float("%.1f" % r["pmv"]) == outputs["pmv"]
            assert np.round(r["ppd"], 1) == outputs["ppd"]

    for row in data_test_pmv_ip:
        assert (
            abs(
                round(
                    pmv_ppd(
                        row["tdb"],
                        row["tdb"],
                        row["vr"],
                        row["rh"],
                        row["met"],
                        row["clo"],
                        standard="ashrae",
                        units="ip",
                    )["pmv"],
                    1,
                )
                - row["pmv"]
            )
            < 0.011
        )
        assert (
            abs(
                round(
                    pmv_ppd(
                        row["tdb"],
                        row["tdb"],
                        row["vr"],
                        row["rh"],
                        row["met"],
                        row["clo"],
                        standard="ashrae",
                        units="ip",
                    )["ppd"],
                    1,
                )
                - row["ppd"]
            )
            < 1
        )

    assert (
        round(pmv_ppd(67.28, 67.28, 0.328084, 86, 1.1, 1, units="ip")["pmv"], 1)
    ) == -0.5

    np.testing.assert_equal(
        np.around(pmv_ppd([70, 70], 67.28, 0.328084, 86, 1.1, 1, units="ip")["pmv"], 1),
        [-0.3, -0.3],
    )

    # test airspeed limits
    np.testing.assert_equal(
        pmv_ppd(
            [26, 24, 22, 26, 24, 22],
            [26, 24, 22, 26, 24, 22],
            [0.9, 0.6, 0.3, 0.9, 0.6, 0.3],
            50,
            [1.1, 1.1, 1.1, 1.3, 1.3, 1.3],
            [0.5, 0.5, 0.5, 0.7, 0.7, 0.7],
            standard="ashrae",
            airspeed_control=False,
        )["pmv"],
        [np.nan, np.nan, np.nan, -0.14, -0.43, -0.57],
    )

    with pytest.raises(ValueError):
        pmv_ppd(25, 25, 0.1, 50, 1.1, 0.5, standard="random")

    # checking that returns np.nan when outside standard applicability limits
    np.testing.assert_equal(
        pmv_ppd(
            [31, 20, 20, 20, 20, 30],
            [20, 41, 20, 20, 20, 20],
            [0.1, 0.1, 2, 0.1, 0.1, 0.1],
            50,
            [1.1, 1.1, 1.1, 0.7, 1.1, 4.1],
            [0.5, 0.5, 0.5, 0.5, 2.1, 0.1],
            standard="iso",
        ),
        {
            "pmv": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "ppd": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        },
    )

    np.testing.assert_equal(
        pmv_ppd(
            [41, 20, 20, 20, 20, 39],
            [20, 41, 20, 20, 20, 39],
            [0.1, 0.1, 2.1, 0.1, 0.1, 0.1],
            50,
            [1.1, 1.1, 1.1, 0.7, 1.1, 3.9],
            [0.5, 0.5, 0.5, 0.5, 2.1, 1.9],
            standard="ashrae",
        ),
        {
            "pmv": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "ppd": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        },
    )

    # check results with limit_inputs disabled
    np.testing.assert_equal(
        pmv_ppd(31, 41, 2, 50, 0.7, 2.1, standard="iso", limit_inputs=False),
        {"pmv": 2.4, "ppd": 91.0},
    )

    np.testing.assert_equal(
        pmv_ppd(41, 41, 2, 50, 0.7, 2.1, standard="ashrae", limit_inputs=False),
        {"pmv": 4.48, "ppd": 100.0},
    )

    for table in reference_tables["reference_data"]["pmv_ppd"]:
        standard = "ISO"
        if "ASHRAE" in table["source"]:
            standard = "ASHRAE"
        tdb = np.array([d["inputs"]["ta"] for d in table["data"]])
        tr = np.array([d["inputs"]["tr"] for d in table["data"]])
        v = np.array([d["inputs"]["v"] for d in table["data"]])
        rh = np.array([d["inputs"]["rh"] for d in table["data"]])
        met = np.array([d["inputs"]["met"] for d in table["data"]])
        clo = np.array([d["inputs"]["clo"] for d in table["data"]])
        pmv_exp = np.array([d["outputs"]["pmv"] for d in table["data"]])
        ppd_exp = np.array([d["outputs"]["ppd"] for d in table["data"]])
        results = pmv_ppd(tdb, tr, v, rh, met, clo, standard=standard)
        pmv_r = [float("%.1f" % x) for x in results["pmv"]]

        np.testing.assert_equal(pmv_r, pmv_exp)
        np.testing.assert_equal(results["ppd"], ppd_exp)


def test_pmv_ppd_optimized():
    assert (round(_pmv_ppd_optimized(25, 25, 0.3, 50, 1.5, 0.7, 0), 2)) == 0.55

    np.testing.assert_equal(
        np.around(_pmv_ppd_optimized([25, 25], 25, 0.3, 50, 1.5, 0.7, 0), 2),
        [0.55, 0.55],
    )
