import numpy as np

from pythermalcomfort.models import adaptive_ashrae


def test_adaptive_ashrae():
    data_test_adaptive_ashrae = (
        [  # I have commented the lines of code that don't pass the test
            {
                "tdb": 19.6,
                "tr": 19.6,
                "t_running_mean": 17,
                "v": 0.1,
                "return": {"acceptability_80": True},
            },
            {
                "tdb": 19.6,
                "tr": 19.6,
                "t_running_mean": 17,
                "v": 0.1,
                "return": {"acceptability_90": False},
            },
            {
                "tdb": 19.6,
                "tr": 19.6,
                "t_running_mean": 25,
                "v": 0.1,
                "return": {"acceptability_80": False},
            },
            {
                "tdb": 19.6,
                "tr": 19.6,
                "t_running_mean": 25,
                "v": 0.1,
                "return": {"acceptability_80": False},
            },
            {
                "tdb": 26,
                "tr": 26,
                "t_running_mean": 16,
                "v": 0.1,
                "return": {"acceptability_80": True},
            },
            {
                "tdb": 26,
                "tr": 26,
                "t_running_mean": 16,
                "v": 0.1,
                "return": {"acceptability_90": False},
            },
            {
                "tdb": 30,
                "tr": 26,
                "t_running_mean": 16,
                "v": 0.1,
                "return": {"acceptability_80": False},
            },
            {
                "tdb": 25,
                "tr": 25,
                "t_running_mean": 23,
                "v": 0.1,
                "return": {"acceptability_80": True},
            },
            {
                "tdb": 25,
                "tr": 25,
                "t_running_mean": 23,
                "v": 0.1,
                "return": {"acceptability_90": True},
            },
        ]
    )
    for row in data_test_adaptive_ashrae:
        assert (
            adaptive_ashrae(row["tdb"], row["tr"], row["t_running_mean"], row["v"])[
                list(row["return"].keys())[0]
            ]
        ) == row["return"][list(row["return"].keys())[0]]

    assert (adaptive_ashrae(77, 77, 68, 0.3, units="ip")["tmp_cmf"]) == 75.2

    # test limit_inputs and array input
    np.testing.assert_equal(
        adaptive_ashrae(tdb=25, tr=25, t_running_mean=[9, 10], v=0.1).__dict__,
        {
            "tmp_cmf": [np.nan, 20.9],
            "tmp_cmf_80_low": [np.nan, 17.4],
            "tmp_cmf_80_up": [np.nan, 24.4],
            "tmp_cmf_90_low": [np.nan, 18.4],
            "tmp_cmf_90_up": [np.nan, 23.4],
            "acceptability_80": [False, False],
            "acceptability_90": [False, False],
        },
    )
    np.testing.assert_equal(
        adaptive_ashrae(
            tdb=[77, 74], tr=77, t_running_mean=[48, 68], v=0.3, units="ip"
        ).__dict__,
        {
            "tmp_cmf": [np.nan, 75.2],
            "tmp_cmf_80_low": [np.nan, 68.9],
            "tmp_cmf_80_up": [np.nan, 81.5],
            "tmp_cmf_90_low": [np.nan, 70.7],
            "tmp_cmf_90_up": [np.nan, 79.7],
            "acceptability_80": [False, True],
            "acceptability_90": [False, True],
        },
    )
