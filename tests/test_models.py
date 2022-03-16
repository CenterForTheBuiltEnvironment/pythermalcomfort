import pytest
import numpy as np
import json
import warnings
import requests

from pythermalcomfort.models import (
    solar_gain,
    pmv_ppd,
    set_tmp,
    cooling_effect,
    adaptive_ashrae,
    clo_tout,
    vertical_tmp_grad_ppd,
    utci,
    pmv,
    ankle_draft,
    phs,
    use_fans_heatwaves,
    wbgt,
    heat_index,
    humidex,
    two_nodes,
    net,
    at,
    wc,
)
from pythermalcomfort.psychrometrics import (
    t_dp,
    t_wb,
    enthalpy,
    psy_ta_rh,
    p_sat,
    t_mrt,
    t_o,
)
from pythermalcomfort.utilities import (
    transpose_sharp_altitude,
    f_svv,
    clo_dynamic,
    running_mean_outdoor_temperature,
    units_converter,
    body_surface_area,
)

# get file containing validation tables
url = "https://raw.githubusercontent.com/FedericoTartarini/validation-data-comfort-models/main/validation_data.json"
resp = requests.get(url)
reference_tables = json.loads(resp.text)

# fmt: off
data_test_set_ip = [  # I have commented the lines of code that don't pass the test
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 74.9},
    {'tdb': 59, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 66.5},
    {'tdb': 68, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 70.7},
    {'tdb': 86, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 79.6},
    {'tdb': 104, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 93.6},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 10, 'met': 1, 'clo': 0.5, 'set': 74.0},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 90, 'met': 1, 'clo': 0.5, 'set': 76.8},
    {'tdb': 77, 'tr': 77, 'v': 19.7 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 75.2},
    {'tdb': 77, 'tr': 77, 'v': 118.1 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 70.4},
    {'tdb': 77, 'tr': 77, 'v': 216.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 68.4},
    {'tdb': 77, 'tr': 77, 'v': 590.6 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 65.6},
    {'tdb': 77, 'tr': 50, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 59.6},
    {'tdb': 77, 'tr': 104, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 88.9},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 1, 'set': 81.0},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 2, 'set': 90.4},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 4, 'set': 100.0},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 0.8, 'clo': 0.5, 'set': 73.9},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.1, 'set': 69.3},
    {'tdb': 50, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 62.3},
    {'tdb': 32, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 53.7},
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
            assert round(r["pmv"], 1) == outputs["pmv"]
            assert round(r["ppd"], 1) == outputs["ppd"]

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

    with pytest.raises(ValueError):
        pmv_ppd(25, 25, 0.1, 50, 1.1, 0.5, standard="random")


def test_pmv():
    for table in reference_tables["reference_data"]["pmv_ppd"]:
        for entry in table["data"]:
            standard = "ISO"
            if "ASHRAE" in table["source"]:
                standard = "ASHRAE"
            inputs = entry["inputs"]
            outputs = entry["outputs"]
            assert (
                round(
                    pmv(
                        inputs["ta"],
                        inputs["tr"],
                        inputs["v"],
                        inputs["rh"],
                        inputs["met"],
                        inputs["clo"],
                        standard=standard,
                    ),
                    1,
                )
                == outputs["pmv"]
            )


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
    assert (set_tmp(25, 29, 1.5, 50, 1.5, 0.5, calculate_ce=True)) == 21.1
    assert (set_tmp(25, 31, 1.7, 50, 1.5, 0.5, calculate_ce=True)) == 21.4

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
            abs(
                set_tmp(
                    row["tdb"],
                    row["tr"],
                    row["v"],
                    row["rh"],
                    row["met"],
                    row["clo"],
                    units="IP",
                )
                - row["set"]
            )
            < 0.11
        )


def test_solar_gain():
    for table in reference_tables["reference_data"]["solar_gain"]:
        for entry in table["data"]:
            inputs = entry["inputs"]
            outputs = entry["outputs"]
            sg = solar_gain(
                inputs["alt"],
                inputs["sharp"],
                inputs["I_dir"],
                inputs["t_sol"],
                inputs["f_svv"],
                inputs["f_bes"],
                inputs["asa"],
                inputs["posture"],
            )
            assert sg["erf"] == outputs["erf"]
            assert sg["delta_mrt"] == outputs["t_rsw"]


def test_transpose_sharp_altitude():
    assert transpose_sharp_altitude(sharp=0, altitude=0) == (0, 90)
    assert transpose_sharp_altitude(sharp=0, altitude=20) == (0, 70)
    assert transpose_sharp_altitude(sharp=0, altitude=45) == (0, 45)
    assert transpose_sharp_altitude(sharp=0, altitude=60) == (0, 30)
    assert transpose_sharp_altitude(sharp=90, altitude=0) == (90, 0)
    assert transpose_sharp_altitude(sharp=90, altitude=45) == (45, 0)
    assert transpose_sharp_altitude(sharp=90, altitude=30) == (60, 0)
    assert transpose_sharp_altitude(sharp=135, altitude=60) == (22.208, 20.705)
    assert transpose_sharp_altitude(sharp=120, altitude=75) == (13.064, 7.435)
    assert transpose_sharp_altitude(sharp=150, altitude=30) == (40.893, 48.590)


def test_use_fans_heatwaves():
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=0.7, clo=0.3, body_position="sitting"
        )["heat_strain_w"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=0.7, clo=0.5, body_position="sitting"
        )["q_skin"]
        == 37.7
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=0.7, clo=0.7, body_position="sitting"
        )["m_rsw"]
        == 67.6
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=1.3, clo=0.3, body_position="sitting"
        )["m_rsw"]
        == 115.9
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=1.3, clo=0.5, body_position="sitting"
        )["m_rsw"]
        == 115.1
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=1.3, clo=0.7, body_position="sitting"
        )["m_rsw"]
        == 114.4
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=2, clo=0.3, body_position="sitting"
        )["heat_strain_w"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=2, clo=0.5, body_position="sitting"
        )["w"]
        == 0.5
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=2, clo=0.7, body_position="sitting"
        )["t_skin"]
        == 36.3
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=0.7, clo=0.3, body_position="sitting"
        )["heat_strain_blood_flow"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=0.7, clo=0.5, body_position="sitting"
        )["t_core"]
        == 36.9
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=0.7, clo=0.7, body_position="sitting"
        )["m_rsw"]
        == 72.8
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=1.3, clo=0.3, body_position="sitting"
        )["m_rsw"]
        == 124.2
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=1.3, clo=0.5, body_position="sitting"
        )["e_rsw"]
        == 83.3
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=1.3, clo=0.7, body_position="sitting"
        )["e_rsw"]
        == 82.1
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=2, clo=0.3, body_position="sitting"
        )["q_res"]
        == 6.2
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=2, clo=0.5, body_position="sitting"
        )["w"]
        == 0.7
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=2, clo=0.7, body_position="sitting"
        )["w_max"]
        == 0.7
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=20, met=0.7, clo=0.3, body_position="sitting"
        )["e_diff"]
        == 17.9
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=20, met=0.7, clo=0.5, body_position="sitting"
        )["heat_strain_sweating"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=20, met=0.7, clo=0.7, body_position="sitting"
        )["t_skin"]
        == 35.8
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=20, met=1.3, clo=0.3, body_position="sitting"
        )["heat_strain_blood_flow"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=20, met=1.3, clo=0.5, body_position="sitting"
        )["heat_strain"]
        == 0
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=20, met=1.3, clo=0.7, body_position="sitting"
        )["w"]
        == 0.4
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=20, met=2, clo=0.3, body_position="sitting"
        )["q_res"]
        == 9.0
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=20, met=2, clo=0.5, body_position="sitting"
        )["e_skin"]
        == 124.9
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=20, met=2, clo=0.7, body_position="sitting"
        )["e_diff"]
        == 6.3
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=40, met=0.7, clo=0.3, body_position="sitting"
        )["w_max"]
        == 0.6
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=40, met=0.7, clo=0.5, body_position="sitting"
        )["m_rsw"]
        == 75.5
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=40, met=0.7, clo=0.7, body_position="sitting"
        )["e_skin"]
        == 56.1
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=40, met=1.3, clo=0.3, body_position="sitting"
        )["e_rsw"]
        == 84.6
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=40, met=1.3, clo=0.5, body_position="sitting"
        )["e_diff"]
        == 6.4
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=40, met=1.3, clo=0.7, body_position="sitting"
        )["w_max"]
        == 0.6
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=40, met=2, clo=0.3, body_position="sitting"
        )["w"]
        == 0.5
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=40, met=2, clo=0.5, body_position="sitting"
        )["e_max"]
        == 190.7
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=1, rh=40, met=2, clo=0.7, body_position="sitting"
        )["e_skin"]
        == 104.3
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=20, met=0.7, clo=0.3, body_position="sitting"
        )["e_max"]
        == 473.2
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=20, met=0.7, clo=0.5, body_position="sitting"
        )["e_skin"]
        == 64.3
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=20, met=0.7, clo=0.7, body_position="sitting"
        )["t_skin"]
        == 35.8
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=20, met=1.3, clo=0.3, body_position="sitting"
        )["q_sensible"]
        == -32.0
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=20, met=1.3, clo=0.5, body_position="sitting"
        )["q_res"]
        == 5.8
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=20, met=1.3, clo=0.7, body_position="sitting"
        )["w"]
        == 0.3
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=20, met=2, clo=0.3, body_position="sitting"
        )["q_skin"]
        == 106.5
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=20, met=2, clo=0.5, body_position="sitting"
        )["e_skin"]
        == 129.5
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=20, met=2, clo=0.7, body_position="sitting"
        )["e_diff"]
        == 8.7
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=40, met=0.7, clo=0.3, body_position="sitting"
        )["e_skin"]
        == 71.9
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=40, met=0.7, clo=0.5, body_position="sitting"
        )["q_res"]
        == 2.2
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=40, met=0.7, clo=0.7, body_position="sitting"
        )["e_skin"]
        == 59.1
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=40, met=1.3, clo=0.3, body_position="sitting"
        )["q_skin"]
        == 71.6
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=40, met=1.3, clo=0.5, body_position="sitting"
        )["w_max"]
        == 0.5
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=40, met=1.3, clo=0.7, body_position="sitting"
        )["e_diff"]
        == 5.9
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=40, met=2, clo=0.3, body_position="sitting"
        )["e_rsw"]
        == 126.4
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=40, met=2, clo=0.5, body_position="sitting"
        )["e_max"]
        == 236.1
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=4, rh=40, met=2, clo=0.7, body_position="sitting"
        )["w"]
        == 0.5
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=20, met=0.7, clo=0.3, body_position="sitting"
        )["heat_strain_sweating"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=20, met=0.7, clo=0.5, body_position="sitting"
        )["q_res"]
        == 2.8
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=20, met=0.7, clo=0.7, body_position="sitting"
        )["heat_strain_blood_flow"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=20, met=1.3, clo=0.3, body_position="sitting"
        )["t_skin"]
        == 36.6
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=20, met=1.3, clo=0.5, body_position="sitting"
        )["t_core"]
        == 37.3
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=20, met=1.3, clo=0.7, body_position="sitting"
        )["m_bl"]
        == 80.0
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=20, met=2, clo=0.3, body_position="sitting"
        )["e_rsw"]
        == 163.6
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=20, met=2, clo=0.5, body_position="sitting"
        )["m_bl"]
        == 80.0
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=20, met=2, clo=0.7, body_position="sitting"
        )["q_skin"]
        == 89.6
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=40, met=0.7, clo=0.3, body_position="sitting"
        )["t_core"]
        == 37.2
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=40, met=0.7, clo=0.5, body_position="sitting"
        )["heat_strain_sweating"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=40, met=0.7, clo=0.7, body_position="sitting"
        )["m_bl"]
        == 80.0
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=40, met=1.3, clo=0.3, body_position="sitting"
        )["w"]
        == 0.7
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=40, met=1.3, clo=0.5, body_position="sitting"
        )["heat_strain_sweating"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=40, met=1.3, clo=0.7, body_position="sitting"
        )["t_core"]
        == 38.0
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=40, met=2, clo=0.3, body_position="sitting"
        )["heat_strain"]
        == 1
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=40, met=2, clo=0.5, body_position="sitting"
        )["q_sensible"]
        == -42.9
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=0.2, rh=40, met=2, clo=0.7, body_position="sitting"
        )["m_bl"]
        == 80.0
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=20, met=0.7, clo=0.3, body_position="sitting"
        )["q_sensible"]
        == -70.2
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=20, met=0.7, clo=0.5, body_position="sitting"
        )["heat_strain"]
        == 0
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=20, met=0.7, clo=0.7, body_position="sitting"
        )["w"]
        == 0.4
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=20, met=1.3, clo=0.3, body_position="sitting"
        )["heat_strain_w"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=20, met=1.3, clo=0.5, body_position="sitting"
        )["heat_strain"]
        == 1
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=20, met=1.3, clo=0.7, body_position="sitting"
        )["e_skin"]
        == 117.9
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=20, met=2, clo=0.3, body_position="sitting"
        )["heat_strain_w"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=20, met=2, clo=0.5, body_position="sitting"
        )["w"]
        == 0.6
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=20, met=2, clo=0.7, body_position="sitting"
        )["heat_strain_blood_flow"]
        == True
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=40, met=0.7, clo=0.3, body_position="sitting"
        )["heat_strain_w"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=40, met=0.7, clo=0.5, body_position="sitting"
        )["t_core"]
        == 37.1
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=40, met=0.7, clo=0.7, body_position="sitting"
        )["heat_strain_blood_flow"]
        == True
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=40, met=1.3, clo=0.3, body_position="sitting"
        )["e_max"]
        == 189.2
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=40, met=1.3, clo=0.5, body_position="sitting"
        )["q_sensible"]
        == -52.3
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=40, met=1.3, clo=0.7, body_position="sitting"
        )["e_rsw"]
        == 78.4
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=40, met=2, clo=0.3, body_position="sitting"
        )["t_skin"]
        == 37.7
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=40, met=2, clo=0.5, body_position="sitting"
        )["heat_strain"]
        == 1
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=1, rh=40, met=2, clo=0.7, body_position="sitting"
        )["e_max"]
        == 137.6
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=20, met=0.7, clo=0.3, body_position="sitting"
        )["heat_strain_sweating"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=20, met=0.7, clo=0.5, body_position="sitting"
        )["q_skin"]
        == 37.4
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=20, met=0.7, clo=0.7, body_position="sitting"
        )["e_max"]
        == 238.9
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=20, met=1.3, clo=0.3, body_position="sitting"
        )["q_skin"]
        == 69.3
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=20, met=1.3, clo=0.5, body_position="sitting"
        )["e_max"]
        == 312.6
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=20, met=1.3, clo=0.7, body_position="sitting"
        )["w"]
        == 0.5
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=20, met=2, clo=0.3, body_position="sitting"
        )["w"]
        == 0.4
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=20, met=2, clo=0.5, body_position="sitting"
        )["m_rsw"]
        == 230.0
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=20, met=2, clo=0.7, body_position="sitting"
        )["q_skin"]
        == 88.9
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=40, met=0.7, clo=0.3, body_position="sitting"
        )["e_skin"]
        == 129.3
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=40, met=0.7, clo=0.5, body_position="sitting"
        )["m_bl"]
        == 80.0
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=40, met=0.7, clo=0.7, body_position="sitting"
        )["t_core"]
        == 37.3
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=40, met=1.3, clo=0.3, body_position="sitting"
        )["q_res"]
        == 2.7
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=40, met=1.3, clo=0.5, body_position="sitting"
        )["t_skin"]
        == 37.3
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=40, met=1.3, clo=0.7, body_position="sitting"
        )["heat_strain_sweating"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=40, met=2, clo=0.3, body_position="sitting"
        )["e_skin"]
        == 162.7
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=40, met=2, clo=0.5, body_position="sitting"
        )["heat_strain_w"]
        == True
    )
    assert (
        use_fans_heatwaves(
            tdb=45, tr=45, v=4, rh=40, met=2, clo=0.7, body_position="sitting"
        )["w_max"]
        == 0.5
    )


def test_f_svv():
    assert round(f_svv(30, 10, 3.3), 2) == 0.27
    assert round(f_svv(150, 10, 3.3), 2) == 0.31
    assert round(f_svv(30, 6, 3.3), 2) == 0.20
    assert round(f_svv(150, 6, 3.3), 2) == 0.23
    assert round(f_svv(30, 10, 6), 2) == 0.17
    assert round(f_svv(150, 10, 6), 2) == 0.21
    assert round(f_svv(30, 6, 6), 2) == 0.11
    assert round(f_svv(150, 6, 6), 2) == 0.14
    assert round(f_svv(6, 9, 3.3), 2) == 0.14
    assert round(f_svv(6, 6, 3.3), 2) == 0.11
    assert round(f_svv(6, 6, 6), 2) == 0.04
    assert round(f_svv(4, 4, 3.3), 2) == 0.06
    assert round(f_svv(4, 4, 6), 2) == 0.02


def test_t_dp():
    assert t_dp(31.6, 59.6) == 22.6
    assert t_dp(29.3, 75.4) == 24.3
    assert t_dp(27.1, 66.4) == 20.2


def test_t_wb():
    assert t_wb(27.1, 66.4) == 22.4
    assert t_wb(25, 50) == 18.0


def test_enthalpy():
    assert enthalpy(25, 0.01) == 50561.25
    assert enthalpy(27.1, 0.01) == 52707.56


def test_psy_ta_rh():
    assert psy_ta_rh(25, 50, patm=101325) == {
        "p_sat": 3169.2,
        "p_vap": 1584.6,
        "hr": 0.009881547577511219,
        "t_wb": 18.0,
        "t_dp": 13.8,
        "h": 50259.66,
    }


def test_cooling_effect():
    assert (cooling_effect(tdb=25, tr=25, vr=0.05, rh=50, met=1, clo=0.6)) == 0
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=50, met=1, clo=0.6)) == 2.17
    assert (cooling_effect(tdb=27, tr=25, vr=0.5, rh=50, met=1, clo=0.6)) == 1.85
    assert (cooling_effect(tdb=29, tr=25, vr=0.5, rh=50, met=1, clo=0.6)) == 1.63
    assert (cooling_effect(tdb=31, tr=25, vr=0.5, rh=50, met=1, clo=0.6)) == 1.42
    assert (cooling_effect(tdb=25, tr=27, vr=0.5, rh=50, met=1, clo=0.6)) == 2.44
    assert (cooling_effect(tdb=25, tr=29, vr=0.5, rh=50, met=1, clo=0.6)) == 2.81
    assert (cooling_effect(tdb=25, tr=25, vr=0.2, rh=50, met=1, clo=0.6)) == 0.67
    assert (cooling_effect(tdb=25, tr=25, vr=0.8, rh=50, met=1, clo=0.6)) == 2.93
    assert (cooling_effect(tdb=25, tr=25, vr=0.0, rh=50, met=1, clo=0.6)) == 0
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=60, met=1, clo=0.6)) == 2.13
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=80, met=1, clo=0.6)) == 2.06
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=20, met=1, clo=0.6)) == 2.29
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=60, met=1.3, clo=0.6)) == 2.83
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=60, met=1.6, clo=0.6)) == 3.5
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=60, met=1, clo=0.3)) == 2.41
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=60, met=1, clo=1)) == 2.05

    assert (
        cooling_effect(tdb=77, tr=77, vr=1.64, rh=50, met=1, clo=0.6, units="IP")
    ) == 3.95


def test_running_mean_outdoor_temperature():
    assert (running_mean_outdoor_temperature([20, 20], alpha=0.7)) == 20
    assert (running_mean_outdoor_temperature([20, 20], alpha=0.9)) == 20
    assert (running_mean_outdoor_temperature([20, 20, 20, 20], alpha=0.7)) == 20
    assert (running_mean_outdoor_temperature([20, 20, 20, 20], alpha=0.5)) == 20
    assert (
        running_mean_outdoor_temperature(
            [77, 77, 77, 77, 77, 77, 77], alpha=0.8, units="IP"
        )
    ) == 77
    assert (
        running_mean_outdoor_temperature(
            [77, 77, 77, 77, 77, 77, 77], alpha=0.8, units="ip"
        )
    ) == 77


def test_ip_units_converter():
    assert (units_converter(tdb=77, tr=77, v=3.2, from_units="ip")) == [
        25.0,
        25.0,
        0.975312404754648,
    ]
    assert (units_converter(pressure=1, area=1 / 0.09, from_units="ip")) == [
        101325,
        1.0322474090590033,
    ]


def test_p_sat():
    assert (p_sat(tdb=25)) == 3169.2
    assert (p_sat(tdb=50)) == 12349.9


def test_t_globe():
    assert (t_mrt(tg=53.2, tdb=30, v=0.3, d=0.1, emissivity=0.95)) == 74.8
    assert (t_mrt(tg=55, tdb=30, v=0.3, d=0.1, emissivity=0.95)) == 77.8


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

    with pytest.raises(ValueError):
        adaptive_ashrae(20, 20, 9, 0.1)

    with pytest.raises(ValueError):
        adaptive_ashrae(20, 20, 34, 0.1)


# todo implement test for adaptive_en()


def test_clo_tout():
    assert (clo_tout(tout=80.6, units="ip")) == 0.46
    assert (clo_tout(tout=27)) == 0.46


def test_vertical_tmp_grad_ppd():
    assert (
        vertical_tmp_grad_ppd(77, 77, 0.328, 50, 1.2, 0.5, 7 / 1.8, units="ip")[
            "PPD_vg"
        ]
    ) == 13.0
    assert (
        vertical_tmp_grad_ppd(77, 77, 0.328, 50, 1.2, 0.5, 7 / 1.8, units="ip")[
            "Acceptability"
        ]
    ) == False
    assert (vertical_tmp_grad_ppd(25, 25, 0.1, 50, 1.2, 0.5, 7)["PPD_vg"]) == 12.6
    assert (vertical_tmp_grad_ppd(25, 25, 0.1, 50, 1.2, 0.5, 4)["PPD_vg"]) == 1.7
    assert (
        vertical_tmp_grad_ppd(25, 25, 0.1, 50, 1.2, 0.5, 4)["Acceptability"]
    ) == True

    with pytest.raises(ValueError):
        vertical_tmp_grad_ppd(25, 25, 0.3, 50, 1.2, 0.5, 7)


def test_ankle_draft():
    assert (ankle_draft(25, 25, 0.2, 50, 1.2, 0.5, 0.3, units="SI")["PPD_ad"]) == 18.5
    assert (
        ankle_draft(77, 77, 0.2 * 3.28, 50, 1.2, 0.5, 0.4 * 3.28, units="IP")["PPD_ad"]
    ) == 23.5

    with pytest.raises(ValueError):
        ankle_draft(25, 25, 0.3, 50, 1.2, 0.5, 7)


@pytest.fixture
def data_test_utci():
    return [  # I have commented the lines of code that don't pass the test
        {"tdb": 25, "tr": 27, "rh": 50, "v": 1, "return": {"utci": 25.2}},
        {"tdb": 19, "tr": 24, "rh": 50, "v": 1, "return": {"utci": 20.0}},
        {"tdb": 19, "tr": 14, "rh": 50, "v": 1, "return": {"utci": 16.8}},
        {"tdb": 27, "tr": 22, "rh": 50, "v": 1, "return": {"utci": 25.5}},
        {"tdb": 27, "tr": 22, "rh": 50, "v": 10, "return": {"utci": 20.0}},
        {"tdb": 27, "tr": 22, "rh": 50, "v": 16, "return": {"utci": 15.8}},
        {"tdb": 51, "tr": 22, "rh": 50, "v": 16, "return": {"utci": np.nan}},
        {"tdb": 27, "tr": 22, "rh": 50, "v": 0, "return": {"utci": np.nan}},
    ]


def test_utci(data_test_utci):
    for row in data_test_utci:
        np.testing.assert_equal(
            utci(row["tdb"], row["tr"], row["v"], row["rh"]),
            row["return"][list(row["return"].keys())[0]],
        )

    assert (utci(tdb=77, tr=77, v=3.28, rh=50, units="ip")) == 76.4

    assert (
        utci(tdb=25, tr=27, v=1, rh=50, units="si", return_stress_category=True)
    ) == {"utci": 25.2, "stress_category": "no thermal stress"}
    assert (
        utci(tdb=25, tr=25, v=1, rh=50, units="si", return_stress_category=True)
    ) == {"utci": 24.6, "stress_category": "no thermal stress"}


def test_utci_numpy(data_test_utci):
    tdb = np.array([d["tdb"] for d in data_test_utci])
    tr = np.array([d["tr"] for d in data_test_utci])
    rh = np.array([d["rh"] for d in data_test_utci])
    v = np.array([d["v"] for d in data_test_utci])
    expect = np.array([d["return"]["utci"] for d in data_test_utci])

    np.testing.assert_equal(utci(tdb, tr, v, rh), expect)

    tdb = np.array([25, 25])
    tr = np.array([27, 25])
    v = np.array([1, 1])
    rh = np.array([50, 50])
    expect = {
        "utci": np.array([25.2, 24.6]),
        "stress_category": np.array(["no thermal stress", "no thermal stress"]),
    }

    result = utci(tdb, tr, v, rh, units="si", return_stress_category=True)
    np.testing.assert_equal(result["utci"], expect["utci"])
    np.testing.assert_equal(result["stress_category"], expect["stress_category"])


def test_clo_dynamic():
    assert (clo_dynamic(clo=1, met=1, standard="ASHRAE")) == 1
    assert (clo_dynamic(clo=1, met=0.5, standard="ASHRAE")) == 1
    assert (clo_dynamic(clo=2, met=0.5, standard="ASHRAE")) == 2


def test_phs():
    assert phs(tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2) == {
        "d_lim_loss_50": 440,
        "d_lim_loss_95": 298,
        "d_lim_t_re": 480,
        "water_loss": 6166.0,
        "t_re": 37.5,
    }
    assert phs(tdb=35, tr=35, rh=71, v=0.3, met=150, clo=0.5, posture=2) == {
        "d_lim_loss_50": 385,
        "d_lim_loss_95": 256,
        "d_lim_t_re": 75,
        "water_loss": 6935.0,
        "t_re": 39.8,
    }
    assert phs(tdb=30, tr=50, posture=2, rh=70.65, v=0.3, met=150, clo=0.5) == {
        "t_re": 37.7,
        "water_loss": 7166.0,  # in the standard is 6935
        "d_lim_t_re": 480,
        "d_lim_loss_50": 380,
        "d_lim_loss_95": 258,
    }
    assert phs(
        tdb=28, tr=58, acclimatized=0, posture=2, rh=79.31, v=0.3, met=150, clo=0.5
    ) == {
        "t_re": 41.2,
        "water_loss": 5807,
        "d_lim_t_re": 57,
        "d_lim_loss_50": 466,
        "d_lim_loss_95": 314,
    }
    assert phs(
        tdb=35, tr=35, acclimatized=0, posture=1, rh=53.3, v=1, met=150, clo=0.5
    ) == {
        "t_re": 37.6,
        "water_loss": 3892.0,
        "d_lim_t_re": 480,
        "d_lim_loss_50": 480,
        "d_lim_loss_95": 463,
    }
    assert phs(tdb=43, tr=43, posture=1, rh=34.7, v=0.3, met=103, clo=0.5) == {
        "t_re": 37.3,
        "water_loss": 6765.0,
        "d_lim_t_re": 480,
        "d_lim_loss_50": 401,
        "d_lim_loss_95": 271,
    }
    assert phs(
        tdb=35, tr=35, acclimatized=0, posture=2, rh=53.3, v=0.3, met=206, clo=0.5
    ) == {
        "t_re": 39.2,
        "water_loss": 7236.0,
        "d_lim_t_re": 70,
        "d_lim_loss_50": 372,
        "d_lim_loss_95": 247,
    }
    assert phs(tdb=40, tr=40, rh=40.63, v=0.3, met=150, clo=0.4, posture=2) == {
        "t_re": 37.5,
        "water_loss": 6683.0,
        "d_lim_t_re": 480,
        "d_lim_loss_50": 407,
        "d_lim_loss_95": 276,
    }
    assert phs(
        tdb=40,
        tr=40,
        rh=40.63,
        v=0.3,
        met=150,
        clo=0.4,
        posture=2,
        theta=90,
        walk_sp=1,
    ) == {
        "t_re": 37.6,
        "water_loss": 5379.0,
        "d_lim_t_re": 480,
        "d_lim_loss_50": 480,
        "d_lim_loss_95": 339,
    }


def test_check_standard_compliance():
    with pytest.warns(
        UserWarning,
        match="ISO 7933:2004 air temperature applicability limits between 15 and 50 °C",
    ):
        warnings.warn(
            phs(tdb=70, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2),
            UserWarning,
        )

    with pytest.warns(
        UserWarning,
        match="ISO 7933:2004 t_r - t_db applicability limits between 0 and 60 °C",
    ):
        warnings.warn(
            phs(tdb=20, tr=0, rh=33.85, v=0.3, met=150, clo=0.5, posture=2),
            UserWarning,
        )

    with pytest.warns(
        UserWarning,
        match="ISO 7933:2004 air speed applicability limits between 0 and 3 m/s",
    ):
        warnings.warn(
            phs(tdb=40, tr=40, rh=33.85, v=5, met=150, clo=0.5, posture=2),
            UserWarning,
        )

    with pytest.warns(
        UserWarning,
        match="ISO 7933:2004 met applicability limits between 100 and 450 met",
    ):
        warnings.warn(
            phs(tdb=40, tr=40, rh=33.85, v=2, met=1, clo=0.5, posture=2),
            UserWarning,
        )

    with pytest.warns(
        UserWarning,
        match="ISO 7933:2004 clo applicability limits between 0.1 and 1 clo",
    ):
        warnings.warn(
            phs(tdb=40, tr=40, rh=33.85, v=2, met=150, clo=2, posture=2),
            UserWarning,
        )

    with pytest.warns(
        UserWarning,
        match="ISO 7933:2004 t_r - t_db applicability limits between 0 and",
    ):
        warnings.warn(
            phs(tdb=40, tr=40, rh=61, v=2, met=150, clo=2, posture=2),
            UserWarning,
        )


def test_body_surface_area():
    assert body_surface_area(weight=80, height=1.8) == 1.9917607971689137


def test_t_o():
    assert t_o(25, 25, 0.1) == 25
    assert round(t_o(25, 30, 0.3), 2) == 26.83
    assert round(t_o(20, 30, 0.3), 2) == 23.66
    assert t_o(25, 25, 0.1, standard="ASHRAE") == 25
    assert t_o(20, 30, 0.1, standard="ASHRAE") == 25
    assert t_o(20, 30, 0.3, standard="ASHRAE") == 24
    assert t_o(20, 30, 0.7, standard="ASHRAE") == 23


def test_wbgt():
    assert wbgt(25, 30) == 26.5
    assert wbgt(twb=25, tg=32) == 27.1
    assert wbgt(twb=25, tg=32, tdb=20) == 27.1
    assert wbgt(twb=25, tg=32, tdb=20, with_solar_load=True) == 25.9
    with pytest.raises(ValueError):
        wbgt(twb=25, tg=32, with_solar_load=True)
    # data from Table D.1 ISO 7243
    assert wbgt(twb=17.3, tg=40, round=True) == 24.1
    assert wbgt(twb=21.1, tg=55, round=True) == 31.3
    assert wbgt(twb=16.7, tg=40, round=True) == 23.7


def test_at():
    assert at(tdb=25, rh=30, v=0.1) == 24.1
    assert at(tdb=23, rh=70, v=1) == 24.8
    assert at(tdb=23, rh=70, v=1, q=50) == 28.1


def test_heat_index():
    assert heat_index(25, 50) == 25.9
    assert heat_index(77, 50, units="IP") == 78.6
    assert heat_index(30, 80) == 37.7
    assert heat_index(86, 80, units="IP") == 99.8


def test_wc():
    assert wc(tdb=0, v=0.1) == {"wci": 518.6}
    assert wc(tdb=0, v=1.5) == {"wci": 813.5}
    assert wc(tdb=-5, v=5.5) == {"wci": 1255.2}
    assert wc(tdb=-10, v=11) == {"wci": 1631.1}
    assert wc(tdb=-5, v=11) == {"wci": 1441.4}


def test_humidex():
    assert humidex(25, 50) == {"humidex": 28.2, "discomfort": "Little or no discomfort"}
    assert humidex(30, 80) == {
        "humidex": 43.3,
        "discomfort": "Intense discomfort; avoid exertion",
    }
    assert humidex(31.6, 57.1) == {
        "humidex": 40.8,
        "discomfort": "Intense discomfort; avoid exertion",
    }


def test_net():
    assert net(37, 100, 0.1) == 37
    assert net(37, 100, 4.5) == 37
    assert net(25, 100, 4.5) == 20
    assert net(25, 100, 0.1) == 25.4
    assert net(40, 48.77, 0.1) == 33.8
    assert net(36, 50.196, 0.1) == 30.9


def test_two_nodes():
    # todo write more tests to validate all the following
    #  effective temperature (already implemented in two_nodes)
    #  pt set (already implemented in two_nodes)
    #  pd (already implemented in two_nodes)
    #  ps (already implemented in two_nodes)
    #  t_sens (already implemented in two_nodes)

    assert two_nodes(25, 25, 1.1, 50, 2, 0.5)["disc"] == 0.3
    assert two_nodes(tdb=25, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)["disc"] == 0.2
    assert two_nodes(tdb=30, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)["disc"] == 1.0
    assert two_nodes(tdb=30, tr=30, v=0.1, rh=50, met=1.2, clo=0.5)["disc"] == 1.5
    assert two_nodes(tdb=28, tr=28, v=0.4, rh=50, met=1.2, clo=0.5)["disc"] == 0.7

    assert two_nodes(tdb=30, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)["pmv_gagge"] == 0.9
    assert two_nodes(tdb=30, tr=30, v=0.1, rh=50, met=1.2, clo=0.5)["pmv_gagge"] == 1.5
    assert two_nodes(tdb=28, tr=28, v=0.4, rh=50, met=1.2, clo=0.5)["pmv_gagge"] == 0.7

    assert two_nodes(tdb=30, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)["pmv_set"] == 0.9
    assert two_nodes(tdb=30, tr=30, v=0.1, rh=50, met=1.2, clo=0.5)["pmv_set"] == 1.4
    assert two_nodes(tdb=28, tr=28, v=0.4, rh=50, met=1.2, clo=0.5)["pmv_set"] == 0.5
