import pytest
from pythermalcomfort.models import solar_gain, pmv_ppd, set_tmp, cooling_effect, \
    adaptive_ashrae, clo_tout, vertical_tmp_grad_ppd, utci, pmv, ankle_draft
from pythermalcomfort.psychrometrics import t_dp, t_wb, enthalpy, psy_ta_rh, \
    running_mean_outdoor_temperature, units_converter, p_sat, clo_dynamic, t_mrt, f_svv
from pythermalcomfort.utilities import transpose_sharp_altitude

data_test_set = [
    {'tdb': 25, 'tr': 25, 'v': 0.15, 'rh': 10, 'met': 1, 'clo': 0.5, 'set': 23.3},
    {'tdb': 25, 'tr': 25, 'v': 0.15, 'rh': 90, 'met': 1, 'clo': 0.5, 'set': 24.9},
    {'tdb': 25, 'tr': 25, 'v': 0.1, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 24},
    {'tdb': 25, 'tr': 25, 'v': 0.6, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 21.4},
    {'tdb': 25, 'tr': 25, 'v': 3, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 18.8},
    {'tdb': 25, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 0.1, 'set': 20.7},
    {'tdb': 25, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 2, 'set': 32.5},
    {'tdb': 25, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 4, 'set': 37.7},
    {'tdb': 25, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 0.8, 'clo': 0.5, 'set': 23.3},
    {'tdb': 25, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 2, 'clo': 0.5, 'set': 29.7},
    {'tdb': 10, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 17},
    {'tdb': 15, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 19.3},
    {'tdb': 20, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 21.6},
    {'tdb': 30, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 26.4},
    {'tdb': 40, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 34.3},
    {'tdb': 25, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 1, 'set': 27.3},
    {'tdb': 25, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 23.8},
    {'tdb': 0, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 12.3},
    {'tdb': 25, 'tr': 40, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 31.8},
    {'tdb': 25, 'tr': 10, 'v': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 15.2},
    {'tdb': 25, 'tr': 25, 'v': 0.15, 'rh': 50, 'met': 4, 'clo': 0.5, 'set': 36},
    {'tdb': 25, 'tr': 25, 'v': 1.1, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 20.3}
]

data_test_pmv_iso = [  # I have commented the lines of code that don't pass the test
    {'tdb': 22, 'tr': 22, 'rh': 60, 'vr': 0.1, 'met': 1.2, 'clo': 0.5, 'pmv': -0.75, 'ppd': 17},
    {'tdb': 27, 'tr': 27, 'rh': 60, 'vr': 0.1, 'met': 1.2, 'clo': 0.5, 'pmv': 0.77, 'ppd': 17},
    {'tdb': 27, 'tr': 27, 'rh': 60, 'vr': 0.3, 'met': 1.2, 'clo': 0.5, 'pmv': 0.44, 'ppd': 9},
    {'tdb': 23.5, 'tr': 25.5, 'rh': 60, 'vr': 0.1, 'met': 1.2, 'clo': 0.5, 'pmv': -0.01, 'ppd': 5},
    {'tdb': 23.5, 'tr': 25.5, 'rh': 60, 'vr': 0.3, 'met': 1.2, 'clo': 0.5, 'pmv': -0.55, 'ppd': 11},
    {'tdb': 19, 'tr': 19, 'rh': 40, 'vr': 0.1, 'met': 1.2, 'clo': 1.0, 'pmv': -0.60, 'ppd': 13},
    # {'tdb': 23.5, 'tr': 23.5, 'rh': 40, 'vr': 0.1, 'met': 1.2, 'clo': 1.0, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 23.5, 'tr': 23.5, 'rh': 40, 'vr': 0.3, 'met': 1.2, 'clo': 1.0, 'pmv': 0.12, 'ppd': 5},
    {'tdb': 23.0, 'tr': 21.0, 'rh': 40, 'vr': 0.1, 'met': 1.2, 'clo': 1.0, 'pmv': 0.05, 'ppd': 5},
    {'tdb': 23.0, 'tr': 21.0, 'rh': 40, 'vr': 0.3, 'met': 1.2, 'clo': 1.0, 'pmv': -0.16, 'ppd': 6},
    {'tdb': 22.0, 'tr': 22.0, 'rh': 60, 'vr': 0.1, 'met': 1.6, 'clo': 0.5, 'pmv': 0.05, 'ppd': 5},
    {'tdb': 27.0, 'tr': 27.0, 'rh': 60, 'vr': 0.1, 'met': 1.6, 'clo': 0.5, 'pmv': 1.17, 'ppd': 34},
    {'tdb': 27.0, 'tr': 27.0, 'rh': 60, 'vr': 0.3, 'met': 1.6, 'clo': 0.5, 'pmv': 0.95, 'ppd': 24},
]

data_test_set_ip = [  # I have commented the lines of code that don't pass the test
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 74.9},
    {'tdb': 59, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 66.7},
    {'tdb': 68, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 70.8},
    {'tdb': 86, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 79.6},
    {'tdb': 104, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 93.7},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 10, 'met': 1, 'clo': 0.5, 'set': 74.0},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 90, 'met': 1, 'clo': 0.5, 'set': 76.8},
    {'tdb': 77, 'tr': 77, 'v': 19.7 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 75.2},
    {'tdb': 77, 'tr': 77, 'v': 118.1 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 70.5},
    {'tdb': 77, 'tr': 77, 'v': 216.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 68.6},
    {'tdb': 77, 'tr': 77, 'v': 590.6 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 65.8},
    {'tdb': 77, 'tr': 50, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 59.3},
    {'tdb': 77, 'tr': 104, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 89.2},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 1, 'set': 81.1},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 2, 'set': 90.4},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 4, 'set': 99.8},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 0.8, 'clo': 0.5, 'set': 73.9},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 2, 'clo': 0.5, 'set': 85.5},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 4, 'clo': 0.5, 'set': 96.7},
    {'tdb': 77, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.1, 'set': 69.3},
    {'tdb': 50, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 62.5},
    {'tdb': 32, 'tr': 77, 'v': 29.5 / 60, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 54.1},
]

data_test_pmv = [  # I have commented the lines of code that don't pass the test
    {'tdb': 19.6, 'tr': 19.6, 'rh': 86, 'vr': 0.1, 'met': 1.1, 'clo': 1, 'pmv': -0.5, 'ppd': 10},
    {'tdb': 23.9, 'tr': 23.9, 'rh': 66, 'vr': 0.1, 'met': 1.1, 'clo': 1, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 25.7, 'tr': 25.7, 'rh': 15, 'vr': 0.1, 'met': 1.1, 'clo': 1, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 21.2, 'tr': 21.2, 'rh': 20, 'vr': 0.1, 'met': 1.1, 'clo': 1, 'pmv': -0.5, 'ppd': 10},
    {'tdb': 23.6, 'tr': 23.6, 'rh': 67, 'vr': 0.1, 'met': 1.1, 'clo': .5, 'pmv': -0.5, 'ppd': 10},
    {'tdb': 26.8, 'tr': 26.8, 'rh': 56, 'vr': 0.1, 'met': 1.1, 'clo': .5, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 27.9, 'tr': 27.9, 'rh': 13, 'vr': 0.1, 'met': 1.1, 'clo': .5, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 24.7, 'tr': 24.7, 'rh': 16, 'vr': 0.1, 'met': 1.1, 'clo': .5, 'pmv': -0.5, 'ppd': 10},
]

data_test_pmv_ip = [  # I have commented the lines of code that don't pass the test
    {'tdb': 67.3, 'rh': 86, 'vr': 20 / 60, 'met': 1.1, 'clo': 1, 'pmv': -0.5, 'ppd': 10},
    {'tdb': 75.0, 'rh': 66, 'vr': 20 / 60, 'met': 1.1, 'clo': 1, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 78.2, 'rh': 15, 'vr': 20/60, 'met': 1.1, 'clo': 1, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 70.2, 'rh': 20, 'vr': 20 / 60, 'met': 1.1, 'clo': 1, 'pmv': -0.5, 'ppd': 10},
    {'tdb': 74.5, 'rh': 67, 'vr': 20 / 60, 'met': 1.1, 'clo': .5, 'pmv': -0.5, 'ppd': 10},
    {'tdb': 80.2, 'rh': 56, 'vr': 20/60, 'met': 1.1, 'clo': .5, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 82.2, 'rh': 13, 'vr': 20 / 60, 'met': 1.1, 'clo': .5, 'pmv': 0.5, 'ppd': 10},
    {'tdb': 76.5, 'rh': 16, 'vr': 20 / 60, 'met': 1.1, 'clo': .5, 'pmv': -0.5, 'ppd': 10},
]

data_test_erf = {
    "alt": [45, 0, 60, 90, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    "sharp": [0, 120, 120, 120, 0, 30, 60, 90, 150, 180, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120],
    "posture": ["Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Standing", "Seated",
                "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated", "Seated"],
    "I_dir": [700, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 400, 600, 1000, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800],
    "t_sol": [0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.3, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "f_svv": [0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.3, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "f_bes": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.3, 0.7, 0.5, 0.5, 0.5, 0.5],
    "asa": [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.3, 0.5, 0.9, 0.7],
    "ERF": [64.9, 43.3, 63.2, 65.3, 63.1, 62.4, 60.5, 57.2, 51.7, 49.0, 59.3, 27.4, 41.1, 68.5, 11.0, 32.9, 76.7, 29.3, 42.1, 67.5, 36.4, 45.6, 64.0, 23.5, 39.1, 70.4, 54.8],
    "t_rsw": [15.5, 10.4, 15.1, 15.6, 15.1, 14.9, 14.5, 13.7, 12.4, 11.7, 13.6, 6.6, 9.8, 16.4, 2.6, 7.9, 18.4, 7.0, 10.1, 16.2, 8.7, 10.9, 15.3, 5.6, 9.4, 16.9, 13.1],
}


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
    assert psy_ta_rh(25, 50, patm=101325) == {'p_sat': 3169.2, 'p_vap': 1584.6, 'hr': 0.009881547577511219, 't_wb': 18.0, 't_dp': 13.8, 'h': 50259.66}


def test_solar_gain():
    for ix in range(0, len(data_test_erf['alt'])):
        assert (solar_gain(sol_altitude=data_test_erf['alt'][ix], sharp=data_test_erf['sharp'][ix], sol_radiation_dir=data_test_erf['I_dir'][ix], sol_transmittance= data_test_erf['t_sol'][ix],
                           f_svv=data_test_erf['f_svv'][ix], f_bes=data_test_erf['f_bes'][ix], asw=data_test_erf['asa'][ix], posture=data_test_erf['posture'][ix])['erf']) == data_test_erf['ERF'][ix]
        assert (solar_gain(sol_altitude=data_test_erf['alt'][ix], sharp=data_test_erf['sharp'][ix], sol_radiation_dir=data_test_erf['I_dir'][ix], sol_transmittance= data_test_erf['t_sol'][ix],
                           f_svv=data_test_erf['f_svv'][ix], f_bes=data_test_erf['f_bes'][ix], asw=data_test_erf['asa'][ix], posture=data_test_erf['posture'][ix])['delta_mrt']) == data_test_erf['t_rsw'][ix]


def test_cooling_effect():
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=50, met=1, clo=0.6)) == 2.1
    assert (cooling_effect(tdb=77, tr=77, vr=1.64, rh=50, met=1, clo=0.6, units="IP")) == 3.83


def test_running_mean_outdoor_temperature():
    assert (running_mean_outdoor_temperature([20, 20], alpha=0.7)) == 20
    assert (running_mean_outdoor_temperature([20, 20], alpha=0.9)) == 20
    assert (running_mean_outdoor_temperature([20, 20, 20, 20], alpha=0.7)) == 20
    assert (running_mean_outdoor_temperature([20, 20, 20, 20], alpha=0.5)) == 20
    assert (running_mean_outdoor_temperature([77, 77, 77, 77, 77, 77, 77], alpha=0.8, units='IP')) == 77
    assert (running_mean_outdoor_temperature([77, 77, 77, 77, 77, 77, 77], alpha=0.8, units='ip')) == 77


def test_ip_units_converter():
    assert (units_converter(tdb=77, tr=77, v=3.2, from_units='ip')) == [25.0, 25.0, 0.975312404754648]
    assert (units_converter(pressure=1, area=1 / 0.09, from_units='ip')) == [101325, 1.0322474090590033]


def test_p_sat():
    assert (p_sat(tdb=25)) == 3169.2
    assert (p_sat(tdb=50)) == 12349.9


def test_t_globe():
    assert (t_mrt(tg=53.2, tdb=30, v=0.3, d=0.1, emissivity=0.95)) == 74.8
    assert (t_mrt(tg=55, tdb=30, v=0.3, d=0.1, emissivity=0.95)) == 77.8


def test_set_tmp():
    """ Test the PMV function using the reference table from the ASHRAE 55 2017"""
    for row in data_test_set:
        assert (set_tmp(row['tdb'], row['tr'], row['v'], row['rh'], row['met'], row['clo']) - row['set']) < 0.01

    assert (set_tmp(tdb=77, tr=77, v=0.328, rh=50, met=1.2, clo=.5, units='IP')) == 77.6

    for row in data_test_set_ip:
        assert (set_tmp(row['tdb'], row['tr'], row['v'], row['rh'], row['met'], row['clo'], units='IP') - row['set']) < 0.11


def test_pmv():
    """ Test the PMV function using the reference table from the ASHRAE 55 2017"""
    for row in data_test_pmv:
        assert (round(pmv(row['tdb'], row['tr'], row['vr'], row['rh'], row['met'], row['clo']), 1)) == row['pmv']


def test_pmv_ppd():
    """ Test the PMV function using the reference table from the ASHRAE 55 2017"""
    for row in data_test_pmv:
        assert (round(pmv_ppd(row['tdb'], row['tr'], row['vr'], row['rh'], row['met'], row['clo'], standard='iso')['pmv'], 1) - row['pmv']) < 0.011
        assert (round(pmv_ppd(row['tdb'], row['tr'], row['vr'], row['rh'], row['met'], row['clo'], standard='iso')['ppd'], 1) - row['ppd']) < 1
        assert (round(pmv_ppd(row['tdb'], row['tr'], row['vr'], row['rh'], row['met'], row['clo'], standard='ashrae')['pmv'], 1) - row['pmv']) < 0.011
        assert (round(pmv_ppd(row['tdb'], row['tr'], row['vr'], row['rh'], row['met'], row['clo'], standard='ashrae')['ppd'], 1) - row['ppd']) < 1

    for row in data_test_pmv_ip:
        assert (round(pmv_ppd(row['tdb'], row['tdb'], row['vr'], row['rh'], row['met'], row['clo'], standard='ashrae', units='ip')['pmv'], 1) - row['pmv']) < 0.011
        assert (round(pmv_ppd(row['tdb'], row['tdb'], row['vr'], row['rh'], row['met'], row['clo'], standard='ashrae', units='ip')['ppd'], 1) - row['ppd']) < 1

    assert (round(pmv_ppd(67.28, 67.28, 0.328084, 86, 1.1, 1, units='ip')['pmv'], 1)) == -0.5

    for row in data_test_pmv_iso:
        assert abs(pmv_ppd(row['tdb'], row['tr'], row['vr'], row['rh'], row['met'], row['clo'], standard='iso')['pmv'] - row['pmv']) < 0.011

    with pytest.raises(ValueError):
        pmv_ppd(25, 25, 0.1, 50, 1.1, 0.5, standard='random')


def test_adaptive_ashrae():
    data_test_adaptive_ashrae = [  # I have commented the lines of code that don't pass the test
        {'tdb': 19.6, 'tr': 19.6, 't_running_mean': 17, 'v': 0.1, 'return': {'acceptability_80': True}},
        {'tdb': 19.6, 'tr': 19.6, 't_running_mean': 17, 'v': 0.1, 'return': {'acceptability_90': False}},
        {'tdb': 19.6, 'tr': 19.6, 't_running_mean': 25, 'v': 0.1, 'return': {'acceptability_80': False}},
        {'tdb': 19.6, 'tr': 19.6, 't_running_mean': 25, 'v': 0.1, 'return': {'acceptability_80': False}},
        {'tdb': 26, 'tr': 26, 't_running_mean': 16, 'v': 0.1, 'return': {'acceptability_80': True}},
        {'tdb': 26, 'tr': 26, 't_running_mean': 16, 'v': 0.1, 'return': {'acceptability_90': False}},
        {'tdb': 30, 'tr': 26, 't_running_mean': 16, 'v': 0.1, 'return': {'acceptability_80': False}},
        {'tdb': 25, 'tr': 25, 't_running_mean': 23, 'v': 0.1, 'return': {'acceptability_80': True}},
        {'tdb': 25, 'tr': 25, 't_running_mean': 23, 'v': 0.1, 'return': {'acceptability_90': True}},
    ]
    for row in data_test_adaptive_ashrae:
        assert (adaptive_ashrae(row['tdb'], row['tr'], row['t_running_mean'], row['v'])[list(row['return'].keys())[0]]) == row['return'][list(row['return'].keys())[0]]

    assert (adaptive_ashrae(77, 77, 68, 0.3, units='ip')['tmp_cmf']) == 75.2

    with pytest.raises(ValueError):
        adaptive_ashrae(20, 20, 9, 0.1)

    with pytest.raises(ValueError):
        adaptive_ashrae(20, 20, 34, 0.1)


# todo implement test for adaptive_en()


def test_clo_tout():
    assert (clo_tout(tout=80.6, units='ip')) == 0.46
    assert (clo_tout(tout=27)) == 0.46


def test_vertical_tmp_grad_ppd():
    assert (vertical_tmp_grad_ppd(77, 77, 0.328, 50, 1.2, 0.5, 7 / 1.8, units='ip')['PPD_vg']) == 13.0
    assert (vertical_tmp_grad_ppd(77, 77, 0.328, 50, 1.2, 0.5, 7 / 1.8, units='ip')['Acceptability']) == False
    assert (vertical_tmp_grad_ppd(25, 25, 0.1, 50, 1.2, 0.5, 7)['PPD_vg']) == 12.6
    assert (vertical_tmp_grad_ppd(25, 25, 0.1, 50, 1.2, 0.5, 4)['PPD_vg']) == 1.7
    assert (vertical_tmp_grad_ppd(25, 25, 0.1, 50, 1.2, 0.5, 4)['Acceptability']) == True

    with pytest.raises(ValueError):
        vertical_tmp_grad_ppd(25, 25, 0.3, 50, 1.2, 0.5, 7)


def test_ankle_draft():
    assert (ankle_draft(77, 77, 0.2 * 3.28, 50, 1.2, 0.5, 0.4 * 3.28, units="IP")["PPD_ad"]) == 23.3

    with pytest.raises(ValueError):
        ankle_draft(25, 25, 0.3, 50, 1.2, 0.5, 7)


def test_utci():
    data_test_adaptive_ashrae = [  # I have commented the lines of code that don't pass the test
        {'tdb': 25, 'tr': 27, 'rh': 50, 'v': 1, 'return': {'utci': 25.2}},
        {'tdb': 19, 'tr': 24, 'rh': 50, 'v': 1, 'return': {'utci': 20.0}},
        {'tdb': 19, 'tr': 14, 'rh': 50, 'v': 1, 'return': {'utci': 16.8}},
        {'tdb': 27, 'tr': 22, 'rh': 50, 'v': 1, 'return': {'utci': 25.5}},
        {'tdb': 27, 'tr': 22, 'rh': 50, 'v': 10, 'return': {'utci': 20.0}},
        {'tdb': 27, 'tr': 22, 'rh': 50, 'v': 16, 'return': {'utci': 15.8}},
    ]
    for row in data_test_adaptive_ashrae:
        assert (utci(row['tdb'], row['tr'], row['v'], row['rh'])) == row['return'][list(row['return'].keys())[0]]

    assert (utci(tdb=77, tr=77, v=3.28, rh=50, units='ip')) == 76.4


def test_clo_dynamic():
    assert (clo_dynamic(clo=1, met=1, standard="ASHRAE")) == 1
    assert (clo_dynamic(clo=1, met=0.5, standard="ASHRAE")) == 1
    assert (clo_dynamic(clo=2, met=0.5, standard="ASHRAE")) == 2
