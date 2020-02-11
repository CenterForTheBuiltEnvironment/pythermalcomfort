from pythermalcomfort.models import pmv, pmv_ppd

data_test_set = [
    {'ta': 25, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 23.8},
    {'ta': 25, 'tr': 25, 'vr': 0.15, 'rh': 10, 'met': 1, 'clo': 0.5, 'set': 23.3},
    {'ta': 25, 'tr': 25, 'vr': 0.15, 'rh': 90, 'met': 1, 'clo': 0.5, 'set': 24.9},
    {'ta': 25, 'tr': 25, 'vr': 0.1, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 24},
    {'ta': 25, 'tr': 25, 'vr': 0.6, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 21.4},
    {'ta': 25, 'tr': 25, 'vr': 3, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 18.8},
    {'ta': 25, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 0.1, 'set': 20.7},
    {'ta': 25, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 1, 'set': 27.3},
    {'ta': 25, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 2, 'set': 32.5},
    {'ta': 25, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 4, 'set': 37.7},
    {'ta': 25, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 0.8, 'clo': 0.5, 'set': 23.3},
    {'ta': 25, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 2, 'clo': 0.5, 'set': 29.7},
    {'ta': 10, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 17},
    {'ta': 15, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 19.3},
    {'ta': 20, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 21.6},
    {'ta': 30, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 26.4},
    {'ta': 40, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 34.3},
    {'ta': 0, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 12.3},
    {'ta': 25, 'tr': 40, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 31.8},
    {'ta': 25, 'tr': 10, 'vr': 0.15, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 15.2},
    {'ta': 25, 'tr': 25, 'vr': 0.15, 'rh': 50, 'met': 4, 'clo': 0.5, 'set': 36},
    {'ta': 25, 'tr': 25, 'vr': 1.1, 'rh': 50, 'met': 1, 'clo': 0.5, 'set': 20.3}
]

data_test_pmv = [
    {'ta': 19.6, 'tr': 19.6, 'rh': 86, 'vr': 0.1, 'met': 1.1, 'clo': 1, 'pmv': -0.5, 'ppd': 10},
    {'ta': 23.9, 'tr': 23.9, 'rh': 66, 'vr': 0.1, 'met': 1.1, 'clo': 1, 'pmv': 0.5, 'ppd': 10},
    {'ta': 25.7, 'tr': 25.7, 'rh': 15, 'vr': 0.1, 'met': 1.1, 'clo': 1, 'pmv': 0.5, 'ppd': 11},
    {'ta': 21.2, 'tr': 21.2, 'rh': 20, 'vr': 0.1, 'met': 1.1, 'clo': 1, 'pmv': -0.5, 'ppd': 10},
    {'ta': 23.6, 'tr': 23.6, 'rh': 67, 'vr': 0.1, 'met': 1.1, 'clo': .5, 'pmv': -0.5, 'ppd': 10},
    {'ta': 26.8, 'tr': 26.8, 'rh': 56, 'vr': 0.1, 'met': 1.1, 'clo': .5, 'pmv': 0.5, 'ppd': 11},
    {'ta': 27.9, 'tr': 27.9, 'rh': 13, 'vr': 0.1, 'met': 1.1, 'clo': .5, 'pmv': 0.5, 'ppd': 10},
    {'ta': 24.7, 'tr': 24.7, 'rh': 16, 'vr': 0.1, 'met': 1.1, 'clo': .5, 'pmv': -0.5, 'ppd': 10},
]


def test_pmv():
    """ Test the PMV function using the reference table from the ASHRAE 55 2017"""
    for row in data_test_pmv:
        assert (round(pmv(row['ta'], row['tr'], row['vr'], row['rh'], row['met'], row['clo']), 1)) == row['pmv']


def test_pmv_ppd():
    """ Test the PMV function using the reference table from the ASHRAE 55 2017"""
    for row in data_test_pmv:
        assert (round(pmv_ppd(row['ta'], row['tr'], row['vr'], row['rh'], row['met'], row['clo'])['pmv'], 1)) == row['pmv']
        assert (round(pmv_ppd(row['ta'], row['tr'], row['vr'], row['rh'], row['met'], row['clo'])['ppd'], 0)) == row['ppd']
