from itertools import product

import numpy as np
import pytest

from pythermalcomfort.models import cooling_effect


def test_cooling_effect():

    t_range = np.arange(10, 40, 10)
    rh_range = np.arange(10, 75, 25)
    v_range = np.arange(0.1, 4, 1)
    all_combinations = list(product(t_range, rh_range, v_range))
    results = [
        0,
        8.19,
        10.94,
        12.54,
        0,
        8.05,
        10.77,
        12.35,
        0,
        7.91,
        10.6,
        12.16,
        0,
        5.04,
        6.62,
        7.51,
        0,
        4.84,
        6.37,
        7.24,
        0,
        4.64,
        6.12,
        6.97,
        0,
        3.64,
        4.32,
        4.69,
        0,
        3.55,
        4.25,
        4.61,
        0,
        3.4,
        4.1,
        4.46,
    ]
    for ix, comb in enumerate(all_combinations):
        pytest.approx(
            cooling_effect(
                tdb=comb[0],
                tr=comb[0],
                rh=comb[1],
                vr=comb[2],
                met=1,
                clo=0.5,
            )
            == results[ix],
            0.1,
        )

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
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=60, met=1.3, clo=0.6)) == 2.84
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=60, met=1.6, clo=0.6)) == 3.5
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=60, met=1, clo=0.3)) == 2.41
    assert (cooling_effect(tdb=25, tr=25, vr=0.5, rh=60, met=1, clo=1)) == 2.05

    # test what happens when the cooling effect cannot be calculated
    assert (cooling_effect(tdb=0, tr=80, vr=5, rh=60, met=3, clo=1)) == 0

    assert (
        cooling_effect(tdb=77, tr=77, vr=1.64, rh=50, met=1, clo=0.6, units="IP")
    ) == 3.95
