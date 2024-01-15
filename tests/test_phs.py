import numpy as np
import pytest

from pythermalcomfort.models import phs


def test_phs_array():
    np.testing.assert_equal(
        phs(tdb=[40, 40], tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2),
        {
            "t_re": [37.5, 37.5],
            "t_sk": [35.3, 35.3],
            "t_cr": [37.5, 37.5],
            "t_cr_eq": [37.1, 37.1],
            "t_sk_t_cr_wg": [0.24, 0.24],
            "d_lim_loss_50": [440.0, 440.0],
            "d_lim_loss_95": [298.0, 298.0],
            "d_lim_t_re": [480, 480],
            "water_loss_watt": [266.1, 266.1],
            "water_loss": [6166.4, 6166.4],
        },
    )


def test_phs():
    assert phs(tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2) == {
        "d_lim_loss_50": 440.0,
        "d_lim_loss_95": 298.0,
        "d_lim_t_re": 480.0,
        "water_loss": 6166.4,
        "t_re": 37.5,
        "t_cr": 37.5,
        "t_sk": 35.3,
        "t_cr_eq": 37.1,
        "t_sk_t_cr_wg": 0.24,
        "water_loss_watt": 266.1,
    }

    assert phs(tdb=35, tr=35, rh=71, v=0.3, met=150, clo=0.5, posture=2) == {
        "d_lim_loss_50": 385.0,
        "d_lim_loss_95": 256.0,
        "d_lim_t_re": 75.0,
        "water_loss": 6934.6,
        "t_re": 39.8,
        "t_cr": 39.7,
        "t_sk": 36.4,
        "t_cr_eq": 37.1,
        "t_sk_t_cr_wg": 0.1,
        "water_loss_watt": 276.9,
    }

    assert phs(tdb=30, tr=50, posture=2, rh=70.65, v=0.3, met=150, clo=0.5) == {
        "d_lim_loss_50": 380.0,
        "d_lim_loss_95": 258.0,
        "d_lim_t_re": 480.0,
        "water_loss": 7166.2,
        "t_re": 37.7,
        "t_cr": 37.7,
        "t_sk": 35.7,
        "t_cr_eq": 37.1,
        "t_sk_t_cr_wg": 0.22,
        "water_loss_watt": 312.5,
    }
    assert phs(
        tdb=28, tr=58, acclimatized=0, posture=2, rh=79.31, v=0.3, met=150, clo=0.5
    ) == {
        "d_lim_loss_50": 466.0,
        "d_lim_loss_95": 314.0,
        "d_lim_t_re": 57.0,
        "water_loss": 5807.3,
        "t_re": 41.2,
        "t_cr": 41.1,
        "t_sk": 37.8,
        "t_cr_eq": 37.1,
        "t_sk_t_cr_wg": 0.1,
        "water_loss_watt": 250.0,
    }
    assert phs(
        tdb=35, tr=35, acclimatized=0, posture=1, rh=53.3, v=1, met=150, clo=0.5
    ) == {
        "d_lim_loss_50": 480.0,
        "d_lim_loss_95": 463.0,
        "d_lim_t_re": 480.0,
        "water_loss": 3891.8,
        "t_re": 37.6,
        "t_cr": 37.5,
        "t_sk": 34.8,
        "t_cr_eq": 37.1,
        "t_sk_t_cr_wg": 0.24,
        "water_loss_watt": 165.7,
    }
    assert phs(tdb=43, tr=43, posture=1, rh=34.7, v=0.3, met=103, clo=0.5) == {
        "d_lim_loss_50": 401.0,
        "d_lim_loss_95": 271.0,
        "d_lim_t_re": 480.0,
        "water_loss": 6765.1,
        "t_re": 37.3,
        "t_cr": 37.2,
        "t_sk": 35.3,
        "t_cr_eq": 37.0,
        "t_sk_t_cr_wg": 0.26,
        "water_loss_watt": 293.6,
    }
    assert phs(
        tdb=35, tr=35, acclimatized=0, posture=2, rh=53.3, v=0.3, met=206, clo=0.5
    ) == {
        "d_lim_loss_50": 372.0,
        "d_lim_loss_95": 247.0,
        "d_lim_t_re": 70.0,
        "water_loss": 7235.9,
        "t_re": 39.2,
        "t_cr": 39.1,
        "t_sk": 36.1,
        "t_cr_eq": 37.3,
        "t_sk_t_cr_wg": 0.1,
        "water_loss_watt": 295.7,
    }
    assert phs(
        tdb=34, tr=34, acclimatized=0, posture=2, rh=56.3, v=0.3, met=150, clo=1
    ) == {
        "d_lim_loss_50": 480.0,
        "d_lim_loss_95": 318.0,
        "d_lim_t_re": 67.0,
        "water_loss": 5547.7,
        "t_re": 41.0,
        "t_cr": 40.9,
        "t_sk": 36.7,
        "t_cr_eq": 37.1,
        "t_sk_t_cr_wg": 0.1,
        "water_loss_watt": 213.9,
    }
    assert phs(tdb=40, tr=40, rh=40.63, v=0.3, met=150, clo=0.4, posture=2) == {
        "d_lim_loss_50": 407.0,
        "d_lim_loss_95": 276.0,
        "d_lim_t_re": 480.0,
        "water_loss": 6683.4,
        "t_re": 37.5,
        "t_cr": 37.4,
        "t_sk": 35.5,
        "t_cr_eq": 37.1,
        "t_sk_t_cr_wg": 0.24,
        "water_loss_watt": 290.4,
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
        "d_lim_loss_50": 480.0,
        "d_lim_loss_95": 339.0,
        "d_lim_t_re": 480.0,
        "water_loss": 5379.1,
        "t_re": 37.6,
        "t_cr": 37.5,
        "t_sk": 35.5,
        "t_cr_eq": 37.1,
        "t_sk_t_cr_wg": 0.24,
        "water_loss_watt": 231.5,
    }


def test_check_standard_compliance():
    # checking that returns correct values when inside standard applicability limits
    np.testing.assert_equal(
        phs(
            [20, 20, 20, 20, 20, 20],
            [20, 20, 20, 20, 20, 20],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [
                50,
                50,
                50,
                50,
                50,
                50,
            ],
            [
                150,
                150,
                150,
                150,
                150,
                150,
            ],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            posture=1,
        )["t_re"],
        [37.4, 37.4, 37.4, 37.4, 37.4, 37.4],
    )

    # same values as above but with one value outside the standard applicability limits
    np.testing.assert_equal(
        phs(
            [51, 20, 20, 35, 20, 20],
            [20, 61, 20, 20, 20, 20],
            [0.1, 0.1, 3.1, 0.1, 0.1, 0.1],
            [
                50,
                50,
                50,
                90,
                50,
                50,
            ],
            [
                150,
                150,
                150,
                150,
                90,
                150,
            ],
            [0.5, 0.5, 0.5, 0.5, 0.5, 1.1],
            posture=1,
        )["t_re"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    )


def test_value_acclimatized():
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=150,
            clo=0.5,
            posture=2,
            acclimatized=101,
        )

    with pytest.raises(ValueError):
        phs(
            tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2, acclimatized=-1
        )


def test_value_weight():
    with pytest.raises(ValueError):
        phs(tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2, weight=1001)

    with pytest.raises(ValueError):
        phs(tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2, weight=0)


def test_value_drink():
    with pytest.raises(ValueError):
        phs(tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2, drink=0.5)
    with pytest.raises(ValueError):
        phs(tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2, drink=2)
