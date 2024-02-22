import numpy as np

from pythermalcomfort.psychrometrics import (
    t_dp,
    t_wb,
    enthalpy,
    psy_ta_rh,
    p_sat,
    t_mrt,
    t_o,
    PsychrometricValues,
)


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
    assert psy_ta_rh(25, 50, p_atm=101325) == PsychrometricValues(
        p_sat=3169.2,
        p_vap=1584.6,
        hr=0.009881547577511219,
        t_wb=18.0,
        t_dp=13.8,
        h=50259.66,
    )


def test_t_o():
    assert t_o(25, 25, 0.1) == 25
    assert round(t_o(25, 30, 0.3), 2) == 26.83
    assert round(t_o(20, 30, 0.3), 2) == 23.66
    assert t_o(25, 25, 0.1, standard="ASHRAE") == 25
    assert t_o(20, 30, 0.1, standard="ASHRAE") == 25
    assert t_o(20, 30, 0.3, standard="ASHRAE") == 24
    assert t_o(20, 30, 0.7, standard="ASHRAE") == 23


def test_p_sat():
    assert (p_sat(tdb=25)) == 3169.2
    assert (p_sat(tdb=50)) == 12349.9


def test_t_mrt():
    np.testing.assert_equal(
        t_mrt(
            tg=[53.2, 55],
            tdb=30,
            v=0.3,
            d=0.1,
            standard="ISO",
        ),
        [74.8, 77.8],
    )
    np.testing.assert_equal(
        t_mrt(
            tg=[25.42, 26.42, 26.42, 26.42],
            tdb=26.10,
            v=0.1931,
            d=[0.1, 0.1, 0.5, 0.03],
            standard="Mixed Convection",
        ),
        [24.2, 27.0, np.nan, np.nan],
    )
