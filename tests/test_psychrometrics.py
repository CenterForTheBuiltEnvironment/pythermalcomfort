import numpy as np
import pytest

from pythermalcomfort.utilities import (
    PsychrometricValues,
    dew_point_tmp,
    enthalpy_air,
    mean_radiant_tmp,
    operative_tmp,
    p_sat,
    psy_ta_rh,
    wet_bulb_tmp,
)
from tests.conftest import is_equal


def test_t_dp():
    assert dew_point_tmp(31.6, 59.6) == 22.6
    assert dew_point_tmp(29.3, 75.4) == 24.3
    assert dew_point_tmp(27.1, 66.4) == 20.2


def test_t_wb():
    assert wet_bulb_tmp(27.1, 66.4) == 22.4
    assert wet_bulb_tmp(25, 50) == 18.0


def test_enthalpy():
    assert is_equal(enthalpy_air(25, 0.01), 50561.25, 0.1)
    assert is_equal(enthalpy_air(27.1, 0.01), 52707.56, 0.1)


def test_psy_ta_rh():
    assert psy_ta_rh(25, 50, p_atm=101325) == PsychrometricValues(
        p_sat=pytest.approx(3169.2, abs=1e-1),
        p_vap=pytest.approx(1584.6, abs=1e-1),
        hr=pytest.approx(0.009881547577511219, abs=1e-7),
        wet_bulb_tmp=18.0,
        dew_point_tmp=13.8,
        h=pytest.approx(50259.79, abs=1e-1),
    )


def test_t_o():
    assert operative_tmp(25, 25, 0.1) == 25
    assert round(operative_tmp(25, 30, 0.3), 2) == 26.83
    assert round(operative_tmp(20, 30, 0.3), 2) == 23.66
    assert operative_tmp(25, 25, 0.1, standard="ASHRAE") == 25
    assert operative_tmp(20, 30, 0.1, standard="ASHRAE") == 25
    assert operative_tmp(20, 30, 0.3, standard="ASHRAE") == 24
    assert operative_tmp(20, 30, 0.7, standard="ASHRAE") == 23


def test_p_sat():
    assert pytest.approx(p_sat(tdb=25), abs=1e-1) == 3169.2
    assert pytest.approx(p_sat(tdb=50), abs=1e-1) == 12349.9


def test_t_mrt():
    np.testing.assert_equal(
        mean_radiant_tmp(
            tg=[53.2, 55, 55],
            tdb=30,
            v=[0.3, 0.3, 0.1],
            d=0.1,
            standard="ISO",
        ),
        [74.8, 77.8, 71.9],
    )
    np.testing.assert_equal(
        mean_radiant_tmp(
            tg=[25.42, 26.42, 26.42, 26.42],
            tdb=26.10,
            v=0.1931,
            d=[0.1, 0.1, 0.5, 0.03],
            standard="Mixed Convection",
        ),
        [24.2, 27.0, np.nan, np.nan],
    )
