import math

import numpy as np
import pytest

from pythermalcomfort.models.pmv_ppd import _pmv_ppd_optimized, PMVPPD

from pythermalcomfort.models import pmv_ppd
from tests.conftest import Urls, retrieve_reference_table, validate_result, is_equal


def test_pmv_ppd(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.PMV_PPD.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = pmv_ppd(**inputs)

        validate_result(result, outputs, tolerance)


class TestPmvPpd:

    #  Raises a ValueError if standard is not ISO or ASHRAE
    def test_raises_value_error_if_standard_is_not_iso_or_ashrae(self):
        # Arrange
        tdb = [25]
        tr = [23]
        vr = [0.5]
        rh = [50]
        met = [1.2]
        clo = [0.5]

        # Act & Assert
        with pytest.raises(ValueError):
            pmv_ppd(tdb, tr, vr, rh, met, clo, standard="Invalid")

    #  Returns NaN for invalid input values
    def test_returns_nan_for_invalid_input_values(self):
        # Arrange
        tdb = [25, 50]
        tr = [23, 45]
        vr = [0.5, 3]
        rh = [50, 80]
        met = [1.2, 2.5]
        clo = [0.5, 1.8]

        # Act
        result = pmv_ppd(tdb, tr, vr, rh, met, clo)

        # Assert
        assert math.isnan(result.pmv[1])
        assert math.isnan(result.ppd[1])

        assert (
            round(pmv_ppd(67.28, 67.28, 0.328084, 86, 1.1, 1, units="ip").pmv, 1)
        ) == -0.5

        np.testing.assert_equal(
            np.around(
                pmv_ppd([70, 70], 67.28, 0.328084, 86, 1.1, 1, units="ip").pmv, 1
            ),
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
            ).pmv,
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
            ).pmv,
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
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
            ).pmv,
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        )

        # check results with limit_inputs disabled
        np.testing.assert_equal(
            pmv_ppd(31, 41, 2, 50, 0.7, 2.1, standard="iso", limit_inputs=False),
            PMVPPD(pmv=np.float64(2.4), ppd=np.float64(91.0)),
        )

        np.testing.assert_equal(
            pmv_ppd(41, 41, 2, 50, 0.7, 2.1, standard="ashrae", limit_inputs=False),
            PMVPPD(pmv=np.float64(4.48), ppd=np.float64(100.0)),
        )

    def test_pmv_ppd_optimized(self):
        assert math.isclose(
            _pmv_ppd_optimized(25, 25, 0.3, 50, 1.5, 0.7, 0), 0.55, abs_tol=0.01
        )

        np.testing.assert_equal(
            np.around(_pmv_ppd_optimized([25, 25], 25, 0.3, 50, 1.5, 0.7, 0), 2),
            [0.55, 0.55],
        )


class TestPmvPpdOptimized:

    #  The function returns the correct PMV value for typical input values.
    def test_pmv_typical_input(self):
        # Typical input values
        tdb = 25
        tr = 23
        vr = 0.1
        rh = 50
        met = 1.2
        clo = 0.5
        wme = 0

        expected_pmv = -0.197

        assert math.isclose(
            _pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme),
            expected_pmv,
            abs_tol=0.01,
        )

    #  The function returns the correct PMV value for extreme input values.
    def test_pmv_extreme_input(self):
        # Extreme input values
        tdb = 35
        tr = 45
        vr = 2
        rh = 10
        met = 2.5
        clo = 1.5
        wme = 1

        expected_pmv = 1.86

        assert math.isclose(
            _pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme),
            expected_pmv,
            abs_tol=0.01,
        )

    #  The function returns NaN if any of the input values are NaN.
    def test_nan_input_values(self):
        # Input values with NaN
        tdb = float("nan")
        tr = 23
        vr = 0.1
        rh = 50
        met = 1.2
        clo = 0.5
        wme = 0

        assert math.isnan(_pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme))

    #  The function returns NaN if any of the input values are infinite.
    def test_infinite_input_values(self):
        # Input values with infinity
        tdb = float("inf")
        tr = 23
        vr = 0.1
        rh = 50
        met = 1.2
        clo = 0.5
        wme = 0

        assert math.isnan(_pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme))

    #  Returns a dictionary with 'pmv' and 'ppd' keys
    def test_check_wrong_input(self):
        with pytest.raises(TypeError):
            pmv_ppd(25, 25, 0.1, 50, 1.1, 0.5, stardard="random")
