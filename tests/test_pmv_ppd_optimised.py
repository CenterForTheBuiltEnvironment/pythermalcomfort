import math

import numpy as np

from pythermalcomfort.models._pmv_ppd_optimized import _pmv_ppd_optimized


class TestPmvPpdOptimized:
    #  Returns NaN for invalid input values

    def test_pmv_ppd_optimized(self):
        assert math.isclose(
            _pmv_ppd_optimized(25, 25, 0.3, 50, 1.5, 0.7, 0), 0.55, abs_tol=0.01
        )

        np.testing.assert_equal(
            np.around(_pmv_ppd_optimized([25, 25], 25, 0.3, 50, 1.5, 0.7, 0), 2),
            [0.55, 0.55],
        )

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
