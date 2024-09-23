import numpy as np
import math
import pytest

from pythermalcomfort.models import (
    pmv_ppd,
)
from pythermalcomfort.models.pmv_ppd import _pmv_ppd_optimized


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

    def test_pmv_ppd(self, retrieve_data, get_pmv_pdd_url, is_equal):
        reference_table = retrieve_data(get_pmv_pdd_url)
        tolerance = reference_table["tolerance"]
        for entry in reference_table["data"]:
            inputs = entry["inputs"]
            outputs = entry["outputs"]
            result = pmv_ppd(**inputs)
            for key in outputs:
                # Use the custom is_equal for other types
                try:
                    assert is_equal(result[key], outputs[key], tolerance.get(key, 1e-6))
                except AssertionError as e:
                    print(
                        f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                    )
                    raise

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
            ),
            {
                "pmv": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "ppd": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            },
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
            ),
            {
                "pmv": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "ppd": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            },
        )

        # check results with limit_inputs disabled
        np.testing.assert_equal(
            pmv_ppd(31, 41, 2, 50, 0.7, 2.1, standard="iso", limit_inputs=False),
            {"pmv": 2.4, "ppd": 91.0},
        )

        np.testing.assert_equal(
            pmv_ppd(41, 41, 2, 50, 0.7, 2.1, standard="ashrae", limit_inputs=False),
            {"pmv": 4.48, "ppd": 100.0},
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
