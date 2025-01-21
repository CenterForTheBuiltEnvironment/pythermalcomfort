import math

import numpy as np
import pytest

from pythermalcomfort.classes_return import PMVPPD
from pythermalcomfort.models.pmv_ppd_iso import pmv_ppd_iso
from pythermalcomfort.utilities import Models, Units
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_pmv_ppd(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.PMV_PPD.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        # todo change the validation table code and removed the following
        if "standard" not in inputs.keys():
            inputs["standard"] = "iso"
        if inputs["standard"] == "iso":
            inputs["model"] = Models.iso_7730_2005.value
            del inputs["standard"]
            result = pmv_ppd_iso(**inputs)

            validate_result(result, outputs, tolerance)


class TestPmvPpd:
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
        result = pmv_ppd_iso(
            tdb, tr, vr, rh, met, clo, model=Models.iso_7730_2005.value
        )

        # Assert
        assert math.isnan(result.pmv[1])
        assert math.isnan(result.ppd[1])

        assert (
            round(
                pmv_ppd_iso(
                    67.28,
                    67.28,
                    0.328084,
                    86,
                    1.1,
                    1,
                    units=Units.IP.value,
                    model=Models.iso_7730_2005.value,
                ).pmv,
                1,
            )
        ) == -0.5

        np.testing.assert_equal(
            np.around(
                pmv_ppd_iso(
                    [70, 70],
                    67.28,
                    0.328084,
                    86,
                    1.1,
                    1,
                    units=Units.IP.value,
                    model=Models.iso_7730_2005.value,
                ).pmv,
                1,
            ),
            [-0.3, -0.3],
        )

        # checking that returns np.nan when outside standard applicability limits
        np.testing.assert_equal(
            pmv_ppd_iso(
                [31, 20, 20, 20, 20, 30],
                [20, 41, 20, 20, 20, 20],
                [0.1, 0.1, 2, 0.1, 0.1, 0.1],
                50,
                [1.1, 1.1, 1.1, 0.7, 1.1, 4.1],
                [0.5, 0.5, 0.5, 0.5, 2.1, 0.1],
                model=Models.iso_7730_2005.value,
            ).pmv,
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        )

        # check results with limit_inputs disabled
        np.testing.assert_equal(
            pmv_ppd_iso(
                31,
                41,
                2,
                50,
                0.7,
                2.1,
                model=Models.iso_7730_2005.value,
                limit_inputs=False,
            ),
            PMVPPD(pmv=np.float64(2.4), ppd=np.float64(91.0), tsv="Warm"),
        )

    def test_wrong_standard(self):
        with pytest.raises(ValueError):
            pmv_ppd_iso(25, 25, 0.1, 50, 1.1, 0.5, model="random")

    def test_no_rounding(self):
        np.isclose(
            pmv_ppd_iso(
                25,
                25,
                0.1,
                50,
                1.1,
                0.5,
                round_output=False,
                model=Models.iso_7730_2005.value,
            ).pmv,
            -0.13201636,
            atol=0.01,
        )
