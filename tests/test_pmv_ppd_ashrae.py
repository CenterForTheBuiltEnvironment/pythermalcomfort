import numpy as np
import pytest

from pythermalcomfort.classes_return import PMVPPD
from pythermalcomfort.models import pmv_ppd_ashrae
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
        if inputs["standard"] == "ashrae":
            inputs["model"] = "55-2023"
            del inputs["standard"]
            result = pmv_ppd_ashrae(**inputs)

            validate_result(result, outputs, tolerance)


class TestPmvPpd:
    #  Returns NaN for invalid input values
    def test_returns_nan_for_invalid_input_values(self):
        # test airspeed limits
        np.testing.assert_equal(
            pmv_ppd_ashrae(
                [26, 24, 22, 26, 24, 22],
                [26, 24, 22, 26, 24, 22],
                [0.9, 0.6, 0.3, 0.9, 0.6, 0.3],
                50,
                [1.1, 1.1, 1.1, 1.3, 1.3, 1.3],
                [0.5, 0.5, 0.5, 0.7, 0.7, 0.7],
                model="55-2023",
                airspeed_control=False,
            ).pmv,
            [np.nan, np.nan, np.nan, -0.14, -0.43, -0.57],
        )

        np.testing.assert_equal(
            pmv_ppd_ashrae(
                [41, 20, 20, 20, 20, 39],
                [20, 41, 20, 20, 20, 39],
                [0.1, 0.1, 2.1, 0.1, 0.1, 0.1],
                50,
                [1.1, 1.1, 1.1, 0.7, 1.1, 3.9],
                [0.5, 0.5, 0.5, 0.5, 2.1, 1.9],
                model="55-2023",
            ).pmv,
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        )

        np.testing.assert_equal(
            pmv_ppd_ashrae(
                41, 41, 2, 50, 0.7, 2.1, model="55-2023", limit_inputs=False
            ),
            PMVPPD(pmv=np.float64(4.48), ppd=np.float64(100.0)),
        )

    def test_wrong_standard(self):
        with pytest.raises(ValueError):
            pmv_ppd_ashrae(25, 25, 0.1, 50, 1.1, 0.5, model="random")
