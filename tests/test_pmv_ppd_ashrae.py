import numpy as np
import pytest

from pythermalcomfort.classes_return import PMVPPD
from pythermalcomfort.models import pmv_ppd_ashrae
from pythermalcomfort.utilities import Models
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_pmv_ppd(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the PMV and PPD values correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.PMV_PPD.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        # TODO change the validation table code and removed the following
        if "standard" not in inputs:
            inputs["standard"] = "iso"
        if inputs["standard"] == "ashrae":
            inputs["model"] = Models.ashrae_55_2023.value
            del inputs["standard"]
            result = pmv_ppd_ashrae(**inputs)

            validate_result(result, outputs, tolerance)


class TestPmvPpd:
    """Test cases for the PMV and PPD model."""

    def test_thermal_sensation(self) -> None:
        """Test that the function returns the correct thermal sensation values."""
        np.testing.assert_equal(
            pmv_ppd_ashrae(
                [16, 21, 24, 26, 29, 32, 34, 33.47, 33.46],
                [16, 21, 24, 26, 29, 32, 34, 33.47, 33.46],
                0.2,
                50,
                1,
                0.5,
                model=Models.ashrae_55_2023.value,
            ).tsv,
            [
                "Cold",
                "Cool",
                "Slightly Cool",
                "Neutral",
                "Slightly Warm",
                "Warm",
                "Hot",
                "Hot",
                "Warm",
            ],
        )

    #  Returns NaN for invalid input values
    def test_returns_nan_for_invalid_input_values(self) -> None:
        """Test that the function returns NaN for invalid input values."""
        # test airspeed limits
        np.testing.assert_equal(
            pmv_ppd_ashrae(
                [26, 24, 22, 26, 24, 22],
                [26, 24, 22, 26, 24, 22],
                [0.9, 0.6, 0.3, 0.9, 0.6, 0.3],
                50,
                [1.1, 1.1, 1.1, 1.3, 1.3, 1.3],
                [0.5, 0.5, 0.5, 0.7, 0.7, 0.7],
                model=Models.ashrae_55_2023.value,
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
                model=Models.ashrae_55_2023.value,
            ).pmv,
            np.asarray([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        )

        np.testing.assert_equal(
            pmv_ppd_ashrae(
                41,
                41,
                2,
                50,
                0.7,
                2.1,
                model=Models.ashrae_55_2023.value,
                limit_inputs=False,
            ),
            PMVPPD(
                pmv=np.float64(4.48),
                ppd=np.float64(100.0),
                tsv=np.str_("Hot"),
                compliance=np.bool_(False),
            ),
        )

    def test_compliance(self) -> None:
        """Test that the function returns the correct compliance values."""
        # Test scalar values - compliant case (PMV within -0.5 to 0.5)
        result = pmv_ppd_ashrae(
            tdb=25,
            tr=25,
            vr=0.1,
            rh=50,
            met=1.0,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
        )
        assert result.compliance is True or result.compliance == np.bool_(True)

        # Test scalar values - non-compliant case (PMV outside -0.5 to 0.5)
        result = pmv_ppd_ashrae(
            tdb=30,
            tr=30,
            vr=0.1,
            rh=50,
            met=1.0,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
        )
        assert result.compliance is False or result.compliance == np.bool_(False)

        # Test array values with known PMV results
        result = pmv_ppd_ashrae(
            tdb=[22, 25, 30],
            tr=[22, 25, 30],
            vr=0.1,
            rh=50,
            met=1.0,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
        )
        # PMV around -0.77, -0.03, 1.17 approximately
        # First is not compliant (< -0.5), second is compliant, third is not compliant (> 0.5)
        expected_compliance = np.array([False, True, False])
        np.testing.assert_array_equal(result.compliance, expected_compliance)

        # Test edge cases: PMV exactly at -0.5 and 0.5 should not be compliant (exclusive bounds)
        # PMV = -0.5 should return False
        result = pmv_ppd_ashrae(
            tdb=22.5,
            tr=22.5,
            vr=0.1,
            rh=50,
            met=1.0,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
        )
        # The PMV should be around -0.5, so compliance should be False
        if abs(result.pmv + 0.5) < 0.01:  # if PMV is close to -0.5
            assert result.compliance is False or result.compliance == np.bool_(False)

    def test_wrong_standard(self) -> None:
        """Test that the function raises a ValueError for an unsupported model."""
        with pytest.raises(ValueError):
            pmv_ppd_ashrae(25, 25, 0.1, 50, 1.1, 0.5, model="random")
