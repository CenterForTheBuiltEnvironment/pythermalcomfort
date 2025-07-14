import pytest

from pythermalcomfort.models import phs
from pythermalcomfort.utilities import met_to_w_m2
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_phs(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the Predicted Heat Strain (PHS) correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.PHS.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        inputs["model"] = "7933-2004"
        outputs = entry["outputs"]
        result = phs(**inputs)

        validate_result(result, outputs, tolerance)


weight = 75
height = 1.8
a_dubois = 0.202 * (weight**0.425) * (height**0.725)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (
            {
                "tdb": 40,
                "tr": 40,
                "rh": 35,
                "v": 0.3,
                "met": 300 / met_to_w_m2 / a_dubois,
                "clo": 0.5,
                "posture": "standing",
                "wme": 0,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 100,
                "round_output": False,
            },
            {
                "water_loss": 6538,
                "t_cr": 37.6,
                "d_lim_loss_95": 280,
            },
        ),
        (
            {
                "tdb": 35,
                "tr": 35,
                "rh": 60,
                "v": 0.1,
                "met": 300 / met_to_w_m2 / a_dubois,
                "clo": 0.5,
                "posture": "standing",
                "wme": 0,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 0,
                "round_output": False,
            },
            {
                "water_loss": 6345,
                "t_cr": 40.8,
                "d_lim_loss_95": 250,
                "d_lim_t_re": 62,
            },
        ),
        pytest.param(
            {
                "tdb": 30,
                "tr": 54.2,
                "rh": 35,
                "v": 0.1,
                "met": 300 / met_to_w_m2 / a_dubois,
                "clo": 0.8,
                "posture": "standing",
                "wme": 0,
                "a_p": 0.3,
                "f_r": 0.85,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 0,
                "round_output": False,
            },
            {
                "water_loss": 6419,
                "t_cr": 38.7,
                "d_lim_loss_95": 280,
                "d_lim_t_re": 149,
            },
            id="high-radiant-temp",
            marks=pytest.mark.xfail(
                reason="Known discrepancy t_cr and d_lim_t_re",
                strict=True,
            ),
        ),
        (
            {
                "tdb": 30,
                "tr": 30,
                "rh": 45,
                "v": 1.0,
                "met": 450 / met_to_w_m2 / a_dubois,
                "clo": 0.5,
                "posture": "standing",
                "wme": 0,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 0,
                "round_output": False,
            },
            {
                "water_loss": 4593,
                "t_cr": 38.0,
                "d_lim_loss_95": 400,
            },
        ),
        (
            {
                "tdb": 35,
                "tr": 74.6,
                "rh": 30,
                "v": 1.0,
                "met": 250 / met_to_w_m2 / a_dubois,
                "clo": 1,
                "posture": "sitting",
                "wme": 0,
                "a_p": 0.2,
                "f_r": 0.85,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 100,
                "round_output": False,
            },
            {
                "water_loss": 5813,
                "t_cr": 37.5,
                "d_lim_loss_95": 310,
            },
        ),
        pytest.param(
            {
                "tdb": [35, 35],
                "tr": 74.6,
                "rh": 30,
                "v": 1.0,
                "met": 250 / met_to_w_m2 / a_dubois,
                "clo": 1,
                "posture": "sitting",
                "wme": 0,
                "a_p": 0.2,
                "f_r": 0.85,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 100,
                "round_output": False,
            },
            {
                "water_loss": [5813, 5813],
                "t_cr": [37.5, 37.5],
                "d_lim_loss_95": [310, 310],
            },
            id="array-input",
        ),
    ],
)
def test_2023_standard(inputs, expected) -> None:
    """Test the 2023 PHS model with various inputs."""
    result = phs(**inputs)
    assert result.water_loss == pytest.approx(expected["water_loss"], rel=0.025)
    assert result.t_cr == pytest.approx(expected["t_cr"], abs=0.3)
    assert result.d_lim_loss_95 == pytest.approx(expected["d_lim_loss_95"], abs=10)
    if "d_lim_t_re" in expected:
        assert result.d_lim_t_re == pytest.approx(expected["d_lim_t_re"], abs=10)


def test_value_acclimatized() -> None:
    """Test that the function raises ValueError for invalid acclimatized values."""
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            acclimatized=101,
        )

    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            acclimatized=-1,
        )


def test_value_weight() -> None:
    """Test that the function raises a ValueError for invalid weight values."""
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            weight=1001,
        )

    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            weight=0,
        )


def test_value_drink() -> None:
    """Test that drink input is within valid range."""
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            drink=0.5,
        )
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            drink=2,
        )
