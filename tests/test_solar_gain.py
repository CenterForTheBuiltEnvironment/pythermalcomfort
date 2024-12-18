import numpy as np

from pythermalcomfort.models import solar_gain
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_solar_gain(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.SOLAR_GAIN.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = solar_gain(**inputs)

        validate_result(result, outputs, tolerance)


def test_solar_gain_array():
    np.allclose(
        solar_gain(
            sol_altitude=[0, 30],
            sharp=[120, 60],
            sol_radiation_dir=[800, 600],
            sol_transmittance=[0.5, 0.6],
            f_svv=[0.5, 0.4],
            f_bes=[0.5, 0.6],
            asw=0.7,
            posture="sitting",
        ).erf,
        np.array([46.4, 52.8]),
        atol=0.1,
    )
