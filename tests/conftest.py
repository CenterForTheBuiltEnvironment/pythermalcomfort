import json
from enum import Enum

import numpy as np
import pytest
import requests

# The conftest.py file serves as a means of providing fixtures for an entire directory.
# Fixtures defined in a conftest.py can be used by any test in that package
# without needing to import them (pytest will automatically discover them).


unit_test_data_prefix = "https://raw.githubusercontent.com/FedericoTartarini/validation-data-comfort-models/main/"


class Urls(Enum):
    ADAPTIVE_EN = "ts_adaptive_en.json"
    ADAPTIVE_ASHRAE = "ts_adaptive_ashrae.json"
    A_PMV = "ts_a_pmv.json"
    TWO_NODES = "ts_two_nodes_gagge.json"
    SOLAR_GAIN = "ts_solar_gain.json"
    ANKLE_DRAFT = "ts_ankle_draft.json"
    PHS = "ts_phs.json"
    E_PMV = "ts_e_pmv.json"
    AT = "ts_at.json"
    ATHB = "ts_athb.json"
    CLO_TOUT = "ts_clo_tout.json"
    COOLING_EFFECT = "ts_cooling_effect.json"
    VERTICAL_TMP_GRAD_PPD = "ts_vertical_tmp_grad_ppd.json"
    WBGT = "ts_wbgt.json"
    HEAT_INDEX = "ts_heat_index.json"
    NET = "ts_net.json"
    PMV_PPD = "ts_pmv_ppd.json"
    PMV = "ts_pmv.json"
    SET = "ts_set.json"
    HUMIDEX = "ts_humidex.json"
    USE_FANS_HEATWAVES = "ts_use_fans_heatwaves.json"
    UTCI = "ts_utci.json"
    WIND_CHILL = "ts_wind_chill.json"
    PET_STEADY = "ts_pet_steady.json"
    DISCOMFORT_INDEX = "ts_discomfort_index.json"


@pytest.fixture
def get_test_url():
    def _get_test_url(model_name):
        try:
            return unit_test_data_prefix + Urls[model_name.upper()].value
        except KeyError:
            return ""

    return _get_test_url


@pytest.fixture
def retrieve_data():
    def _retrieve_data(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return json.loads(response.text)
            else:
                response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
        return None

    return _retrieve_data


def is_equal(a, b, tolerance=1e-6):
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray):
            b = np.array(b, dtype=a.dtype)
        if a.dtype.kind in "UOS":  # U = unicode, O = objects, S = string
            return np.array_equal(a, b)
        else:
            b = np.where(b is None, np.nan, b)  # Replace None with np.nan
            # Return True if arrays are close enough, including handling of NaN values
            return np.allclose(a, b, atol=tolerance, equal_nan=True)
    elif (a is None and np.isnan(b)) or (b is None and np.isnan(a)):
        return True
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # Compare scalar values with tolerance
        return np.isclose(a, b, atol=tolerance)
    else:
        return a == b


def retrieve_reference_table(get_test_url, retrieve_data, url_name):
    reference_table = retrieve_data(get_test_url(url_name))
    if reference_table is None:
        pytest.fail(f"Failed to retrieve reference table for {url_name.lower()}")
    return reference_table


def validate_result(result, expected_output, tolerance: dict):
    """Parameters
    ----------
    result this is the result of the function that is being tested
    expected_output this is the expected output of the function that is being tested
    tolerance this is the tolerance that is used to compare the result with the
    expected output

    Returns
    -------
    None

    """
    for key in expected_output:
        _expected_output = expected_output[key]

        # some functions return a dictionary or class with multiple values while
        # others return a single value
        # todo remove this once all the functions return a dictionary or class
        try:
            _result = result[key]
        except (IndexError, KeyError, TypeError):
            _result = result

        # if the key is not in the tolerance dictionary we set the tolerance to 1e-6
        if key in tolerance:
            _tolerance = tolerance[key]
        else:
            _tolerance = 1e-6

        try:
            assert is_equal(_result, _expected_output, _tolerance)
        except Exception as e:
            print(f"Expected {_expected_output}, got {_result}\nError: {str(e)}")
            raise
