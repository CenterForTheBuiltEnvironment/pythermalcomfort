import pytest
import json
import requests
import numpy as np

# The conftest.py file serves as a means of providing fixtures for an entire directory.
# Fixtures defined in a conftest.py can be used by any test in that package
# without needing to import them (pytest will automatically discover them).


unit_test_data_prefix = "https://raw.githubusercontent.com/TwinGan/validation-data-comfort-models/release_v1.0/"
test_adaptive_en_url = unit_test_data_prefix + "ts_adaptive_en.json"
test_adaptive_ashrae_url = unit_test_data_prefix + "ts_adaptive_ashrae.json"
test_a_pmv_url = unit_test_data_prefix + "ts_a_pmv.json"
test_two_nodes_url = unit_test_data_prefix + "ts_two_nodes.json"
test_solar_gain_url = unit_test_data_prefix + "ts_solar_gain.json"
test_ankle_draft_url = unit_test_data_prefix + "ts_ankle_draft.json"
test_phs_url = unit_test_data_prefix + "ts_phs.json"
test_e_pmv_url = unit_test_data_prefix + "ts_e_pmv.json"
test_at_url = unit_test_data_prefix + "ts_at.json"
test_athb_url = unit_test_data_prefix + "ts_athb.json"
test_clo_tout_url = unit_test_data_prefix + "ts_clo_tout.json"
test_cooling_effect_url = unit_test_data_prefix + "ts_cooling_effect.json"
test_vertical_tmp_grad_ppd_url = unit_test_data_prefix + "ts_vertical_tmp_grad_ppd.json"


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

@pytest.fixture
def is_equal():
    def compare(a, b):
        if isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray):
                b = np.array(b, dtype=float)
            b = np.where(b == None, np.nan, b)  # Replace None with np.nan
            # Return True if arrays are close enough, including handling of NaN values
            return np.allclose(a, b, equal_nan=True)
        elif (a is None and np.isnan(b)) or (b is None and np.isnan(a)):
            return True
        else:
            return a == b
    return compare

@pytest.fixture
def get_adaptive_en_url():
    return test_adaptive_en_url


@pytest.fixture
def get_adaptive_ashrae_url():
    return test_adaptive_ashrae_url


@pytest.fixture
def get_a_pmv_url():
    return test_a_pmv_url


@pytest.fixture
def get_two_nodes_url():
    return test_two_nodes_url


@pytest.fixture
def get_solar_gain_url():
    return test_solar_gain_url


@pytest.fixture
def get_ankle_draft_url():
    return test_ankle_draft_url


@pytest.fixture
def get_phs_url():
    return test_phs_url

@pytest.fixture
def get_vertical_tmp_grad_ppd_url():
    return test_vertical_tmp_grad_ppd_url

@pytest.fixture
def get_e_pmv_url():
    return test_e_pmv_url

@pytest.fixture
def get_at_url():
    return test_at_url

@pytest.fixture
def get_athb_url():
    return test_athb_url

@pytest.fixture
def get_clo_tout_url():
    return test_clo_tout_url

@pytest.fixture
def get_cooling_effect_url():
    return test_cooling_effect_url

