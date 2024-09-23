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
test_wbgt_url = unit_test_data_prefix + "ts_wbgt.json"
test_heat_index_url = unit_test_data_prefix + "ts_heat_index.json"
test_net_url = unit_test_data_prefix + "ts_net.json"
test_pmv_ppd_url = unit_test_data_prefix + "ts_pmv_ppd.json"
test_pmv_url = unit_test_data_prefix + "ts_pmv.json"
test_set_url = unit_test_data_prefix + "ts_set.json"
test_humidex_url = unit_test_data_prefix + "ts_humidex.json"
test_use_fans_heatwaves_url = unit_test_data_prefix + "ts_use_fans_heatwaves.json"
test_utci_url = unit_test_data_prefix + "ts_utci.json"
test_wind_chill_url = unit_test_data_prefix + "ts_wind_chill.json"
test_pet_steady_url = unit_test_data_prefix + "ts_pet_steady.json"
test_discomfort_index_url = unit_test_data_prefix + "ts_discomfort_index.json"

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
    def compare(a, b, tolerance = 1e-6):
        if isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray):
                b = np.array(b, dtype=a.dtype)
            if a.dtype.kind in "UOS":  # U = unicode, O = objects, S = string
                return np.array_equal(a, b)
            else:
              b = np.where(b == None, np.nan, b)  # Replace None with np.nan
              # Return True if arrays are close enough, including handling of NaN values
              return np.allclose(a, b, atol = tolerance, equal_nan=True)
        elif (a is None and np.isnan(b)) or (b is None and np.isnan(a)):
            return True
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            # Compare scalar values with tolerance
            return np.isclose(a, b, atol=tolerance)
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
def get_humidex_url():
    return test_humidex_url


@pytest.fixture
def get_wbgt_url():
    return test_wbgt_url


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


@pytest.fixture
def get_heat_index_url():
    return test_heat_index_url


@pytest.fixture
def get_net_url():
    return test_net_url


@pytest.fixture
def get_pmv_ppd_url():
    return test_pmv_ppd_url


@pytest.fixture
def get_pmv_url():
    return test_pmv_url


@pytest.fixture
def get_set_url():
    return test_set_url


@pytest.fixture
def get_use_fans_heatwaves_url():
    return test_use_fans_heatwaves_url


@pytest.fixture
def get_utci_url():
    return test_utci_url


@pytest.fixture
def get_wind_chill_url():
    return test_wind_chill_url


@pytest.fixture
def get_pet_steady_url():
    return test_pet_steady_url

@pytest.fixture
def get_discomfort_index_url():
    return test_discomfort_index_url
