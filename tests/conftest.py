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


# Custom equal method
## Json null equal to np.nan
## np.ndarry compare with list
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


# get test data for adaptove_en()
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

def get_ankle_draft_url():
    return test_ankle_draft_url
