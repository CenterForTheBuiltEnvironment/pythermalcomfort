import pytest
import json
import requests
import numpy as np
# The conftest.py file serves as a means of providing fixtures for an entire directory.
# Fixtures defined in a conftest.py can be used by any test in that package 
# without needing to import them (pytest will automatically discover them).

unit_test_data_prefix = "https://raw.githubusercontent.com/TwinGan/validation-data-comfort-models/release_v1.0/"
test_adaptive_en_url = unit_test_data_prefix + "ts_adaptive_en.json"

def retrieve_data(url):
    try:
      resp = requests.get(url)
      reference_tables = json.loads(resp.text)
      return reference_tables
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

# Custom equal method
## Json null equal to np.nan
@pytest.fixture
def is_equal():
    def compare(a, b):
        if (a is None and np.isnan(b)) or (b is None and np.isnan(a)):
            return True
        return a == b
    return compare

# get test data for adaptove_en()
@pytest.fixture
def get_adaptive_en_test_data():
    return retrieve_data(test_adaptive_en_url)