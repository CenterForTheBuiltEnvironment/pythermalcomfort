import pytest
from pythermalcomfort.models import ankle_draft

def test_ankle_draft(get_ankle_draft_url, retrieve_data):
    test_data = retrieve_data(get_ankle_draft_url)
    
    if test_data is None:
        pytest.skip("Failed to retrieve test data")

    # Tests using data from URL
    for case in test_data.get("data", []):
        inputs = case["inputs"]
        print(case["outputs"])
        expected_output = case["outputs"]["PPD_ad"]
        result = ankle_draft(
            inputs["tdb"], inputs["tr"], inputs["v"], inputs["rh"],
            inputs["met"], inputs["clo"], inputs["v_ankle"], units=inputs["units"]
        )
        assert round(result["PPD_ad"], 1) == round(expected_output, 1)

    # Test for ValueError
    with pytest.raises(ValueError):
        ankle_draft(25, 25, 0.3, 50, 1.2, 0.5, 7)