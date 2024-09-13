import pytest
from pythermalcomfort.models import ankle_draft

def test_ankle_draft(get_ankle_draft_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_ankle_draft_url)
    
    if reference_table is None:
        pytest.skip("Failed to retrieve test data")

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        
        # Check if we're dealing with multiple inputs (list)
        if isinstance(inputs["tdb"], list):
            for i in range(len(inputs["tdb"])):
                mapped_inputs = {
                    "tdb": inputs["tdb"][i],
                    "tr": inputs["tr"][i],
                    "vr": inputs["v"][i],
                    "rh": inputs["rh"][i],
                    "met": inputs["met"][i],
                    "clo": inputs["clo"][i],
                    "v_ankle": inputs["v_ankle"][i],
                    "units": inputs["units"][i] if isinstance(inputs["units"], list) else inputs["units"]
                }
                result = ankle_draft(**mapped_inputs)
                for key in outputs:
                    try:
                        assert is_equal(result[key], outputs[key][i])
                    except AssertionError as e:
                        print(
                            f"Assertion failed for {key}. Expected {outputs[key][i]}, got {result[key]}, inputs={mapped_inputs}\nError: {str(e)}"
                        )
                        raise
        else:
            # Single input case
            mapped_inputs = {
                "tdb": inputs["tdb"],
                "tr": inputs["tr"],
                "vr": inputs["v"],
                "rh": inputs["rh"],
                "met": inputs["met"],
                "clo": inputs["clo"],
                "v_ankle": inputs["v_ankle"],
                "units": inputs["units"]
            }
            result = ankle_draft(**mapped_inputs)
            for key in outputs:
                try:
                    assert is_equal(result[key], outputs[key])
                except AssertionError as e:
                    print(
                        f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={mapped_inputs}\nError: {str(e)}"
                    )
                    raise

    # Test for ValueError
    with pytest.raises(ValueError):
        ankle_draft(tdb=25, tr=25, vr=0.3, rh=50, met=1.2, clo=0.5, v_ankle=7)