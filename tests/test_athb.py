import numpy as np
from pythermalcomfort.models import athb

def test_athb(get_athb_url, retrieve_data, is_equal):
    
    reference_table = retrieve_data(get_athb_url)
    
    
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        expected_output = entry["outputs"]["athb_pmv"]
        
       
        result = athb(
            tdb=inputs["tdb"],
            tr=inputs["tr"],
            vr=inputs["vr"],
            rh=inputs["rh"],
            met=inputs["met"],
            t_running_mean=inputs["t_running_mean"]
        )
        
        try:
            if isinstance(expected_output, list):
                np.testing.assert_equal(result, expected_output)
            else:
                assert is_equal(result, expected_output)
        except AssertionError as e:
            print(f"Assertion failed for athb. Expected {expected_output}, got {result}, inputs={inputs}\nError: {str(e)}")
            raise
