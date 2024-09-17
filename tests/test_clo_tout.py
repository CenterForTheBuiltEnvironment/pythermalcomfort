import numpy as np
from pythermalcomfort.models import clo_tout

def test_clo_tout(get_clo_tout_url, retrieve_data, is_equal):
    
    reference_table = retrieve_data(get_clo_tout_url)
    
    
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        expected_output = entry["outputs"]["clo_tout"]
        
        
        result = clo_tout(
            tout=inputs["tout"],
            units=inputs.get("units", "SI")  
        )
        
        
        try:
            if isinstance(expected_output, list):
                np.testing.assert_equal(result, expected_output)
            else:
                assert is_equal(result, expected_output)
        except AssertionError as e:
            print(f"Assertion failed for clo_tout. Expected {expected_output}, got {result}, inputs={inputs}\nError: {str(e)}")
            raise