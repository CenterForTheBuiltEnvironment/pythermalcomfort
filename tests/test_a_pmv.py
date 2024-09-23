from pythermalcomfort.models import a_pmv

def test_a_pmv(get_a_pmv_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_a_pmv_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = a_pmv(**inputs)
        try:
            for key in outputs:
              assert is_equal(result, outputs[key], tolerance.get(key, 1e-6))
        except AssertionError as e:
            print(
                f"Assertion failed for a_pmv. Expected {outputs}, got {result}, inputs={inputs}\nError: {str(e)}"
            )
            raise
