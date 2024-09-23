from pythermalcomfort.models import heat_index


def test_heat_index(get_heat_index_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_heat_index_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = heat_index(
            **inputs
        )
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result, outputs[key], tolerance.get(key, 1e-6))
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result}, inputs={inputs}\nError: {str(e)}"
                )
                raise