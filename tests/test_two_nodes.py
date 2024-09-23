from pythermalcomfort.models import two_nodes


def test_two_nodes(get_two_nodes_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_two_nodes_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = two_nodes(**inputs)
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result[key], outputs[key], tolerance.get(key, 1e-6))
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise
