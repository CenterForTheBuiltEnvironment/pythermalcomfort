from pythermalcomfort.models import net


def test_net(get_net_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_net_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = net(**inputs)
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result, outputs[key], tolerance.get(key, 1e-6))
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise
