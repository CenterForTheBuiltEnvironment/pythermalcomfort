from pythermalcomfort.models import at


def test_at(get_at_url, retrieve_data, is_equal):

    reference_table = retrieve_data(get_at_url)

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        expected_output = entry["outputs"]["at"]

        result = at(
            tdb=inputs["tdb"],
            rh=inputs["rh"],
            v=inputs["v"],
            q=inputs.get("q", None),  # q is elective
        )

        try:
            assert is_equal(result, expected_output)
        except AssertionError as e:
            print(
                f"Assertion failed for at. Expected {expected_output}, got {result}, inputs={inputs}\nError: {str(e)}"
            )
            raise
