from pythermalcomfort.models import pet_steady


def test_pet_steady(get_pet_steady_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_pet_steady_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        expected_output = entry["outputs"]["PET"]

        result = pet_steady(
            tdb=inputs["tdb"],
            tr=inputs["tr"],
            rh=inputs["rh"],
            v=inputs["v"],
            met=inputs["met"],
            clo=inputs["clo"],
        )

        try:
            is_equal(result, expected_output, tolerance.get("PET", 1e-6))
        except AssertionError as e:
            print(
                f"Assertion failed for pet_steady. Expected {expected_output}, got {result}, inputs={inputs}\nError: {str(e)}"
            )
            raise
