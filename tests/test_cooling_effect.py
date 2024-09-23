from pythermalcomfort.models import cooling_effect


def test_cooling_effect(get_cooling_effect_url, retrieve_data, is_equal):

    reference_table = retrieve_data(get_cooling_effect_url)
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        expected_output = entry["outputs"]["cooling_effect"]

        result = cooling_effect(
            tdb=inputs["tdb"],
            tr=inputs["tr"],
            rh=inputs["rh"],
            vr=inputs["vr"],
            met=inputs["met"],
            clo=inputs["clo"],
            units=inputs.get("units", "SI"),
        )

        # To determine whether the result is as expected, use np.allclose for arrays and np.isclose for single
        try:
            is_equal(result, expected_output, tolerance.get("cooling_effect"))
        except AssertionError as e:
            print(
                f"Assertion failed for cooling_effect. Expected {expected_output}, got {result}, inputs={inputs}\nError: {str(e)}"
            )
            raise
