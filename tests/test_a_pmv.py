from pythermalcomfort.models import a_pmv


def test_a_pmv(get_a_pmv_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_a_pmv_url)
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = a_pmv(
            inputs["tdb"],
            inputs["tr"],
            inputs["vr"],
            inputs["rh"],
            inputs["met"],
            inputs["clo"],
            inputs["a_coefficient"],
        )
        try:
            assert is_equal(result, outputs["a_pmv"])
        except AssertionError as e:
            print(
                f"Assertion failed for a_pmv. Expected {outputs}, got {result}, inputs={inputs}\nError: {str(e)}"
            )
            raise
