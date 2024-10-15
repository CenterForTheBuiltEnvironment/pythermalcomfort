from pythermalcomfort.models import cooling_effect


def test_cooling_effect(get_cooling_effect_url, retrieve_data, is_equal):

    reference_table = retrieve_data(get_cooling_effect_url)
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = cooling_effect(**inputs)
        for key in outputs:
          # To determine whether the result is as expected, use np.allclose for arrays and np.isclose for single
          try:
              is_equal(result, outputs[key], tolerance.get("cooling_effect", 1e-6))
          except AssertionError as e:
              print(
                  f"Assertion failed for cooling_effect. Expected {outputs[key]}, got {result}, inputs={inputs}\nError: {str(e)}"
              )
              raise
