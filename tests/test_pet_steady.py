from pythermalcomfort.models import pet_steady


def test_pet_steady(get_pet_steady_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_pet_steady_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = pet_steady(**inputs)
        for key in outputs:
          try:
              is_equal(result, outputs[key], tolerance.get("PET", 1e-6))
          except AssertionError as e:
              print(
                  f"Assertion failed for pet_steady. Expected {outputs[key]}, got {result}, inputs={inputs}\nError: {str(e)}"
              )
              raise
