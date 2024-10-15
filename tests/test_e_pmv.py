import numpy as np
from pythermalcomfort.models import e_pmv


def test_e_pmv(get_e_pmv_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_e_pmv_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = e_pmv(**inputs)

        for key in outputs:
          try:
              assert is_equal(result, outputs[key], tolerance.get("pmv", 1e-6))
          except AssertionError as e:
              print(
                  f"Assertion failed for e_pmv. Expected {outputs[key]}, got {result}, inputs={inputs}\nError: {str(e)}"
              )
              raise
