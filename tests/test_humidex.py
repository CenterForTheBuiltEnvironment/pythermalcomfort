import numpy as np
import pytest
from pythermalcomfort.models import humidex


def test_humidex(get_humidex_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_humidex_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = humidex(**inputs)
        try:
            for key in outputs:
                assert is_equal(result[key], outputs[key], tolerance.get(key, 1e-6))
        except AssertionError as e:
            print(
                f"Assertion failed for a_pmv. Expected {outputs}, got {result}, inputs={inputs}\nError: {str(e)}"
            )
            raise

    with pytest.raises(TypeError):
        humidex("25", 50)

    with pytest.raises(TypeError):
        humidex(25, "50")

    with pytest.raises(ValueError):
        humidex(tdb=25, rh=110)

    with pytest.raises(ValueError):
        humidex(25, -10)
