import numpy as np
import pytest
from pythermalcomfort.models import humidex


def test_humidex(get_humidex_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_humidex_url)
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = humidex(**inputs)
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result[key], outputs[key])
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
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
