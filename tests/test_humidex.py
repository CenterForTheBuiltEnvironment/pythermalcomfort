import numpy as np
import pytest
from pythermalcomfort.models import humidex


def test_humidex(get_humidex_url, retrieve_data, is_equal):
    test_data = retrieve_data(get_humidex_url)

    if test_data is None:
        pytest.skip("Failed to retrieve test data")

    for case in test_data.get("data", []):
        inputs = case["inputs"]
        expected_outputs = case["outputs"]

        if isinstance(inputs, list):
            for inp, expected in zip(inputs, expected_outputs):
                result = humidex(inp["tdb"], inp["rh"])
                assert result["discomfort"] == expected["discomfort"]
                if "humidex" in expected:
                    assert result["humidex"] == pytest.approx(
                        expected["humidex"], abs=0.1
                    )
        else:
            if "round" in inputs:
                result = humidex(inputs["tdb"], inputs["rh"], round=inputs["round"])
            else:
                result = humidex(inputs["tdb"], inputs["rh"])

            for key, expected_value in expected_outputs.items():
                if key == "humidex":
                    assert result[key] == pytest.approx(expected_value, abs=0.1)
                else:
                    assert result[key] == expected_value

    with pytest.raises(TypeError):
        humidex("25", 50)

    with pytest.raises(TypeError):
        humidex(25, "50")

    with pytest.raises(ValueError):
        humidex(tdb=25, rh=110)

    with pytest.raises(ValueError):
        humidex(25, -10)
