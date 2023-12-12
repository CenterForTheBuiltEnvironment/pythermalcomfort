from pythermalcomfort.models import humidex
import pytest


class TestHumidex:

    #  Calculate humidex for a given dry bulb air temperature and relative humidity
    def test_calculate_humidex(self):
        result = humidex(25, 50)
        assert result["humidex"] == pytest.approx(28.2, abs=0.1)

    #  Return the humidex and discomfort level for a given dry bulb air temperature and relative humidity
    def test_return_humidex_and_discomfort(self):
        result = humidex(25, 50)
        assert "humidex" in result
        assert "discomfort" in result

    #  Round the output humidex value if the 'round' parameter is set to True
    def test_round_output_humidex(self):
        result = humidex(25, 50, round=True)
        assert isinstance(result["humidex"], int) or isinstance(
            result["humidex"], float
        )

    #  Return an error if the input dry bulb air temperature is not a float
    def test_input_temperature_not_float(self):
        with pytest.raises(TypeError):
            humidex("25", 50)

    #  Return an error if the input relative humidity is not a float
    def test_input_humidity_not_float(self):
        with pytest.raises(TypeError):
            humidex(25, "50")

    #  Return "Evident discomfort" if the calculated humidex is between 35 and 40
    def test_humidex_evident_discomfort(self):
        result = humidex(10, 50)
        assert result["discomfort"] == "Little or no discomfort"

        result = humidex(28, 50)
        assert result["discomfort"] == "Noticeable discomfort"

        result = humidex(30, 50)
        assert result["discomfort"] == "Evident discomfort"

        result = humidex(35, 50)
        assert result["discomfort"] == "Intense discomfort; avoid exertion"

        result = humidex(40, 50)
        assert result["discomfort"] == "Heat stroke probable"

    #  Return an error if the input relative humidity is greater than 100%
    def test_humidex_invalid_rh(self):
        with pytest.raises(ValueError):
            humidex(tdb=25, rh=110)

    #  Return an error if the input relative humidity is less than 0%
    def test_input_relative_humidity_less_than_zero(self):
        with pytest.raises(ValueError):
            humidex(25, -10)

    #  Return "Little or no discomfort" if the calculated humidex is exactly 30
    def test_humidex_discomfort_30(self):
        result = humidex(25, 50)
        assert result["discomfort"] == "Little or no discomfort"

    #  Return "Dangerous discomfort" if the calculated humidex is exactly 54
    def test_humidex_dangerous_discomfort(self):
        result = humidex(30, 100)
        assert result["discomfort"] == "Dangerous discomfort"

    def test_humidex(self):
        assert humidex(25, 50) == {
            "humidex": 28.2,
            "discomfort": "Little or no discomfort",
        }
        assert humidex(30, 80) == {
            "humidex": 43.3,
            "discomfort": "Intense discomfort; avoid exertion",
        }
        assert humidex(31.6, 57.1) == {
            "humidex": 40.8,
            "discomfort": "Intense discomfort; avoid exertion",
        }
