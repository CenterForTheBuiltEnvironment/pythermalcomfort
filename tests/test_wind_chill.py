from pythermalcomfort.models import wc
import pytest


class TestWc:

    #  Calculates the wind chill index (WCI) for given dry bulb air temperature and wind speed
    def test_calculates_wci(self, get_wind_chill_url, retrieve_data, is_equal):
        reference_table = retrieve_data(get_wind_chill_url)
        for entry in reference_table["data"]:
            inputs = entry["inputs"]
            outputs = entry["outputs"]
            result = wc(**inputs)
            for key in outputs:
                # Use the custom is_equal for other types
                try:
                    if(inputs.get("round", True)):
                        assert is_equal(result[key], outputs[key])
                    else:
                        assert abs(result["wci"] - 518.587) < 0.01
                except AssertionError as e:
                    print(
                        f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                    )
                    raise

    #  Returns the WCI value in a dictionary format
    def test_returns_wci_dictionary(self):
        assert isinstance(wc(tdb=0, v=0.1), dict)
        assert isinstance(wc(tdb=0, v=1.5), dict)
        assert isinstance(wc(tdb=-5, v=5.5), dict)
        assert isinstance(wc(tdb=-10, v=11), dict)
        assert isinstance(wc(tdb=-5, v=11), dict)

    #  Raises TypeError if tdb parameter is not provided
    def test_raises_type_error_tdb_not_provided(self):
        with pytest.raises(TypeError):
            wc(v=0.1)

    #  Raises TypeError if v parameter is not provided
    def test_raises_type_error_v_not_provided(self):
        with pytest.raises(TypeError):
            wc(tdb=0)

    #  Raises TypeError if tdb parameter is not a float
    def test_raises_type_error_tdb_not_float(self):
        with pytest.raises(TypeError):
            wc(tdb="0", v=0.1)