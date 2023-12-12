from pythermalcomfort.models import wc
import pytest


class TestWc:

    #  Calculates the wind chill index (WCI) for given dry bulb air temperature and wind speed
    def test_calculates_wci(self):
        assert wc(tdb=0, v=0.1) == {"wci": 518.6}
        assert wc(tdb=0, v=1.5) == {"wci": 813.5}
        assert wc(tdb=-5, v=5.5) == {"wci": 1255.2}
        assert wc(tdb=-10, v=11) == {"wci": 1631.1}
        assert wc(tdb=-5, v=11) == {"wci": 1441.4}

    #  Returns the WCI value in a dictionary format
    def test_returns_wci_dictionary(self):
        assert isinstance(wc(tdb=0, v=0.1), dict)
        assert isinstance(wc(tdb=0, v=1.5), dict)
        assert isinstance(wc(tdb=-5, v=5.5), dict)
        assert isinstance(wc(tdb=-10, v=11), dict)
        assert isinstance(wc(tdb=-5, v=11), dict)

    #  Rounds the output value if round=True
    def test_rounds_output_value(self):
        assert wc(tdb=0, v=0.1, round=True) == {"wci": 518.6}
        result = wc(tdb=0, v=0.1, round=False)
        assert abs(result["wci"] - 518.587) < 0.01

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
