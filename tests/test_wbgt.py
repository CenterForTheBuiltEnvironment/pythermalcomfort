import pytest
import math

from pythermalcomfort.models import wbgt


class TestWbgt:
    def test_wbgt(self):
        assert math.isclose(wbgt(twb=25, tg=32, tdb=20, with_solar_load=True), 25.9)
        with pytest.raises(ValueError):
            wbgt(twb=25, tg=32, with_solar_load=True)
        # data from Table D.1 ISO 7243
        assert math.isclose(wbgt(twb=17.3, tg=40, round=True), 24.1)
        assert math.isclose(wbgt(twb=21.1, tg=55, round=True), 31.3)
        assert math.isclose(wbgt(twb=16.7, tg=40, round=True), 23.7)

        #  Calculate WBGT with twb and tg

    def test_calculate_wbgt_with_twb_and_tg(self):
        assert math.isclose(wbgt(25, 30), 26.5)

    #  Calculate WBGT with twb, tg, and tdb
    def test_calculate_wbgt_with_twb_tg_and_tdb(self):
        assert math.isclose(wbgt(25, 30, 20), 26.5)

    #  Round WBGT to one decimal place
    def test_round_wbgt_to_one_decimal_place(self):
        assert math.isclose(wbgt(25, 30, round=False), 26.5)

    #  Calculate WBGT with twb and tg set to 0
    def test_calculate_wbgt_with_twb_and_tg_set_to_zero(self):
        assert math.isclose(wbgt(0, 0), 0.0)

    #  Calculate WBGT with tdb set to 0
    def test_calculate_wbgt_with_tdb_set_to_zero(self):
        assert math.isclose(wbgt(25, 30, 0), 26.5)

    #  Calculate WBGT with twb and tg set to None
    def test_calculate_wbgt_with_twb_and_tg_set_to_none(self):
        with pytest.raises(TypeError):
            wbgt(None, None)
