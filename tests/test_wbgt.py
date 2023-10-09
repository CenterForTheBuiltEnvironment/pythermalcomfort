import pytest

from pythermalcomfort.models import wbgt


def test_wbgt():
    assert wbgt(25, 30) == 26.5
    assert wbgt(twb=25, tg=32) == 27.1
    assert wbgt(twb=25, tg=32, tdb=20) == 27.1
    assert wbgt(twb=25, tg=32, tdb=20, with_solar_load=True) == 25.9
    with pytest.raises(ValueError):
        wbgt(twb=25, tg=32, with_solar_load=True)
    # data from Table D.1 ISO 7243
    assert wbgt(twb=17.3, tg=40, round=True) == 24.1
    assert wbgt(twb=21.1, tg=55, round=True) == 31.3
    assert wbgt(twb=16.7, tg=40, round=True) == 23.7
