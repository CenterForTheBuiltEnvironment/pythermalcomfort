from pythermalcomfort.models import wc


def test_wc():
    assert wc(tdb=0, v=0.1) == {"wci": 518.6}
    assert wc(tdb=0, v=1.5) == {"wci": 813.5}
    assert wc(tdb=-5, v=5.5) == {"wci": 1255.2}
    assert wc(tdb=-10, v=11) == {"wci": 1631.1}
    assert wc(tdb=-5, v=11) == {"wci": 1441.4}
