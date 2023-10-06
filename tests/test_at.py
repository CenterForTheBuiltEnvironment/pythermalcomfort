from pythermalcomfort.models import at


def test_at():
    assert at(tdb=25, rh=30, v=0.1) == 24.1
    assert at(tdb=23, rh=70, v=1) == 24.8
    assert at(tdb=23, rh=70, v=1, q=50) == 28.1
