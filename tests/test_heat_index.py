from pythermalcomfort.models import heat_index


def test_heat_index():
    assert heat_index(25, 50) == 25.9
    assert heat_index(77, 50, units="IP") == 78.6
    assert heat_index(30, 80) == 37.7
    assert heat_index(86, 80, units="IP") == 99.8
