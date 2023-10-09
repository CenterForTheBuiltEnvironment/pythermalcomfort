from pythermalcomfort.models import net


def test_net():
    assert net(37, 100, 0.1) == 37
    assert net(37, 100, 4.5) == 37
    assert net(25, 100, 4.5) == 20
    assert net(25, 100, 0.1) == 25.4
    assert net(40, 48.77, 0.1) == 33.8
    assert net(36, 50.196, 0.1) == 30.9
