from pythermalcomfort.cli import main
from pythermalcomfort.psychrometrics import *


def test_main():
    assert main([]) == 0


def test_running_mean_outdoor_temperature():
    assert (running_mean_outdoor_temperature([20, 20], alpha=0.7)) == 20
    assert (running_mean_outdoor_temperature([20, 20], alpha=0.9)) == 20
    assert (running_mean_outdoor_temperature([20, 20, 20, 20], alpha=0.7)) == 20
    assert (running_mean_outdoor_temperature([20, 20, 20, 20], alpha=0.5)) == 20
    assert (running_mean_outdoor_temperature([77, 77, 77, 77, 77, 77, 77], alpha=0.8, units='IP')) == 77
    assert (running_mean_outdoor_temperature([77, 77, 77, 77, 77, 77, 77], alpha=0.8, units='ip')) == 77


def test_ip_units_converter():
    assert (units_converter(ta=77, tr=77, v=3.2, from_units='ip')) == [25.0, 25.0, 0.975312404754648]
    assert (units_converter(pressure=1, area=1 / 0.09, from_units='ip')) == [101325, 1.0322474090590033]


def test_t_globe():
    assert (t_mrt(tg=53.2, ta=30, v=0.3, d=0.1, emissivity=0.95)) == 74.8
    assert (t_mrt(tg=55, ta=30, v=0.3, d=0.1, emissivity=0.95)) == 77.8
