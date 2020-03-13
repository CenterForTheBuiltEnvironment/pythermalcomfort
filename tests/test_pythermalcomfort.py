from pythermalcomfort.cli import main
from pythermalcomfort.psychrometrics import *


def test_main():
    assert main([]) == 0
