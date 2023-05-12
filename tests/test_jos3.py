import pytest
from pythermalcomfort.jos3_functions.construction import bsa_rate


def test_bsa_rate_with_defaults():
    assert bsa_rate() == pytest.approx(1.00051943)


def test_bsa_rate_with_custom_height_weight():
    assert bsa_rate(height=1.80, weight=80.0) == pytest.approx(1.0662, rel=1e-4)


def test_bsa_rate_with_invalid_formula():
    with pytest.raises(ValueError):
        bsa_rate(height=1.72, weight=74.43, formula="ciao")
