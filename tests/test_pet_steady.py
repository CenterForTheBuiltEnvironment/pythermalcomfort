import pytest

from pythermalcomfort.models import pet_steady


def test_pet():
    assert pet_steady(tdb=20, tr=20, rh=50, v=0.15, met=1.37, clo=0.5) == pytest.approx(
        18.85, abs=0.1
    )
    assert pet_steady(tdb=30, tr=30, rh=50, v=0.15, met=1.37, clo=0.5) == pytest.approx(
        30.6, abs=0.01
    )
    assert pet_steady(tdb=20, tr=20, rh=50, v=0.5, met=1.37, clo=0.5) == pytest.approx(
        17.16, abs=0.01
    )
    assert pet_steady(tdb=21, tr=21, rh=50, v=0.1, met=1.37, clo=0.9) == pytest.approx(
        21.08, abs=0.01
    )
    assert pet_steady(tdb=20, tr=20, rh=50, v=0.1, met=1.37, clo=0.9) == pytest.approx(
        19.92, abs=0.01
    )
    assert pet_steady(tdb=-5, tr=40, rh=2, v=0.5, met=1.37, clo=0.9) == pytest.approx(
        7.82, abs=0.01
    )
    assert pet_steady(tdb=-5, tr=-5, rh=50, v=5.0, met=1.37, clo=0.9) == pytest.approx(
        -13.38, abs=0.01
    )
    assert pet_steady(tdb=30, tr=60, rh=80, v=1.0, met=1.37, clo=0.9) == pytest.approx(
        44.63, abs=0.01
    )
    assert pet_steady(tdb=30, tr=30, rh=80, v=1.0, met=1.37, clo=0.9) == pytest.approx(
        32.21, abs=0.01
    )
