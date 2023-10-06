from pythermalcomfort.models import pet_steady


def test_pet():
    assert pet_steady(tdb=20, tr=20, rh=50, v=0.15, met=1.37, clo=0.5) == 18.85
    assert pet_steady(tdb=30, tr=30, rh=50, v=0.15, met=1.37, clo=0.5) == 30.59
    assert pet_steady(tdb=20, tr=20, rh=50, v=0.5, met=1.37, clo=0.5) == 17.16
    assert pet_steady(tdb=21, tr=21, rh=50, v=0.1, met=1.37, clo=0.9) == 21.08
    assert pet_steady(tdb=20, tr=20, rh=50, v=0.1, met=1.37, clo=0.9) == 19.92
    assert pet_steady(tdb=-5, tr=40, rh=2, v=0.5, met=1.37, clo=0.9) == 7.82
    assert pet_steady(tdb=-5, tr=-5, rh=50, v=5.0, met=1.37, clo=0.9) == -13.38
    assert pet_steady(tdb=30, tr=60, rh=80, v=1.0, met=1.37, clo=0.9) == 43.05
    assert pet_steady(tdb=30, tr=30, rh=80, v=1.0, met=1.37, clo=0.9) == 31.69
