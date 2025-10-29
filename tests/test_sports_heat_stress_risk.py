import numpy as np
import pytest

from pythermalcomfort.models.sports_heat_stress_risk import (
    Sports,
    sports_heat_stress_risk,
)


def test_sports_heat_stress_risk_scalar(monkeypatch):
    t = 33
    rh = 40
    v = 2
    tr = 20
    result = sports_heat_stress_risk(tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.MTB)
    assert result == pytest.approx(0.88888, rel=1e-3)

    t = 40
    rh = 40
    v = 2
    tr = 20
    result = sports_heat_stress_risk(tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.MTB)
    assert result == pytest.approx(2.32885906, rel=1e-3)

    t = [40, 40]
    rh = 40
    v = 2
    tr = 20
    np.testing.assert_allclose(
        sports_heat_stress_risk(tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.MTB),
        [2.32885906, 2.32885906],
        rtol=0.01,
        atol=0.01,
    )

    t = 10
    rh = 40
    v = 2
    tr = 20
    result = sports_heat_stress_risk(tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.MTB)
    assert result == pytest.approx(0, rel=1e-3)

    t = 60
    rh = 40
    v = 2
    tr = 20
    result = sports_heat_stress_risk(tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.MTB)
    assert result == pytest.approx(3, rel=1e-3)
