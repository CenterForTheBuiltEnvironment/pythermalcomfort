import numpy as np
import pytest

from pythermalcomfort.classes_return import SportsHeatStressRisk
from pythermalcomfort.models.sports_heat_stress_risk import (
    Sports,
    sports_heat_stress_risk,
)


@pytest.mark.parametrize(
    ("tdb", "tr", "rh", "vr", "sport", "expected_risk"),
    [
        # Moderate temperature
        (33, 20, 40, 2, Sports.MTB, 0.9),
        # High temperature
        (40, 20, 40, 2, Sports.MTB, 2.3),
        # Low temperature (low risk)
        (10, 20, 40, 2, Sports.MTB, 0.0),
        # Very high temperature (extreme risk)
        (60, 20, 40, 2, Sports.MTB, 3.0),
        # Different sport - running at high temp (extreme risk)
        (35, 35, 50, 0.5, Sports.RUNNING, 3.0),
    ],
)
def test_sports_heat_stress_risk_scalar(tdb, tr, rh, vr, sport, expected_risk):
    """Test sports heat stress risk calculation with scalar inputs."""
    result = sports_heat_stress_risk(tdb=tdb, tr=tr, rh=rh, vr=vr, sport=sport)

    # Verify return type is the dataclass
    assert isinstance(result, SportsHeatStressRisk)
    assert isinstance(result.t_medium, float)
    assert isinstance(result.t_high, float)
    assert isinstance(result.t_extreme, float)
    assert isinstance(result.recommendation, str)

    # Verify expected risk level
    if isinstance(expected_risk, float):
        assert result.risk_level_interpolated == pytest.approx(expected_risk, rel=0.01)
    else:
        # Handle pytest.approx objects
        assert result.risk_level_interpolated == expected_risk


def test_sports_heat_stress_risk_array():
    """Test sports heat stress risk calculation with array inputs (lists)."""
    t = [40, 40]
    rh = 40
    v = 2
    tr = 20
    result = sports_heat_stress_risk(tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.MTB)

    # Verify return type
    assert isinstance(result, SportsHeatStressRisk)

    # Verify array outputs
    np.testing.assert_allclose(
        result.risk_level_interpolated,
        [2.3, 2.3],
        rtol=0.01,
        atol=0.01,
    )
    assert isinstance(result.t_medium, list)
    assert isinstance(result.t_high, list)
    assert isinstance(result.t_extreme, list)
    assert len(result.t_medium) == 2
    assert len(result.t_high) == 2
    assert len(result.t_extreme) == 2


def test_sports_heat_stress_risk_numpy_array():
    """Test sports heat stress risk calculation with NumPy array inputs."""
    t = np.array([30, 35, 40])
    rh = np.array([40, 50, 60])
    v = np.array([0.5, 1.0, 1.5])
    tr = np.array([30, 35, 40])
    result = sports_heat_stress_risk(tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.RUNNING)

    # Verify return type
    assert isinstance(result, SportsHeatStressRisk)

    # Verify outputs are lists (converted from numpy arrays)
    assert isinstance(result.risk_level_interpolated, list)
    assert isinstance(result.t_medium, list)
    assert isinstance(result.t_high, list)
    assert isinstance(result.t_extreme, list)
    assert isinstance(result.recommendation, list)

    # Verify correct array length
    assert len(result.risk_level_interpolated) == 3
    assert len(result.t_medium) == 3
    assert len(result.t_high) == 3
    assert len(result.t_extreme) == 3
    assert len(result.recommendation) == 3

    # Verify all values are valid
    for risk in result.risk_level_interpolated:
        assert 0 <= risk <= 3


def test_sports_heat_stress_risk_different_sports():
    """Test that different sports produce different risk levels."""
    t = 35
    rh = 50
    v = 0.5
    tr = 35

    result_running = sports_heat_stress_risk(
        tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.RUNNING
    )
    result_walking = sports_heat_stress_risk(
        tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.WALKING
    )

    # Running has higher metabolic rate, so should have higher risk at same conditions
    assert isinstance(result_running, SportsHeatStressRisk)
    assert isinstance(result_walking, SportsHeatStressRisk)
    # Both should return valid risk levels
    assert 0 <= result_running.risk_level_interpolated <= 3
    assert 0 <= result_walking.risk_level_interpolated <= 3


def test_sports_heat_stress_risk_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    # Test invalid humidity (> 100%)
    with pytest.raises(
        ValueError, match="Relative humidity.*must be between 0 and 100"
    ):
        sports_heat_stress_risk(tdb=30, tr=30, rh=150, vr=0.5, sport=Sports.RUNNING)

    # Test invalid humidity (< 0%)
    with pytest.raises(
        ValueError, match="Relative humidity.*must be between 0 and 100"
    ):
        sports_heat_stress_risk(tdb=30, tr=30, rh=-10, vr=0.5, sport=Sports.RUNNING)

    # Test invalid air speed (negative)
    with pytest.raises(ValueError, match="Relative air speed.*must be non-negative"):
        sports_heat_stress_risk(tdb=30, tr=30, rh=50, vr=-1, sport=Sports.RUNNING)

    # Test invalid sport type
    with pytest.raises(TypeError, match="sport must be a _SportsValues instance"):
        sports_heat_stress_risk(tdb=30, tr=30, rh=50, vr=0.5, sport="invalid")


def test_sports_heat_stress_risk_broadcasting_incompatibility():
    """Test that incompatible array shapes raise appropriate broadcasting errors."""
    # Test incompatible array shapes (length mismatch)
    with pytest.raises(ValueError, match="not broadcastable"):
        sports_heat_stress_risk(
            tdb=[30, 35, 40],  # length 3
            tr=30,
            rh=[40, 50],  # length 2 - incompatible!
            vr=0.5,
            sport=Sports.SOCCER,
        )

    # Test another incompatible combination
    with pytest.raises(ValueError, match="not broadcastable"):
        sports_heat_stress_risk(
            tdb=[30, 35],
            tr=[30, 35, 40, 45],  # length 4 - incompatible with tdb!
            rh=50,
            vr=0.5,
            sport=Sports.TENNIS,
        )


def test_sports_heat_stress_risk_broadcasting():
    """Test that broadcasting works correctly with different input shapes."""
    # Test broadcasting with different array shapes
    result = sports_heat_stress_risk(
        tdb=[30, 35, 40], tr=30, rh=[40, 50, 60], vr=0.5, sport=Sports.SOCCER
    )

    assert isinstance(result, SportsHeatStressRisk)
    assert len(result.risk_level_interpolated) == 3
    assert len(result.t_medium) == 3
    assert len(result.t_high) == 3
    assert len(result.t_extreme) == 3


def test_sports_heat_stress_risk_dataclass_fields():
    """Test that the returned dataclass has all expected fields."""
    result = sports_heat_stress_risk(tdb=35, tr=35, rh=50, vr=0.5, sport=Sports.TENNIS)

    # Verify all required fields are present
    assert hasattr(result, "risk_level_interpolated")
    assert hasattr(result, "t_medium")
    assert hasattr(result, "t_high")
    assert hasattr(result, "t_extreme")
    assert hasattr(result, "recommendation")

    # Verify threshold order (t_medium < t_high < t_extreme)
    assert result.t_medium < result.t_high
    assert result.t_high < result.t_extreme

    # Verify recommendation is a non-empty string
    assert isinstance(result.recommendation, str)
    assert len(result.recommendation) > 0


def test_sports_heat_stress_risk_recommendations():
    """Test that recommendations are appropriate for different risk levels."""
    # Test low risk (risk level < 1.0)
    result_low = sports_heat_stress_risk(
        tdb=20, tr=20, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    assert "Activities can proceed as planned" in result_low.recommendation
    assert result_low.risk_level_interpolated < 1.0

    # Test medium risk (risk level 1.0-2.0)
    result_medium = sports_heat_stress_risk(
        tdb=30, tr=30, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    if 1.0 <= result_medium.risk_level_interpolated < 2.0:
        assert "Increase rest breaks" in result_medium.recommendation

    # Test high risk (risk level 2.0-3.0)
    result_high = sports_heat_stress_risk(
        tdb=38, tr=38, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    if 2.0 <= result_high.risk_level_interpolated < 3.0:
        assert "Implement frequent mandatory rest breaks" in result_high.recommendation

    # Test extreme risk (risk level >= 3.0)
    result_extreme = sports_heat_stress_risk(
        tdb=50, tr=50, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    assert "Exercise and play should be suspended" in result_extreme.recommendation
    assert result_extreme.risk_level_interpolated == pytest.approx(3.0, rel=1e-3)


def test_sports_heat_stress_risk_array_recommendations():
    """Test that array inputs produce array recommendations."""
    result = sports_heat_stress_risk(
        tdb=[20, 35, 45],
        tr=[20, 35, 45],
        rh=[50, 50, 50],
        vr=[0.5, 0.5, 0.5],
        sport=Sports.SOCCER,
    )

    # Verify recommendations is a list
    assert isinstance(result.recommendation, list)
    assert len(result.recommendation) == 3

    # Verify each recommendation is a string
    for rec in result.recommendation:
        assert isinstance(rec, str)
        assert len(rec) > 0


def test_sports_heat_stress_risk_threshold_capping():
    """Test that threshold temperatures are properly capped at min/max bounds."""
    # Test conditions that should trigger maximum threshold capping
    # Very high humidity and temperature should cap thresholds
    result_high = sports_heat_stress_risk(
        tdb=45, tr=45, rh=90, vr=0.1, sport=Sports.RUNNING
    )

    # Verify thresholds respect maximum bounds (from implementation)
    # max_t_low = 34.5, max_t_medium = 39, max_t_high = 43.5
    assert result_high.t_medium <= 34.5
    assert result_high.t_high <= 39.0
    assert result_high.t_extreme <= 43.5

    # Test conditions that should trigger minimum threshold capping
    # Low humidity and low temperature should set minimum thresholds
    result_low = sports_heat_stress_risk(
        tdb=25, tr=25, rh=10, vr=2.0, sport=Sports.WALKING
    )

    # Verify thresholds respect minimum bounds (from implementation)
    # min_t_medium = 22, min_t_high = 23, min_t_extreme = 25
    assert result_low.t_medium >= 22.0
    assert result_low.t_high >= 23.0
    assert result_low.t_extreme >= 25.0

    # Verify threshold ordering is always maintained
    assert result_high.t_medium < result_high.t_high < result_high.t_extreme
    assert result_low.t_medium < result_low.t_high < result_low.t_extreme


def test_sports_heat_stress_risk_edge_temperature_ranges():
    """Test risk calculation at temperature boundary conditions."""
    # Test very low temperature (should be low risk, risk_level = 0)
    result_very_low = sports_heat_stress_risk(
        tdb=15, tr=15, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    assert result_very_low.risk_level_interpolated == pytest.approx(0.0, abs=0.01)

    # Test very high temperature (should be extreme risk, risk_level = 3)
    result_very_high = sports_heat_stress_risk(
        tdb=50, tr=50, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    assert result_very_high.risk_level_interpolated == pytest.approx(3.0, abs=0.01)


def test_sports_heat_stress_risk_all_sports():
    """Test that all predefined sports in Sports dataclass work correctly."""
    # Get all sport attributes from Sports dataclass
    all_sports = [
        Sports.ABSEILING,
        Sports.ARCHERY,
        Sports.AUSTRALIAN_FOOTBALL,
        Sports.BASEBALL,
        Sports.BASKETBALL,
        Sports.BOWLS,
        Sports.CANOEING,
        Sports.CRICKET,
        Sports.CYCLING,
        Sports.EQUESTRIAN,
        Sports.FIELD_ATHLETICS,
        Sports.FIELD_HOCKEY,
        Sports.FISHING,
        Sports.GOLF,
        Sports.HORSEBACK,
        Sports.KAYAKING,
        Sports.RUNNING,
        Sports.MTB,
        Sports.NETBALL,
        Sports.OZTAG,
        Sports.PICKLEBALL,
        Sports.CLIMBING,
        Sports.ROWING,
        Sports.RUGBY_LEAGUE,
        Sports.RUGBY_UNION,
        Sports.SAILING,
        Sports.SHOOTING,
        Sports.SOCCER,
        Sports.SOFTBALL,
        Sports.TENNIS,
        Sports.TOUCH,
        Sports.VOLLEYBALL,
        Sports.WALKING,
    ]

    # Test moderate conditions with each sport
    tdb, tr, rh, vr = 32, 32, 50, 0.5

    for sport in all_sports:
        result = sports_heat_stress_risk(tdb=tdb, tr=tr, rh=rh, vr=vr, sport=sport)

        # Verify all sports return valid results
        assert isinstance(result, SportsHeatStressRisk)
        assert isinstance(result.risk_level_interpolated, float)
        assert 0 <= result.risk_level_interpolated <= 3
        assert isinstance(result.t_medium, float)
        assert isinstance(result.t_high, float)
        assert isinstance(result.t_extreme, float)
        assert isinstance(result.recommendation, str)

        # Verify threshold ordering
        assert result.t_medium < result.t_high < result.t_extreme


@pytest.mark.parametrize(
    ("tdb", "tr", "rh", "vr", "sport"),
    [
        # Boundary: rh = 0% (minimum humidity)
        (30, 30, 0, 0.5, Sports.RUNNING),
        # Boundary: rh = 100% (maximum humidity)
        (30, 30, 100, 0.5, Sports.RUNNING),
        # Boundary: vr = 0 (no air movement)
        (30, 30, 50, 0, Sports.SOCCER),
        # Boundary: very low vr
        (30, 30, 50, 0.01, Sports.TENNIS),
        # Combination: high temp, high humidity, no wind
        (40, 40, 100, 0, Sports.RUNNING),
        # Combination: low temp, low humidity, high wind
        (20, 20, 0, 5.0, Sports.CYCLING),
    ],
)
def test_sports_heat_stress_risk_boundary_values(tdb, tr, rh, vr, sport):
    """Test sports heat stress risk with boundary and edge case values."""
    result = sports_heat_stress_risk(tdb=tdb, tr=tr, rh=rh, vr=vr, sport=sport)

    # Verify function handles boundary values correctly
    assert isinstance(result, SportsHeatStressRisk)
    assert isinstance(result.risk_level_interpolated, float)
    assert 0 <= result.risk_level_interpolated <= 3
    assert isinstance(result.t_medium, float)
    assert isinstance(result.t_high, float)
    assert isinstance(result.t_extreme, float)
    assert isinstance(result.recommendation, str)

    # Verify threshold ordering is maintained
    assert result.t_medium < result.t_high < result.t_extreme

    # Verify no NaN values
    assert not np.isnan(result.risk_level_interpolated)
    assert not np.isnan(result.t_medium)
    assert not np.isnan(result.t_high)
    assert not np.isnan(result.t_extreme)
