import numpy as np
import pytest

from pythermalcomfort.models.bfu_occupational import BFU_occupational


def test_foster_fans_scalar():
    result = BFU_occupational(
        tdb=30.0,
        rh=60.0,
        position="standing",
        activity="walk",
        speed=1.5,
        round_output=False,
    )

    assert result.delta_storage < 0  # fan reduces heat storage
    assert result.storage_no_fan > result.storage_fan

    if result.delta_storage < -50:
        expected_interpretation = 1
    elif result.delta_storage <= 50:
        expected_interpretation = 0
    else:
        expected_interpretation = -1
    assert result.interpretation == expected_interpretation


def test_foster_fans_array():
    tdb = np.array([30.0, 32.0])
    rh = np.array([60.0, 55.0])
    speed = np.array([0.5, 0.0])

    result = BFU_occupational(
        tdb=tdb,
        rh=rh,
        position=["standing", "seated"],
        activity=["walk", "rest"],
        speed=speed,
        round_output=False,
    )

    for attribute in (
        result.storage_fan,
        result.storage_no_fan,
        result.delta_storage,
        result.dry_heat_fan,
        result.dry_heat_no_fan,
        result.evaporative_heat_fan,
        result.evaporative_heat_no_fan,
        result.respiratory_heat,
        result.sweat_efficiency_fan,
        result.sweat_efficiency_no_fan,
        result.interpretation,
    ):
        assert isinstance(attribute, np.ndarray)
        assert attribute.shape == tdb.shape

    expected_interpretation = np.select(
        [result.delta_storage < -50, result.delta_storage <= 50],
        [1, 0],
        default=-1,
    )
    np.testing.assert_array_equal(result.interpretation, expected_interpretation)


def test_foster_fans_invalid_position():
    with pytest.raises(ValueError):
        BFU_occupational(
            tdb=30.0,
            rh=60.0,
            position="lying",
        )
