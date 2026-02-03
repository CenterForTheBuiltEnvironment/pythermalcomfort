import numpy as np
import pytest

from pythermalcomfort.models.bfu_rest import BFU_rest


def test_morris_lph_fans_scalar():
    result = BFU_rest(tdb=36.0, rh=60.0, group="YNG")

    assert result.group == "YNG"
    assert result.heat_storage > 0  # fan improves evaporative margin


def test_morris_lph_fans_array():
    tdb = np.array([35.0, 40.0])
    rh = np.array([50.0, 65.0])

    result = BFU_rest(tdb=tdb, rh=rh, group="OLD", round_output=False)

    for attribute in (
        result.heat_storage,
        result.e_req_fan,
        result.e_max_fan,
        result.e_req_no_fan,
        result.e_max_no_fan,
    ):
        assert isinstance(attribute, np.ndarray)
        assert attribute.shape == tdb.shape


def test_morris_lph_fans_invalid_group():
    with pytest.raises(ValueError):
        BFU_rest(tdb=35.0, rh=50.0, group="INVALID")
