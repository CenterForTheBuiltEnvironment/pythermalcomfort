import pytest

from pythermalcomfort.models import work_capacity


# Test incorrect intensity gives value error
def test_workcapacity_dunne_invalid_intensity():
    with pytest.raises(ValueError):
        work_capacity.workcapacity_dunne(30, "foo")


def test_workcapacity_hothaps_invalid_intensity():
    with pytest.raises(ValueError):
        work_capacity.workcapacity_hothaps(30, "foo")


def test_workcapacity_iso_invalid_intensity():
    with pytest.raises(TypeError):
        work_capacity.workcapacity_iso(30, "foo")


def test_workcapacity_niosh_invalid_intensity():
    with pytest.raises(TypeError):
        work_capacity.workcapacity_niosh(30, "foo")


#  Calculate with wbgt set to None
def test_workcapacity_dunne_wbgt_set_to_none():
    with pytest.raises(TypeError):
        work_capacity.workcapacity_dunne(None, "heavy")


def test_workcapacity_hothaps_wbgt_set_to_none():
    with pytest.raises(TypeError):
        work_capacity.workcapacity_hothaps(None, "heavy")
