import pytest

from pythermalcomfort.models import ankle_draft


def test_ankle_draft():
    assert (ankle_draft(25, 25, 0.2, 50, 1.2, 0.5, 0.3, units="SI")["PPD_ad"]) == 18.5
    assert (
        ankle_draft(77, 77, 0.2 * 3.28, 50, 1.2, 0.5, 0.4 * 3.28, units="IP")["PPD_ad"]
    ) == 23.5

    with pytest.raises(ValueError):
        ankle_draft(25, 25, 0.3, 50, 1.2, 0.5, 7)
