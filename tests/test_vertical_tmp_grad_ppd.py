import pytest

from pythermalcomfort.models import vertical_tmp_grad_ppd


def test_vertical_tmp_grad_ppd():
    assert (
        vertical_tmp_grad_ppd(77, 77, 0.328, 50, 1.2, 0.5, 7 / 1.8, units="ip")[
            "PPD_vg"
        ]
    ) == 13.0
    assert (
        vertical_tmp_grad_ppd(77, 77, 0.328, 50, 1.2, 0.5, 7 / 1.8, units="ip")[
            "Acceptability"
        ]
    ) == False
    assert (vertical_tmp_grad_ppd(25, 25, 0.1, 50, 1.2, 0.5, 7)["PPD_vg"]) == 12.6
    assert (vertical_tmp_grad_ppd(25, 25, 0.1, 50, 1.2, 0.5, 4)["PPD_vg"]) == 1.7
    assert (
        vertical_tmp_grad_ppd(25, 25, 0.1, 50, 1.2, 0.5, 4)["Acceptability"]
    ) == True

    with pytest.raises(ValueError):
        vertical_tmp_grad_ppd(25, 25, 0.3, 50, 1.2, 0.5, 7)
