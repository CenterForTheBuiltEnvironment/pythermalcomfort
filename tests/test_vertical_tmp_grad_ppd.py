from pythermalcomfort.models import vertical_tmp_grad_ppd
from tests.conftest import Urls, retrieve_reference_table, validate_result
import pytest


def test_vertical_tmp_grad_ppd(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.VERTICAL_TMP_GRAD_PPD.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = vertical_tmp_grad_ppd(
            tdb=inputs["tdb"],
            tr=inputs["tr"],
            vr=inputs["vr"],
            rh=inputs["rh"],
            met=inputs["met"],
            clo=inputs["clo"],
            vertical_tmp_grad=inputs["delta_t"],
            units=inputs.get("units", "SI"),
        )

        validate_result(result, outputs, tolerance)

    # Test for ValueError
    with pytest.raises(ValueError):
        vertical_tmp_grad_ppd(25, 25, 0.3, 50, 1.2, 0.5, 7)
