from pythermalcomfort.models import e_pmv
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_e_pmv(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.E_PMV.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = e_pmv(
            tdb=inputs["tdb"],
            tr=inputs["tr"],
            vr=inputs["vr"],
            rh=inputs["rh"],
            met=inputs["met"],
            clo=inputs["clo"],
            e_coefficient=inputs["e_coefficient"],
        )

        validate_result(result, outputs, tolerance)
