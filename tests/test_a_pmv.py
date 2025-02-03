import pytest

from pythermalcomfort.models import pmv_a
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_a_pmv(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.A_PMV.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = pmv_a(**inputs)

        validate_result(result, outputs, tolerance)


def test_a_pmv_wrong_input_type():
    with pytest.raises(TypeError):
        pmv_a("25", 25, 0.1, 50, 1.2, 0.5, 7)
    with pytest.raises(ValueError):
        pmv_a(25, 25, 0.1, 50, 1.2, 0.5, 7, units="celsius")


def test_not_valid_units():
    with pytest.raises(ValueError):
        pmv_a(25, 25, 0.1, 50, 1.2, 0.5, 7, units="wrong")
