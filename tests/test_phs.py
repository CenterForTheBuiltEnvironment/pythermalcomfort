import pytest

from pythermalcomfort.models import phs
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_phs(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.PHS.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = phs(**inputs)

        validate_result(result, outputs, tolerance)


def test_value_acclimatized():
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            acclimatized=101,
        )

    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            acclimatized=-1,
        )


def test_value_weight():
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            weight=1001,
        )

    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            weight=0,
        )


def test_value_drink():
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            drink=0.5,
        )
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            drink=2,
        )
