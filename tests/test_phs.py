import pytest
from pythermalcomfort.models import phs


def test_phs(get_phs_url, retrieve_data, is_equal):
    reference_table = retrieve_data(get_phs_url)
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = phs(**inputs)
        for key in outputs:
            # Use the custom is_equal for other types
            try:
                assert is_equal(result[key], outputs[key], tolerance.get(key, 1e-6))
            except AssertionError as e:
                print(
                    f"Assertion failed for {key}. Expected {outputs[key]}, got {result[key]}, inputs={inputs}\nError: {str(e)}"
                )
                raise


def test_value_acclimatized():
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=150,
            clo=0.5,
            posture=2,
            acclimatized=101,
        )

    with pytest.raises(ValueError):
        phs(
            tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2, acclimatized=-1
        )


def test_value_weight():
    with pytest.raises(ValueError):
        phs(tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2, weight=1001)

    with pytest.raises(ValueError):
        phs(tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2, weight=0)


def test_value_drink():
    with pytest.raises(ValueError):
        phs(tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2, drink=0.5)
    with pytest.raises(ValueError):
        phs(tdb=40, tr=40, rh=33.85, v=0.3, met=150, clo=0.5, posture=2, drink=2)
