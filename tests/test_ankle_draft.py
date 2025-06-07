import numpy as np
import pytest

from pythermalcomfort.models.ankle_draft import AnkleDraft, ankle_draft
from pythermalcomfort.utilities import Units
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_ankle_draft(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the ankle draft correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.ANKLE_DRAFT.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = ankle_draft(**inputs)

        validate_result(result, outputs, tolerance)


def test_ankle_draft_invalid_input_range() -> None:
    """Test for ankle_draft with invalid input ranges."""
    # Test for ValueError
    with pytest.raises(ValueError):
        ankle_draft(25, 25, 0.3, 50, 1.2, 0.5, 7)
    # Test for ValueError
    with pytest.raises(ValueError):
        ankle_draft(25, 25, 0.3, 50, 1.2, 0.5, [0.1, 0.2, 0.3, 0.4])


def test_ankle_draft_outside_ashrae_range() -> None:
    """Test for ankle_draft with inputs outside the ASHRAE range."""
    r = ankle_draft(50, 25, 0.1, 50, 1.2, 0.5, 0.1)
    assert np.isnan(r.ppd_ad)

    r = ankle_draft([50, 45], 25, 0.1, 50, 1.2, 0.5, 0.1)
    assert np.all(np.isnan(r.ppd_ad))


def test_ankle_draft_list_inputs() -> None:
    """Test for ankle_draft with list inputs."""
    results = ankle_draft(
        tdb=[25, 26, 27],
        tr=[25, 26, 27],
        vr=[0.1, 0.1, 0.1],
        rh=[50, 50, 50],
        met=[1.2, 1.2, 1.2],
        clo=[0.5, 0.5, 0.5],
        v_ankle=[0.1, 0.1, 0.1],
        units=Units.SI.value,
    )
    assert isinstance(results, AnkleDraft)
    assert len(results.ppd_ad) == 3
    assert len(results.acceptability) == 3


def test_ankle_draft_invalid_list_inputs() -> None:
    """Test for TypeError with invalid types in list inputs."""
    with pytest.raises(TypeError):
        ankle_draft(
            tdb=[25, "invalid", 27],
            tr=[25, 26, 27],
            vr=[0.1, 0.1, 0.1],
            rh=[50, 50, 50],
            met=[1.2, 1.2, 1.2],
            clo=[0.5, 0.5, 0.5],
            v_ankle=[0.1, 0.1, 0.1],
            units=Units.SI.value,
        )
