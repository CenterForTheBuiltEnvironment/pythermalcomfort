import pytest
from pythermalcomfort.models import ankle_draft
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_ankle_draft(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.ANKLE_DRAFT.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = ankle_draft(**inputs)

        validate_result(result, outputs, tolerance)

    # Test for ValueError
    with pytest.raises(ValueError):
        ankle_draft(25, 25, 0.3, 50, 1.2, 0.5, 7)
