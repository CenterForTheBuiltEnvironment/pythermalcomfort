import numpy as np

from pythermalcomfort.models import utci
from pythermalcomfort.models.utci import _utci_optimized
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_utci(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.UTCI.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = utci(**inputs)

        validate_result(result, outputs, tolerance)


def test_utci_optimized():
    np.testing.assert_equal(
        np.around(_utci_optimized([25, 27], 1, 1, 1.5), 2), [24.73, 26.57]
    )
