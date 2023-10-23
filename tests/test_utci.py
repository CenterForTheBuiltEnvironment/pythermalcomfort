import numpy as np
import pytest

from pythermalcomfort.models import (
    utci,
)
from pythermalcomfort.models.utci import _utci_optimized


@pytest.fixture
def data_test_utci():
    return [  # I have commented the lines of code that don't pass the test
        {"tdb": 25, "tr": 27, "rh": 50, "v": 1, "return": {"utci": 25.2}},
        {"tdb": 19, "tr": 24, "rh": 50, "v": 1, "return": {"utci": 20.0}},
        {"tdb": 19, "tr": 14, "rh": 50, "v": 1, "return": {"utci": 16.8}},
        {"tdb": 27, "tr": 22, "rh": 50, "v": 1, "return": {"utci": 25.5}},
        {"tdb": 27, "tr": 22, "rh": 50, "v": 10, "return": {"utci": 20.0}},
        {"tdb": 27, "tr": 22, "rh": 50, "v": 16, "return": {"utci": 15.8}},
        {"tdb": 51, "tr": 22, "rh": 50, "v": 16, "return": {"utci": np.nan}},
        {"tdb": 27, "tr": 22, "rh": 50, "v": 0, "return": {"utci": np.nan}},
    ]


def test_utci(data_test_utci):
    for row in data_test_utci:
        np.testing.assert_equal(
            utci(row["tdb"], row["tr"], row["v"], row["rh"]),
            row["return"][list(row["return"].keys())[0]],
        )

    assert (utci(tdb=77, tr=77, v=3.28, rh=50, units="ip")) == 76.4

    assert (
        utci(tdb=30, tr=27, v=1, rh=50, units="si", return_stress_category=True)
    ) == {"utci": 29.6, "stress_category": "moderate heat stress"}
    assert (utci(tdb=9, tr=9, v=1, rh=50, units="si", return_stress_category=True)) == {
        "utci": 8.7,
        "stress_category": "slight cold stress",
    }


def test_utci_numpy(data_test_utci):
    tdb = np.array([d["tdb"] for d in data_test_utci])
    tr = np.array([d["tr"] for d in data_test_utci])
    rh = np.array([d["rh"] for d in data_test_utci])
    v = np.array([d["v"] for d in data_test_utci])
    expect = np.array([d["return"]["utci"] for d in data_test_utci])

    np.testing.assert_equal(utci(tdb, tr, v, rh), expect)

    tdb = np.array([25, 25])
    tr = np.array([27, 25])
    v = np.array([1, 1])
    rh = np.array([50, 50])
    expect = {
        "utci": np.array([25.2, 24.6]),
        "stress_category": np.array(["no thermal stress", "no thermal stress"]),
    }

    result = utci(tdb, tr, v, rh, units="si", return_stress_category=True)
    np.testing.assert_equal(result["utci"], expect["utci"])
    np.testing.assert_equal(result["stress_category"], expect["stress_category"])


def test_utci_optimized():
    np.testing.assert_equal(
        np.around(_utci_optimized([25, 27], 1, 1, 1.5), 2), [24.73, 26.57]
    )
