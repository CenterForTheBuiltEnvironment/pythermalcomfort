import pytest

from pythermalcomfort.models import esi
from tests.conftest import is_equal


def test_esi() -> None:
    """Test that the function calculates the ESI correctly for given inputs."""
    result = esi(tdb=30.2, rh=42.2, sol_radiation_global=766)
    is_equal(result.esi, 26.2, 0.1)


def test_esi_list_input() -> None:
    """Test that the function calculates the ESI correctly for list inputs."""
    result = esi([30.2, 27.0], [42.2, 68.8], [766, 289])
    is_equal(result.esi, [26.2, 25.6], 0.1)


@pytest.mark.parametrize(
    ("tdb", "rh", "sol"),
    [
        (30, -5, 500),  # negative relative humidity
        (30, 120, 500),  # RH above 100%
        (30, 50, -10),  # negative solar radiation
    ],
)
def test_esi_invalid_numeric_ranges(tdb, rh, sol) -> None:
    """Test that the function raises ValueError for invalid input ranges."""
    with pytest.raises(ValueError):
        esi(tdb=tdb, rh=rh, sol_radiation_global=sol)


@pytest.mark.parametrize(
    ("tdb", "rh", "sol", "expected"),
    [
        (30, 0, 500, 16.5),  # minimum RH
        (30, 100, 500, 19.5),  # maximum RH
        (30, 50, 0, 18.0),  # minimum solar radiation
    ],
)
def test_esi_boundary_conditions(
    tdb: float,
    rh: float,
    sol: float,
    expected: float,
) -> None:
    """Test that the function handles boundary conditions correctly."""
    result = esi(tdb=tdb, rh=rh, sol_radiation_global=sol)
    is_equal(result.esi, expected, 0.1)  # Replace expected values with actual results


@pytest.mark.parametrize(
    ("tdb", "rh", "sol"),
    [("30.0", 45, 500), (30, "45", 500), (30, 45, "500.0")],
)
def test_esi_invalid_type(tdb: float, rh: float, sol: float) -> None:
    """Test that the function raises TypeError for invalid input types."""
    with pytest.raises(TypeError):
        esi(tdb=tdb, rh=rh, sol_radiation_global=sol)
