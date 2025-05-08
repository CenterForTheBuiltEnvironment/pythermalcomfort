from pythermalcomfort.models import esi
from tests.conftest import is_equal


def test_esi():
    result = esi(tdb=30.2, rh=42.2, sol_radiation_dir=766)
    is_equal(result.esi, 26.2, 0.1)


def test_esi_list_input():
    result = esi([30.2, 27.0], [42.2, 68.8], [766, 289])
    is_equal(result.esi, [26.2, 25.6], 0.1)
