import numpy as np

from pythermalcomfort.models import clo_tout


def test_clo_tout():
    assert (clo_tout(tout=80.6, units="ip")) == 0.46
    np.testing.assert_equal(clo_tout(tout=[80.6, 82], units="ip"), [0.46, 0.46])
    assert (clo_tout(tout=27)) == 0.46
    np.testing.assert_equal(clo_tout(tout=[27, 24]), [0.46, 0.48])
