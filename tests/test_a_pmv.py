import numpy as np

from pythermalcomfort.models import a_pmv


def test_a_pmv():
    np.testing.assert_equal(
        a_pmv([24, 30], 30, vr=0.22, rh=50, met=1.4, clo=0.5, a_coefficient=0.293),
        [0.48, 1.09],
    )
