import numpy as np

from pythermalcomfort.models import e_pmv


def test_e_pmv():
    np.testing.assert_equal(
        e_pmv([24, 30], 30, vr=0.22, rh=50, met=1.4, clo=0.5, e_coefficient=0.6),
        [0.29, 0.91],
    )
