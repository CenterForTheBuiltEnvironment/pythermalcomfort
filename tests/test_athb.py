import numpy as np

from pythermalcomfort.models import athb


def test_athb():
    np.testing.assert_equal(
        athb(
            tdb=[25, 25, 15, 25],
            tr=[25, 35, 25, 25],
            vr=[0.1, 0.1, 0.2, 0.1],
            rh=[50, 50, 50, 60],
            met=[1.1, 1.5, 1.2, 2],
            t_running_mean=[20, 20, 20, 20],
        ),
        [0.17, 0.912, -0.755, 0.38],
    )
