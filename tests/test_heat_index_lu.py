import importlib

import numpy as np

from pythermalcomfort.models import heat_index_lu


def test_extended_heat_index() -> None:
    """Test the heat index function with extended inputs."""
    index = 0
    hi_test_values = [
        199.9994020652,
        199.9997010342,
        200.0000000021,
        209.9975943902,
        209.9987971085,
        209.9999998068,
        219.9915822029,
        219.9957912306,
        219.9999999912,
        229.9739691979,
        229.9869861009,
        230.0000001850,
        239.9253828022,
        239.9626700074,
        240.0000000003,
        249.7676757244,
        249.8837049107,
        250.0000000037,
        259.3735990024,
        259.6864068902,
        259.9999999944,
        268.5453870455,
        269.2745889562,
        270.0000002224,
        277.2234200026,
        278.6369451963,
        280.0000000091,
        285.7510545370,
        288.2813660100,
        290.7860610129,
        297.5737503539,
        300.2922595865,
        305.3947127590,
        305.5549530893,
        318.6225524695,
        359.9063248191,
        313.0298872791,
        359.0538750602,
        407.5345212438,
        320.5088548469,
        398.5759733823,
        464.9949352940,
        328.0358006469,
        445.8599463105,
        530.5524786708,
        333.2806160592,
        500.0421800191,
        601.9518435268,
        343.6312984164,
        559.6640227151,
        677.2462089759,
        354.1825692377,
        623.1960299857,
        755.0832658147,
    ]
    for t in range(200, 380, 10):
        for rh in [0, 0.5, 1]:
            hi = heat_index_lu(t - 273.15, rh * 100).hi
            assert np.isclose(hi + 273.15, hi_test_values[index], atol=1)
            index += 1


def test_extended_heat_index_array_input() -> None:
    """Test the heat index function with array inputs."""
    hi = heat_index_lu([20, 40], 50).hi
    assert np.allclose(hi, [19.0, 63.4], atol=0.1)


def test_heat_index_lu_regression_values() -> None:
    """Pin unrounded outputs against the reference (pre-numba) implementation.

    Values were captured from the pure-Python implementation before it was
    rewritten for numba, confirmed to match bit-for-bit across a wide random
    sweep of tdb/rh combinations plus edge cases (rh=0, rh=100, tdb near
    absolute zero).
    """
    cases = [
        (-273.15, 50, -273.15),
        (-40, 0, -40.03628046540541),
        (-40, 100, -39.999999998230464),
        (0, 0, -1.835038354014955),
        (0, 100, -1.722924025671091e-09),
        (25, 50, 24.979891812568553),
        (50, 0, 42.232631716202036),
        (50, 100, 151.4841566476622),
        (100, 50, 370.6571594081703),
    ]
    for tdb, rh, expected in cases:
        hi = heat_index_lu(tdb=tdb, rh=rh, round_output=False).hi
        assert np.isclose(hi, expected, atol=1e-6), (tdb, rh, hi, expected)


def test_heat_index_lu_root_solver_evaluations(monkeypatch) -> None:
    """The root solver needs far fewer residual evaluations than bisection.

    Bisection on [230, 300] K down to 1e-8 K takes 35 evaluations (the two end points
    and 33 halvings).
    """
    hi_lu = importlib.import_module("pythermalcomfort.models.heat_index_lu")

    residual = hi_lu._residual
    calls = []

    def counting_residual(kind, x, p0, p1, p2, p3):
        calls.append(x)
        return residual(kind=kind, x=x, p0=p0, p1=p1, p2=p2, p3=p3)

    monkeypatch.setattr(hi_lu, "_residual", counting_residual)

    # 0 °C and 50 % is in region II, where the outer solve is on rf
    eq_var_code, _, rf, _, _ = hi_lu._find_eq_var(273.15, 0.5)
    assert eq_var_code == hi_lu._EQ_RF
    t = hi_lu._brentq.py_func(
        kind=hi_lu._R_T_RF, p0=rf, p1=0.0, p2=0.0, p3=0.0, x1=230.0, x2=300.0, tol=1e-8
    )

    assert abs(residual(kind=hi_lu._R_T_RF, x=t, p0=rf, p1=0.0, p2=0.0, p3=0.0)) < 1e-6
    assert len(calls) < 20
