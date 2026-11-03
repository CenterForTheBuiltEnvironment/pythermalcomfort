import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pytest

from pythermalcomfort.models import two_nodes_gagge
from pythermalcomfort.models.two_nodes_gagge import _PARALLEL_MIN_POINTS
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_two_nodes(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the two nodes Gagge model correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.TWO_NODES.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = two_nodes_gagge(**inputs)

        validate_result(result, outputs, tolerance)


@pytest.mark.parametrize(
    "w_max", [0.01, 0.06, 1.2, [0.5, 0.05], np.nan, [0.5, np.nan], True, [True, 0.5]]
)
def test_two_nodes_rejects_w_max_out_of_range(w_max) -> None:
    """Reject w_max that would break the wettedness cap (#489).

    Below 0.06 p_rsw is negative at the cap, at exactly 0.06 disc divides by zero
    when met <= 1, NaN disables the cap, and True would silently mean a cap of 1.
    """
    with pytest.raises(ValueError, match="w_max"):
        two_nodes_gagge(tdb=40, tr=40, v=0.1, rh=50, met=1, clo=0.5, w_max=w_max)


@pytest.mark.parametrize("w_max", [False, 0, 0.0601, 0.5, 1])
def test_two_nodes_accepts_w_max_in_range(w_max) -> None:
    """False and 0 still mean that w_max is computed from v and clo."""
    two_nodes_gagge(tdb=40, tr=40, v=0.1, rh=50, met=2, clo=0.5, w_max=w_max)


@pytest.mark.parametrize("name", ["tdb", "tr", "v", "rh"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_two_nodes_returns_nan_for_non_finite_inputs(name, bad) -> None:
    """A non-finite value gives NaN for that point only, instead of StopIteration."""
    inputs = {"tdb": [30, 30], "tr": [30, 30], "v": [0.2, 0.2], "rh": [50, 50]}
    inputs[name] = [inputs[name][0], bad]
    result = two_nodes_gagge(met=1.2, clo=0.5, **inputs)
    reference = two_nodes_gagge(tdb=30, tr=30, v=0.2, rh=50, met=1.2, clo=0.5)

    assert result.disc[0] == reference.disc
    assert np.isnan(result.disc[1])
    assert np.isnan(result.t_core[1])
    assert np.isnan(result.set[1])


def test_two_nodes_array_matches_scalar_calls() -> None:
    """The parallel array loop gives the same result as one call per point (#496).

    Mixes sitting and standing, broadcasts scalar met and clo against array inputs
    and checks that a 2-D input keeps its shape.
    """
    tdb = np.array([[18.0, 25.0, 31.0], [36.0, 40.0, 22.0]])
    tr = np.array([[20.0, 25.0, 35.0], [40.0, 45.0, 22.0]])
    v = np.array([[0.1, 0.3, 0.8], [1.2, 0.2, 0.05]])
    rh = np.array([[30.0, 50.0, 60.0], [40.0, 70.0, 50.0]])
    position = np.array(
        [["sitting", "standing", "sitting"], ["standing", "sitting", "standing"]]
    )
    result = two_nodes_gagge(
        tdb=tdb, tr=tr, v=v, rh=rh, met=1.4, clo=0.6, position=position
    )
    assert result.set.shape == tdb.shape

    for idx in np.ndindex(tdb.shape):
        point = two_nodes_gagge(
            tdb=tdb[idx],
            tr=tr[idx],
            v=v[idx],
            rh=rh[idx],
            met=1.4,
            clo=0.6,
            position=str(position[idx]),
        )
        for field in result.__dataclass_fields__:
            assert getattr(result, field)[idx] == getattr(point, field), field


def test_two_nodes_position_changes_result() -> None:
    """Sitting and standing use different radiative areas, so their SET differs."""
    result = two_nodes_gagge(
        tdb=[30, 30],
        tr=40,
        v=0.1,
        rh=50,
        met=1.2,
        clo=0.5,
        position=["sitting", "standing"],
    )
    assert result.set[0] != result.set[1]


@pytest.mark.parametrize("n_points", [3, 2 * _PARALLEL_MIN_POINTS])
def test_two_nodes_raises_if_any_point_does_not_converge(n_points) -> None:
    """A point that does not converge still raises, from either loop (#496).

    numba drops exceptions raised inside a prange loop, so the kernel flags the
    point and two_nodes_gagge raises after the loop. A finite but extreme tr is used
    because NaN inputs return NaN instead.
    """
    tr = np.full(n_points, 25.0)
    tr[1] = 5000
    with pytest.raises(StopIteration, match="Max iterations exceeded"):
        two_nodes_gagge(tdb=25, tr=tr, v=0.1, rh=50, met=1.2, clo=0.5)


@pytest.mark.parametrize("n_points", [3, 2 * _PARALLEL_MIN_POINTS])
def test_two_nodes_keeps_kernel_exception_from_either_loop(n_points) -> None:
    """An exception in the kernel is raised as itself, from either loop (#518).

    numba drops it inside a prange loop, so without the serial rerun p_atm=0 gave
    StopIteration or uninitialized outputs instead of ZeroDivisionError.
    """
    p_atm = np.full(n_points, 101325.0)
    p_atm[1] = 0
    with pytest.raises(ZeroDivisionError):
        two_nodes_gagge(tdb=25, tr=25, v=0.1, rh=50, met=1.2, clo=0.5, p_atm=p_atm)


def _large_array_inputs() -> dict:
    """Return inputs with enough points that two_nodes_gagge uses the parallel loop."""
    n = 2 * _PARALLEL_MIN_POINTS
    return {
        "tdb": np.linspace(15, 40, n),
        "tr": np.linspace(40, 15, n),
        "v": np.linspace(0.05, 1.5, n),
        "rh": 50,
        "met": 1.3,
        "clo": 0.6,
        "position": np.where(np.arange(n) % 2 == 0, "sitting", "standing"),
    }


def test_two_nodes_serial_loop_matches_parallel_loop() -> None:
    """The one-core loop used off the main thread equals the parallel loop (#496)."""
    inputs = _large_array_inputs()
    parallel = two_nodes_gagge(**inputs, round_output=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        serial = executor.submit(
            partial(two_nodes_gagge, **inputs, round_output=False)
        ).result()

    for field in parallel.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(serial, field), getattr(parallel, field), err_msg=field
        )


def test_two_nodes_can_be_called_from_several_threads() -> None:
    """Concurrent calls from Python threads, as dask makes, must not crash (#496).

    numba's "workqueue" thread pool aborts the whole process when two threads use it
    at once, so the calls run in a subprocess with that pool forced. Linux often has
    the thread-safe OpenMP pool, which would hide the problem.
    """
    script = textwrap.dedent(
        """
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np

        from pythermalcomfort.models import two_nodes_gagge

        tdb = np.linspace(15, 40, 2000)
        expected = two_nodes_gagge(tdb=tdb, tr=25, v=0.2, rh=50, met=1.2, clo=0.5).set

        def call(_):
            return two_nodes_gagge(tdb=tdb, tr=25, v=0.2, rh=50, met=1.2, clo=0.5).set

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(call, range(16)))
        assert all(np.array_equal(result, expected) for result in results)
        print("ok")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "NUMBA_THREADING_LAYER": "workqueue"},
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
