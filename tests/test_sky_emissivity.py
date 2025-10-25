import numpy as np
from pythermalcomfort.environment.sky_emissivity import SkyEmissivity, SkyEmissivityResult
from pythermalcomfort.classes_return import Eps_Sky


def test_sky_emissivity_brunt() -> None:
    sky: SkyEmissivityResult = SkyEmissivity.brunt(tdp=10.0)
    assert isinstance(sky, SkyEmissivityResult)
    assert 0.0 <= sky.eps_sky <= 1.0

def test_sky_emissivity_brunt_array() -> None:
    tdp_array = [0.0, 5.0, 10.0, 20.0]
    sky: SkyEmissivityResult = SkyEmissivity.brunt(tdp=tdp_array)
    assert isinstance(sky, SkyEmissivityResult)
    assert isinstance(sky.eps_sky, np.ndarray)
    assert np.all((0.0 <= sky.eps_sky) & (sky.eps_sky <= 1.0))

def test_sky_emissivity_correction() -> None:
    base: SkyEmissivityResult = SkyEmissivity.brunt(tdp=10.0)
    corrected: SkyEmissivityResult = base.apply_dilley()
    expected = min(1.0, base.eps_sky * 1.05)
    assert np.isclose(corrected.eps_sky, expected, atol=1e-6)

def test_sky_emissivity_chain() -> None:
    chained: SkyEmissivityResult = SkyEmissivity.brunt(tdp=10.0).apply_dilley()
    assert isinstance(chained, Eps_Sky)
    assert 0.0 <= chained.eps_sky <= 1.0

def test_sky_emissivity_chain_array() -> None:
    tdp_array = [0.0, 5.0, 10.0, 20.0]
    base: SkyEmissivityResult = SkyEmissivity.brunt(tdp=tdp_array)
    corrected: SkyEmissivityResult = base.apply_dilley()
    assert isinstance(corrected.eps_sky, np.ndarray)
    assert np.all((0.0 <= corrected.eps_sky) & (corrected.eps_sky <= 1.0))
