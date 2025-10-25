import numpy as np
from pythermalcomfort.environment.sky_emissivity import SkyEmissivity, SkyEmissivityResult
from pythermalcomfort.classes_return import Eps_Sky

from pythermalcomfort.environment.sky_emissivity_2 import SkyEmissivity as SkyEmissivity_2
from pythermalcomfort.environment.sky_emissivity_2 import EpsSky as Eps_Sky_2


# ----------------------
# Old SkyEmissivity tests
# ----------------------
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

# ----------------------
# New SkyEmissivity_2 tests
# ----------------------
def test_sky_emissivity2_brunt() -> None:
    sky: Eps_Sky_2 = SkyEmissivity_2.brunt(tdp=10.0)
    assert isinstance(sky, Eps_Sky_2)
    assert 0.0 <= sky.eps_sky <= 1.0

def test_sky_emissivity2_brunt_array() -> None:
    tdp_array = [0.0, 5.0, 10.0, 20.0]
    sky: Eps_Sky_2 = SkyEmissivity_2.brunt(tdp=tdp_array)
    assert isinstance(sky, Eps_Sky_2)
    assert isinstance(sky.eps_sky, np.ndarray)
    assert np.all((0.0 <= sky.eps_sky) & (sky.eps_sky <= 1.0))

def test_sky_emissivity2_dilley() -> None:
    base: Eps_Sky_2 = SkyEmissivity_2.brunt(tdp=10.0)
    corrected = Eps_Sky_2(eps_sky=Eps_Sky_2.dilly(base.eps_sky))
    expected = min(1.0, base.eps_sky * 1.05)
    assert np.isclose(corrected.eps_sky, expected, atol=1e-6)

def test_sky_emissivity2_prata() -> None:
    base: Eps_Sky_2 = SkyEmissivity_2.brunt(tdp=10.0)
    corrected = Eps_Sky_2(eps_sky=Eps_Sky_2.prata(base.eps_sky))
    expected = min(1.0, base.eps_sky * 1.03)
    assert np.isclose(corrected.eps_sky, expected, atol=1e-6)

def test_sky_emissivity2_chain_array_dilly() -> None:
    tdp_array = [0.0, 5.0, 10.0, 20.0]
    base: Eps_Sky_2 = SkyEmissivity_2.brunt(tdp=tdp_array)
    corrected = Eps_Sky_2(eps_sky=Eps_Sky_2.dilly(base.eps_sky))
    assert isinstance(corrected.eps_sky, np.ndarray)
    assert np.all((0.0 <= corrected.eps_sky) & (corrected.eps_sky <= 1.0))

def test_sky_emissivity2_example_correction_with_inputs() -> None:
    base: Eps_Sky_2 = SkyEmissivity_2.brunt(tdp=10.0)
    
    # Apply example correction (currently just multiplies by 1.03, clipped at 1)
    corrected = Eps_Sky_2(
        eps_sky=Eps_Sky_2.example_correction_with_inputs(base.eps_sky)
    )
    expected = min(1.0, base.eps_sky * 1.03)
    assert np.isclose(corrected.eps_sky, expected, atol=1e-6)