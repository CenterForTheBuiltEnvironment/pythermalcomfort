from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Union, Callable, Optional


from pythermalcomfort.classes_input import (
    SkyEmissivityBruntInputs,
    SkyEmissivitySwinbankInputs,
    SkyEmissivityClarkAllenInputs,
)
from pythermalcomfort.utilities import Units, units_converter

# Variant 02: 
# Example:
# eps_sky = SkyEmissivity.brunt(tdp=10)
# eps_sky = SkyEmissivity.brunt(tdp=10, correction=EpsSky.apply_dilley) 
# ...

@dataclass(frozen=True)
class EpsSky:
    """Immutable container for sky emissivity value."""
    eps_sky: Union[float, np.ndarray]

    @staticmethod
    def dilly(eps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Dilley correction (max 1.0)."""
        return np.minimum(1.0, np.asarray(eps) * 1.05)

    @staticmethod
    def prata(eps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Placeholder for Prata correction example."""
        return np.minimum(1.0, np.asarray(eps) * 1.03)

    @staticmethod
    def example_correction_with_inputs(
        eps: Union[float, np.ndarray],
        **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Example correction that can take extra inputs from kwargs.

        Accepts keys like 'cloud_fraction', 'tdb', etc.
        """
        cloud_fraction = kwargs.get("cloud_fraction", 0.0)
        factor = 1.03 + 0.05 * cloud_fraction
        return np.minimum(1.0, np.asarray(eps) * factor)



class SkyEmissivity:
    """Collection of empirical sky emissivity models."""

    @staticmethod
    def brunt(
        tdp: float | list[float],
        units: str = Units.SI.value,
        correction: Optional[Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]] = None,
    ) -> EpsSky:
        """Brunt (1975) model."""
        SkyEmissivityBruntInputs(tdp=tdp)

        tdp_arr = np.array(tdp, dtype=float)
        if units.upper() == Units.IP.value:
            tdp_arr = units_converter(from_units=Units.IP.value, tdp=tdp_arr)[0]

        eps = np.clip(0.741 + 0.0062 * tdp_arr, 0.0, 1.0)

        if correction:
            eps = correction(eps)

        return EpsSky(eps_sky=eps)

    @staticmethod
    def swinbank(
        tdb: float | list[float],
        units: str = Units.SI.value,
        correction: Optional[Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]] = None,
    ) -> EpsSky:
        """Swinbank (1963) model — simple ambient air temperature approach."""
        SkyEmissivitySwinbankInputs(tdb=tdb)

        tdb_arr = np.array(tdb, dtype=float)
        if units.upper() == Units.IP.value:
            tdb_arr = units_converter(from_units=Units.IP.value, tdb=tdb_arr)[0]

        T_k = tdb_arr + 273.15
        eps = np.clip(9.37e-6 * T_k**2, 0.0, 1.0)

        if correction:
            eps = correction(eps)

        return EpsSky(eps_sky=eps)

    @staticmethod
    def clark_allen(
        tdp: float | list[float],
        cloud_fraction: float | list[float] = 0.0,
        units: str = Units.SI.value,
        correction: Optional[Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]] = None,
    ) -> EpsSky:
        """Clark & Allen (1978) model with optional cloud fraction."""
        SkyEmissivityClarkAllenInputs(tdp=tdp, fcn=cloud_fraction)

        tdp_arr = np.array(tdp, dtype=float)
        cloud_arr = np.array(cloud_fraction, dtype=float)

        if units.upper() == Units.IP.value:
            tdp_arr = units_converter(from_units=Units.IP.value, tdp=tdp_arr)[0]

        T_k = tdp_arr + 273.15
        eps_clear = 0.787 + 0.764 * np.log(T_k)
        eps = np.clip(eps_clear * (1 + 0.23 * cloud_arr), 0.0, 1.0)

        if correction:
            eps = correction(eps)

        return EpsSky(eps_sky=eps)

