from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Literal, Union, List

from pythermalcomfort.classes_input import SkyEmissivityBruntInputs, SkyEmissivitySwinbankInputs
from pythermalcomfort.utilities import Units, units_converter
from pythermalcomfort.classes_return import AutoStrMixin




@dataclass
class SkyEmissivity:
    """Collection of empirical sky emissivity models."""

    # is it reallty useful that this is nested??

    @dataclass(frozen=True, repr=False)
    class EpsSky(AutoStrMixin):
        """Dataclass to represent sky emissivity.

        Attributes
        ----------
        eps_sky : float or list of floats
            Sky emissivity, ranges from 0.0 to 1.0
        """
        eps_sky: Union[float, np.ndarray]

        def apply_dilley(self) -> "SkyEmissivity.EpsSky":
            """Apply Dilley correction (max 1.0)."""
            if isinstance(self.eps_sky, np.ndarray):
                corrected = np.minimum(1.0, self.eps_sky * 1.05)
            else:
                corrected = min(1.0, self.eps_sky * 1.05)
            return SkyEmissivity.EpsSky(eps_sky=corrected)



    @staticmethod
    def brunt(
        tdp: float | list[float],
        units: Literal["SI", "IP"] = Units.SI.value,
        ) -> SkyEmissivity.EpsSky:
        """
        Calculate sky emissivity using the Brunt (1975) empirical model.

        Parameters
        ----------
        tdp : float
            Dew point temperature of the air.
        units : {"SI", "IP"}, default "SI"
            Units of `tdp`. "SI" = °C, "IP" = °F.

        Returns
        -------
        EpsSky
            Instance containing the computed sky emissivity (eps_sky), 
            clipped to the physical range [0.0, 1.0].
        """  
        SkyEmissivityBruntInputs(tdp=tdp)

        if units.upper() == Units.IP.value:
            tdp = units_converter(from_units=Units.IP.value, tdp=tdp)[0]

        tdp_arr = np.array(tdp, dtype=float)

        epsilon = 0.741 + 0.0062 * tdp_arr
        eps = np.clip(epsilon, 0.0, 1.0)

        return SkyEmissivity.EpsSky(eps_sky=eps)

    # Revisit: Should we than even include this if it inaccurate?
    @staticmethod
    def swinbank(
        tdb: float | list[float],
        units: Literal["SI", "IP"] = Units.SI.value,
        ) -> SkyEmissivity.EpsSky:
        """
        Calculate sky emissivity using the Swinbank (1963) empirical formula.

        .. note::
            This is a simple calculation based on ambient air temperature. 
            It has shown large deviations compared to modern methods 
            and is **not recommended for accurate modeling**.

        Parameters
        ----------
        tdb : float or list of floats
            Dry bulb air temperature in degrees Celsius (SI) or Fahrenheit (IP).
        units : str, optional
            Units system, 'SI' or 'IP'. Defaults to 'SI'.

        Returns
        -------
        EpsSky
            A dataclass containing `eps_sky` (clipped to [0, 1]).
        """
        SkyEmissivitySwinbankInputs(tdb=tdb)

        tdb_arr = np.array(tdb, dtype=float)

        if units.upper() == Units.IP.value:
            tdb_arr = units_converter(from_units=Units.IP.value, tdb=tdb_arr)[0]

        T_k = tdb_arr + 273.15
        eps_sky = 9.37e-6 * T_k**2
        eps_sky = np.clip(eps_sky, 0.0, 1.0)

        # Return a typed dataclass
        return SkyEmissivity.EpsSky(eps_sky=eps_sky)   

