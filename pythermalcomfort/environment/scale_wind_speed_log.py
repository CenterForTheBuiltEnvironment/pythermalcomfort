from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import NumericInput, ScaleWindSpeedLogInputs
from pythermalcomfort.classes_return import ScaleWindSpeedLog


def scale_wind_speed_log(
    v_z1: NumericInput,
    z2: NumericInput,
    z1: NumericInput = 10.0,
    z0: NumericInput = 0.01,
    d: NumericInput = 0.0,
    round_output: bool = True,
) -> ScaleWindSpeedLog:
    """Scale wind speed from the reference height to a user specified height using the
    logarithmic wind profile based on surface roughness length [Oke1987]_.

    The logarithmic wind profile is a semi-empirical relationship that describes how
    wind speed changes with height above the ground in the surface layer of the
    atmosphere. This equation assumes neutral atmospheric stability and is valid for
    heights typically ranging from a few meters to about 100 meters above the ground.
    It is commonly used in meteorology, wind engineering, and environmental
    studies to estimate wind speeds at different heights based on a known reference
    height. The formula is given by:

    .. math::
        v(h) = v_{ref} \\times \\frac{\\ln((h - d)/z0)}{\\ln((h_{ref} - d)/z0)}

    where:

    - :math:`v(h)` is the wind speed at height :math:`h`
    - :math:`v_{ref}` is the wind speed at the reference height :math:`h_{ref}` (commonly 10 m)
    - :math:`h` is the height at which the wind speed is to be estimated
    - :math:`h_{ref}` is the reference height (10 m)
    - :math:`z0` is the surface roughness length, which characterizes the roughness of the terrain
    - :math:`d` is the zero-plane displacement height, which accounts for flow obstruction by vegetation or buildings

    .. note::
        This function assumes neutral atmospheric stability conditions. For
        non-neutral conditions, more complex models such as the Monin-Obukhov
        similarity theory should be used.
        Moreover, this function does not account for obstacles or complex terrain
        effects which may require computational fluid dynamics (CFD) simulations or
        measurements.
        The profile only describes wind well above the obstacles. Among buildings
        or trees, a target height of about 2 m is inside the canopy, where the
        logarithmic profile does not apply for any value of ``d``. The function
        then gives a rough estimate only, and street-level wind in dense urban
        areas needs a canopy-layer model or measurements.

    Parameters
    ----------
    v_z1 : float or list of floats
        Wind speed at the reference height z1 (default 10 m), [m/s]
    z2 : float or list of floats
        Height at which wind speed needs to be scaled, [m]
    z1 : float or list of floats, optional
        Reference height, [m]. Default is 10.0 m.
    z0 : float or list of floats, optional
        Surface roughness length, [m]. Default is 0.01 (open terrain). This value
        accounts for the roughness of the terrain and depends on land use. Values from
        Table 1 of [Sharples2023]_, which maps Australian land classes to Davenport
        roughness classes:

        - 0.005 m: Inland waterbodies (lakes and dams)
        - 0.03 m: Irrigated pasture
        - 0.1 m: Irrigated cropping
        - 0.25 m: Irrigated sugar
        - 0.43 m: Closed forest
        - 1.0 m: Urban areas

    d : float or list of floats, optional
        Zero-plane displacement height, [m]. Default is 0.0 (no displacement).
        This value accounts for the height at which the wind speed effectively becomes
        zero due to obstacles like trees or buildings. For scaling 10 m wind to about
        2 m, use 0.0, as recommended in Section 3.3 of [Sharples2023]_. The report
        found that its non-zero values lowered the scaled wind speed and smoothed out
        extremes. Its Table 1 lists 0.5 m for many land classes, including urban areas,
        but those values were capped at 0.5 m because the target height is only 2 m
        (Section 2.4). They are not displacement heights of real buildings, which are
        usually several metres in a city.

    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    ScaleWindSpeedLog
        Dataclass with attribute ``v_z2`` containing the scaled wind speed(s)
        at the target height z2, [m/s].

    Raises
    ------
    TypeError
        If input parameters are not of valid numeric types
    ValueError
        If any parameter values violate physical constraints

    Examples
    --------
    .. code-block:: python

        # Scale wind speed to 2m height (default open terrain z0=0.01)
        scale_wind_speed_log(v_z1=5.0, z2=2.0)

        # Scale wind speed to 2m height over rough terrain
        scale_wind_speed_log(v_z1=5.0, z2=2.0, z0=0.1)

        # Scale multiple wind speeds to different heights
        scale_wind_speed_log(v_z1=[3.0, 5.0], z2=[1.5, 2.5])

        # Scale with different surface roughness for each measurement
        scale_wind_speed_log(v_z1=[3.0, 5.0], z2=[1.5, 2.5], z0=[0.02, 0.1])
    """
    ScaleWindSpeedLogInputs(
        v_z1=v_z1,
        z2=z2,
        z1=z1,
        z0=z0,
        d=d,
    )

    v_z1 = np.asarray(v_z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    z0 = np.asarray(z0, dtype=float)
    d = np.asarray(d, dtype=float)
    z1 = np.asarray(z1, dtype=float)

    # Use numpy.log to support array inputs/broadcasting
    with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
        v_z2 = v_z1 * np.log((z2 - d) / z0) / np.log((z1 - d) / z0)

    if round_output:
        v_z2 = np.around(v_z2, 2)

    # Return a dataclass instance for consistent API
    return ScaleWindSpeedLog(v_z2=v_z2)
