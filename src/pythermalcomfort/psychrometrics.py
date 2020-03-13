import math


def p_sat_torr(t):
    """ Estimates the saturation vapor pressure in [torr]

    Parameters
    ----------
    t : float
        dry bulb air temperature, [C]

    Returns
    -------
    p_sat  : float
        saturation vapor pressure [torr]
    """
    return math.exp(18.6686 - 4030.183 / (t + 235.0))


def v_relative(v, met):
    """ Estimates the relative air velocity

    Parameters
    ----------
    v : float
        air velocity measured by the sensor, [m/s]
    met : float
        metabolic rate, [met]

    Returns
    -------
    vr  : float
        relative air velocity, [m/s]
    """

    if met > 1:
        return 0.3 * (met - 1)
    else:
        return v


def running_mean_outdoor_temperature(temp_array, alpha=0.8, units='SI'):
    """ Estimates the running mean temperature

    Parameters
    ----------
    temp_array: list
        array containing the mean daily temperature in descending order (i.e. from newest/yesterday to oldest) :math:`[\Theta_{day-1}, \Theta_{day-2}, \dots , \Theta_{day-n}]`.
        Where :math:`\Theta_{day-1}` is yesterday's daily mean temperature. The EN 16798-1 2019 [3]_ states that n should be equal to 7
    alpha : float
        constant between 0 and 1. The EN 16798-1 2019 [3]_ recommends a value of 0.8, while the ASHRAE 55 2017 recommends to choose values between 0.9 and 0.6, corresponding to a slow- and fast- response running mean, respectively.
        Adaptive comfort theory suggests that a slow-response running mean (alpha = 0.9) could be more appropriate for climates in which synoptic-scale (day-to- day) temperature dynamics are relatively minor, such as the humid tropics.
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    t_rm  : float
        running mean outdoor temperature
    """

    if units.lower() == 'ip':
        for ix, x in enumerate(temp_array):
            temp_array[ix] = units_converter(ta=temp_array[ix])[0]

    coeff = [alpha ** (ix) for ix, x in enumerate(temp_array)]
    t_rm = sum([a * b for a, b in zip(coeff, temp_array)]) / sum(coeff)

    if units.lower() == 'ip':
        t_rm = units_converter(tmp=t_rm, from_units='si')[0]

    return round(t_rm, 1)


def units_converter(from_units='ip', **kwargs):
    """ Converts IP values to SI units

    Parameters
    ----------
    from_units: str
        specify system to convert from
    **kwargs : [t, v]

    Returns
    -------
    converted values in SI units
    """
    results = list()
    if from_units == 'ip':
        for key, value in kwargs.items():
            if 'tmp' in key or key == 'tr' or key == 'ta':
                results.append((value - 32) * 5 / 9)
            if key in ['v', 'vr', 'vel']:
                results.append(value / 3.281)
            if key == 'area':
                results.append(value / 10.764)
            if key == 'pressure':
                results.append(value * 101325)

    elif from_units == 'si':
        for key, value in kwargs.items():
            if 'tmp' in key or key == 'tr' or key == 'ta':
                results.append((value * 9 / 5) + 32)
            if key in ['v', 'vr', 'vel']:
                results.append(value * 3.281)
            if key == 'area':
                results.append(value * 10.764)
            if key == 'pressure':
                results.append(value / 101325)

    return results


def t_o(ta, tr, v):
    """ Calculates operative temperature in accordance with ISO 7726:1998 [5]_

    Parameters
    ----------
    ta: float
        air temperature, [°C]
    tr: float
        mean radiant temperature temperature, [°C]
    v: float
        air velocity, [m/s]

    Returns
    -------
    to: float
        operative temperature, [°C]
    """

    return (ta * math.sqrt(10 * v) + tr) / (1 + math.sqrt(10 * v))


def t_mrt(tg, ta, v, d=.015, emissivity=0.9):
    """ Converts globe temperature reading into mean radiant temperature in accordance with ISO 7726:1998 [5]_

    Parameters
    ----------
    tg: float
        globe temperature, [°C]
    ta: float
        air temperature, [°C]
    v: float
        air velocity, [m/s]
    d: float
        diameter of the globe, [m]
    emissivity: float
        emissivity of the globe temperature sensor

    Returns
    -------
    tr: float
        mean radiant temperature, [°C]
    """
    tg += 273.15
    ta += 273.15

    # calculate heat transfer coefficient
    h_n = 1.4 * (abs(tg - ta) / d) ** 0.25  # natural convection
    h_f = 6.3 * v ** 0.6 / d ** 0.4  # forced convection

    # get the biggest between the tow coefficients
    h = max(h_f, h_n)
    print(h_n, h_f, h)

    tr = (tg ** 4 + h * (tg - ta) / (emissivity * (5.67 * 10 ** -8))) ** 0.25 - 273.15

    return round(tr, 1)
