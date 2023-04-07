# -*- coding: utf-8 -*-
import math


def pmv(ta, tr, va, rh, met, clo, wmet=0):
    """
    Get PMV value based on the 2017 ASHRAE Handbook—Fundamentals, Chapter 9:
    Thermal Comfort, Equations 63 - 68.
    Parameters
    ----------
    ta : float, optional
        Air temperature [oC]
    tr : float, optional
        Mean radiant temperature [oC]
    va : float, optional
        Air velocity [m/s]
    rh : float, optional
        Relative humidity [%]
    met : float, optional
        Metabolic rate [met]
    clo : float, optional
        Clothing insulation [clo]
    wmet : float, optional
        External work [met], optional. The default is 0.
    Returns
    -------
    PMV value
    """

    met *= 58.15  # chage unit [met] to [W/m2]
    wmet *= 58.15  # chage unit [met] to [W/m2]
    mw = met - wmet  # heat production [W/m2]

    if clo < 0.5:
        fcl = 1 + 0.2 * clo  # clothing area factor [-]
    else:
        fcl = 1.05 + 0.1 * clo

    antoine = lambda x: math.e ** (
        16.6536 - (4030.183 / (x + 235))
    )  # antoine's formula
    pa = antoine(ta) * rh / 100  # vapor pressure [kPa]
    rcl = 0.155 * clo  # clothing thermal resistance [K.m2/W]
    hcf = 12.1 * va**0.5  # forced convective heat transfer coefficience

    hc = hcf  # initial convective heat transfer coefficience
    tcl = (34 + ta) / 2  # initial clothing temp.

    # Cal. clothing temp. by iterative calculation method
    for i in range(100):
        # clothing temp. [oC]
        tcliter = (
            35.7
            - 0.028 * mw
            - rcl
            * (
                39.6 * 10 ** (-9) * fcl * ((tcl + 273) ** 4 - (tr + 273) ** 4)
                + fcl * hc * (tcl - ta)
            )
        )  # Eq.68
        # new clothin temp. [oC]
        tcl = (tcliter + tcl) / 2

        hcn = (
            2.38 * abs(tcl - ta) ** 0.25
        )  # natural convective heat transfer coefficience

        # select forced or natural convection
        if hcn > hcf:
            hc = hcf
        else:
            hc = hcf

        # terminate iterative calculation
        if abs(tcliter - tcl) < 0.0001:
            break

    # tcl = 35.7 - 0.0275 * mw \
    #     - rcl * (mw - 3.05 * (5.73 - 0.007 * mw - pa) \
    #     - 0.42 * (mw - 58.15) - 0.0173 * met * (5.87 - pa) \
    #     + 0.0014 * met * (34 - ta))  # Eq.64

    # Heat loss of human body
    rad = (
        3.96 * (10 ** (-8)) * fcl * ((tcl + 273) ** 4 - (tr + 273) ** 4)
    )  # by radiation
    conv = fcl * hc * (tcl - ta)  # by convction
    diff = 3.05 * (5.73 - 0.007 * mw - pa)  # by insensive perspiration
    sweat = max(0, 0.42 * (mw - 58.15))  # by sweating
    res = 0.0173 * met * (5.87 - pa) + 0.0014 * met * (34 - ta)  # by repiration
    load = mw - rad - conv - diff - sweat - res

    pmv_value = (0.303 * math.exp(-0.036 * met) + 0.028) * load  # Eq.63

    return pmv_value


def preferred_temp(va=0.1, rh=50, met=1, clo=0):
    """
    Calculate operative temperature [oC] at PMV=0.
    Parameters
    ----------
    va : float, optional
        Air velocity [m/s]. The default is 0.1.
    rh : float, optional
        Relative humidity [%]. The default is 50.
    met : float, optional
        Metabolic rate [met]. The default is 1.
    clo : float, optional
        Clothing insulation [clo]. The default is 0.
    Returns
    -------
    to : float
        Operative temperature [oC].
    """

    to = 28  # initial temp
    # Iterate until the PMV (Predicted Mean Vote) value is less than 0.001
    for i in range(100):
        vpmv = pmv(to, to, va, rh, met, clo)
        # Break the loop if the absolute value of PMV is less than 0.001
        if abs(vpmv) < 0.001:
            break
        # Update the temperature based on the PMV value
        else:
            to = to - vpmv / 3
    return to


# def preferred_temp(va=0.1, RH=50, met=1, clo=0):
#     """
#     Calculate operative temperature [oC] at PMV=0.
#     This is for calculating body set-point temperature
#
#     Parameters
#     ----------
#     va : float, optional
#         Air velocity [m/s]. The default is 0.1.
#     RH : float, optional
#         Relative humidity [%]. The default is 50.
#     met : float, optional
#         Metabolic rate [met]. The default is 1.
#     clo : float, optional
#         Clothing insulation [clo]. The default is 0.
#
#     Returns
#     -------
#     to : float
#         Operative temperature [oC].
#     """
#
#     to = 28  # initial temp
#     # Iterate until the PMV (Predicted Mean Vote) value is less than 0.001
#     for i in range(100):
#         vpmv = pmv(to, to, Va, RH, met, clo)
#         # Break the loop if the absolute value of PMV is less than 0.001
#         if abs(vpmv) < 0.001:
#             break
#         else:
#             # Update the temperature based on the PMV value
#             to = to - vpmv / 3
#     return to
