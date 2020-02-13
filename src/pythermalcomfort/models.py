from pythermalcomfort.psychrometrics import *
from pythermalcomfort.utilities import *
import math


def pmv_ppd(ta, tr, vr, rh, met, clo, wme=0):
    """
    Returns Predicted Mean Vote (`PMV`_.) and Predicted Percentage of Dissatisfied (`PPD`_.) calculated in accordance with ASHRAE 55 2017 Standards.

    Parameters
    ----------
    ta : float
        dry bulb air temperature, [C]
    tr : float
        mean radiant temperature, [C]
    vr : float
        relative air velocity, [m/s]
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]
    wme : float
        external work, [met] default 0

    Returns
    -------
    PMV
        Predicted Mean Vote
    PPD
        Predicted Percentage of Dissatisfied occupants, [%]

    Notes
    -----
    You can use this function to calculate the `PMV`_. and `PPD`_. in accordance with the ASHRAE 55 2017 Standard [1]_.

    .. _PMV: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method
    .. _PPD: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import pmv_ppd
        >>> results = pmv_ppd(ta=25, tr=25, vr=0.1, rh=50, met=1.2, clo=0.5, wme=0)
        >>> print(results)
        {'pmv': 0.08, 'ppd': 5.1}

        >>> print(results['pmv'])
        0.08

    Raises
    ------
    StopIteration
        Raised if the number of iterations exceeds the threshold
    """
    check_standard_compliance(standard='ashrae', ta=ta, tr=tr, v=vr, rh=rh, met=met, clo=clo)
    check_standard_compliance(standard='iso', ta=ta, tr=tr, v=vr, rh=rh, met=met, clo=clo)

    pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))

    icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
    m = met * 58.15  # metabolic rate in W/M2
    w = wme * 58.15  # external work in W/M2
    mw = m - w  # internal heat production in the human body
    if icl <= 0.078:
        fcl = 1 + (1.29 * icl)
    else:
        fcl = 1.05 + (0.645 * icl)

    # heat transf. coeff. by forced convection
    hcf = 12.1 * math.sqrt(vr)
    taa = ta + 273
    tra = tr + 273
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * mw) + (p2 * math.pow(tra / 100.0, 4))
    xn = tcla / 100
    xf = tcla / 50
    eps = 0.00015

    n = 0
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * math.pow(abs(100.0 * xf - taa), 0.25)
        if (hcf > hcn):
            hc = hcf
        else:
            hc = hcn
        xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
        n += 1
        if n > 150:
            raise StopIteration('Max iterations exceeded')

    tcl = 100 * xn - 273

    # heat loss diff. through skin
    hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
    # heat loss by sweating
    if mw > 58.15:
        hl2 = 0.42 * (mw - 58.15)
    else:
        hl2 = 0
    # latent respiration heat loss
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    # dry respiration heat loss
    hl4 = 0.0014 * m * (34 - ta)
    # heat loss by radiation
    hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100.0, 4))
    # heat loss by convection
    hl6 = fcl * hc * (tcl - ta)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pow(pmv, 4.0) - 0.2179 * pow(pmv, 2.0))

    return {'pmv': round(pmv, 2), 'ppd': round(ppd, 1)}


def pmv(ta, tr, vr, rh, met, clo, wme=0):
    """
    Returns Predicted Mean Vote (`PMV`_.) calculated in accordance with ASHRAE 55 2017 Standards.

    Parameters
    ----------
    ta : float
        dry bulb air temperature, [C]
    tr : float
        mean radiant temperature, [C]
    vr : float
        relative air velocity, [m/s]
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]
    wme : float
        external work, [met] default 0

    Returns
    -------
    PMV : float
        Predicted Mean Vote

    Notes
    -----
    You can use this function to calculate the `PMV`_. in accordance with the ASHRAE 55 2017 Standard [1]_.

    .. _PMV: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import pmv
        >>> pmv(25, 25, 0.1, 50, 1.2, .5, wme=0)
        0.08
    """

    return pmv_ppd(ta, tr, vr, rh, met, clo, wme)['pmv']


def set_tmp(ta, tr, v, rh, met, clo, wme=0, body_surface_area=1.8258, p_atm=101.325):
    """ Standard effective temperature (SET) calculation using SI units

    Parameters
    ----------
    ta : float
        dry bulb air temperature, [C]
    tr : float
        mean radiant temperature, [C]
    v : float
        air velocity, [m/s]
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]
    wme : float
        external work, [met] default 0
    body_surface_area : float
        body surface area, [m2] default 1.8258
    p_atm : float
        atmospheric pressure, [kPa] default 101.325

    Returns
    -------
    SET : float
        Standard effective temperature, [C]

    Notes
    -----
    You can use this function to calculate the `SET`_. temperature in accordance with the ASHRAE 55 2017 Standard [1]_.

    .. _SET: https://en.wikipedia.org/wiki/Thermal_comfort#Standard_effective_temperature

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import set_tmp
        >>> set_tmp(ta=25, tr=25, v=0.1, rh=50, met=1.2, clo=.5)
        25.31
    """
    check_standard_compliance(standard='ashrae', ta=ta, tr=tr, v=v, rh=rh, met=met, clo=clo)

    # Initial variables as defined in the ASHRAE 55-2017
    vapor_pressure = rh * p_sat_torr(ta) / 100
    air_velocity = max(v, 0.1)
    k_clo = 0.25
    body_weight = 69.9
    met_factor = 58.2
    SBC = 0.000000056697  # Stefan-Boltzmann constant (W/m2K4)
    CSW = 170
    CDIL = 120
    CSTR = 0.5

    temp_skin_neutral = 33.7
    temp_core_neutral = 36.8
    temp_body_neutral = 36.49
    skin_blood_flow_neutral = 6.3

    temp_skin = temp_skin_neutral
    temp_core = temp_core_neutral
    skin_blood_flow = skin_blood_flow_neutral
    ALFA = 0.1
    ESK = 0.1 * met

    pressure_in_atmospheres = p_atm * 0.009869
    LTIME = 60
    RCL = 0.155 * clo

    FACL = 1.0 + 0.15 * clo  # INCREASE IN BODY SURFACE AREA DUE TO CLOTHING
    LR = 2.2 / pressure_in_atmospheres
    RM = met * met_factor
    M = met * met_factor

    if clo <= 0:
        WCRIT = 0.38 * pow(air_velocity, -0.29)
        ICL = 1.0
    else:
        WCRIT = 0.59 * pow(air_velocity, -0.08)
        ICL = 0.45

    CHC = 3.0 * pow(pressure_in_atmospheres, 0.53)
    CHCV = 8.600001 * pow((air_velocity * pressure_in_atmospheres), 0.53)
    CHC = max(CHC, CHCV)

    CHR = 4.7
    CTC = CHR + CHC
    RA = 1.0 / (FACL * CTC)
    TOP = (CHR * tr + CHC * ta) / CTC
    TCL = TOP + (temp_skin - TOP) / (CTC * (RA + RCL))

    TCL_OLD = False
    flag = True
    i = 0
    for TIM in range(LTIME):
        while abs(TCL - TCL_OLD) > 0.01:
            if flag:
                i += 1
                TCL_OLD = TCL
                CHR = 4.0 * SBC * pow(((TCL + tr) / 2.0 + 273.15), 3.0) * 0.72
                CTC = CHR + CHC
                RA = 1.0 / (FACL * CTC)
                TOP = (CHR * tr + CHC * ta) / CTC
            TCL = (RA * temp_skin + RCL * TOP) / (RA + RCL);
            flag = True
        flag = False
        DRY = (temp_skin - TOP) / (RA + RCL)
        HFCS = (temp_core - temp_skin) * (5.28 + 1.163 * skin_blood_flow)
        ERES = 0.0023 * M * (44.0 - vapor_pressure)
        CRES = 0.0014 * M * (34.0 - ta)
        SCR = M - HFCS - ERES - CRES - wme
        SSK = HFCS - DRY - ESK
        TCSK = 0.97 * ALFA * body_weight
        TCCR = 0.97 * (1 - ALFA) * body_weight
        DTSK = (SSK * body_surface_area) / (TCSK * 60.0)
        DTCR = SCR * body_surface_area / (TCCR * 60.0)
        temp_skin = temp_skin + DTSK
        temp_core = temp_core + DTCR
        TB = ALFA * temp_skin + (1 - ALFA) * temp_core
        SKSIG = temp_skin - temp_skin_neutral
        WARMS = (SKSIG > 0) * SKSIG
        COLDS = ((-1.0 * SKSIG) > 0) * (-1.0 * SKSIG)
        CRSIG = (temp_core - temp_core_neutral)
        WARMC = (CRSIG > 0) * CRSIG
        COLDC = ((-1.0 * CRSIG) > 0) * (-1.0 * CRSIG)
        BDSIG = TB - temp_body_neutral
        WARMB = (BDSIG > 0) * BDSIG
        skin_blood_flow = (skin_blood_flow_neutral + CDIL * WARMC) / (1 + CSTR * COLDS)
        if skin_blood_flow > 90.0:
            skin_blood_flow = 90.0
        if skin_blood_flow < 0.5:
            skin_blood_flow = 0.5
        REGSW = CSW * WARMB * math.exp(WARMS / 10.7)
        if REGSW > 500.0:
            REGSW = 500.0
        ERSW = 0.68 * REGSW
        REA = 1.0 / (LR * FACL * CHC)
        RECL = RCL / (LR * ICL)
        EMAX = (p_sat_torr(temp_skin) - vapor_pressure) / (REA + RECL)
        PRSW = ERSW / EMAX
        PWET = 0.06 + 0.94 * PRSW
        EDIF = PWET * EMAX - ERSW
        ESK = ERSW + EDIF
        if PWET > WCRIT:
            PWET = WCRIT
            PRSW = WCRIT / 0.94
            ERSW = PRSW * EMAX
            EDIF = 0.06 * (1.0 - PRSW) * EMAX
        if EMAX < 0:
            EDIF = 0
            ERSW = 0
            PWET = WCRIT
        ESK = ERSW + EDIF
        MSHIV = 19.4 * COLDS * COLDC
        M = RM + MSHIV
        ALFA = 0.0417737 + 0.7451833 / (skin_blood_flow + .585417)

    HSK = DRY + ESK
    W = PWET
    PSSK = p_sat_torr(temp_skin)
    CHRS = CHR
    if met < 0.85:
        CHCS = 3.0
    else:
        CHCS = 5.66 * pow(((met - 0.85)), 0.39)
    if (CHCS < 3.0):
        CHCS = 3.0
    CTCS = CHCS + CHRS
    RCLOS = 1.52 / ((met - wme / met_factor) + 0.6944) - 0.1835
    RCLS = 0.155 * RCLOS
    FACLS = 1.0 + k_clo * RCLOS
    FCLS = 1.0 / (1.0 + 0.155 * FACLS * CTCS * RCLOS)
    IMS = 0.45
    ICLS = IMS * CHCS / CTCS * (1 - FCLS) / (CHCS / CTCS - FCLS * IMS)
    RAS = 1.0 / (FACLS * CTCS)
    REAS = 1.0 / (LR * FACLS * CHCS)
    RECLS = RCLS / (LR * ICLS)
    HD_S = 1.0 / (RAS + RCLS)
    HE_S = 1.0 / (REAS + RECLS)

    DELTA = .0001
    dx = 100.0
    set_old = round(temp_skin - HSK / HD_S, 2)
    while abs(dx) > .01:
        ERR1 = (HSK - HD_S * (temp_skin - set_old) - W * HE_S * (PSSK - 0.5 * p_sat_torr(set_old)))
        ERR2 = (HSK - HD_S * (temp_skin - (set_old + DELTA)) - W * HE_S * (PSSK - 0.5 * p_sat_torr((set_old + DELTA))))
        set = set_old - DELTA * ERR1 / (ERR2 - ERR1)
        dx = set - set_old
        set_old = set
    return round(set, 1)


def adaptive_ashrae(ta, tr, t_running_mean, v):
    """ Determines the adaptive thermal comfort based on ASHRAE 55

    Parameters
    ----------
    ta : float
        dry bulb air temperature, [C]
    tr : float
        mean radiant temperature, [C]
    t_running_mean: float
        running mean temperature, [C]
    v : float
        air velocity, [m/s]


    Returns
    -------
    tmp_cmf : float
        Comfort temperature a that specific running mean temperature, [C]
    tmp_cmf_80_low : float
        Lower acceptable comfort temperature for 80% occupants, [C]
    tmp_cmf_80_up : float
        Upper acceptable comfort temperature for 80% occupants, [C]
    tmp_cmf_90_low : float
        Lower acceptable comfort temperature for 90% occupants, [C]
    tmp_cmf_90_up : float
        Upper acceptable comfort temperature for 90% occupants, [C]
    acceptability_80 : bol
        Acceptability for 80% occupants
    acceptability_90 : bol
        Acceptability for 90% occupants

    Notes
    -----
    You can use this function to calculate if your conditions are within the `adaptive thermal comfort region`.
    Calculations with comply with the ASHRAE 55 2017 Standard [1]_.

    .. _adaptive thermal comfort region: https://en.wikipedia.org/wiki/Thermal_comfort#Standard_effective_temperature

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import adaptive_ashrae
        >>> results = adaptive_ashrae(ta=25, tr=25, t_running_mean=20, v=0.1)
        >>> print(results)
        {'tmp_cmf': 24.0, 'tmp_cmf_80_low': 20.5, 'tmp_cmf_80_up': 27.5, 'tmp_cmf_90_low': 21.5, 'tmp_cmf_90_up': 26.5, 'acceptability_80': True, 'acceptability_90': False}

        >>> print(results['acceptability_80'])
        True
        # The conditions you entered are considered to be comfortable for by 80% of the occupants

        >>> results = adaptive_ashrae(ta=25, tr=25, t_running_mean=9, v=0.1)
        ValueError: The running mean is outside the standards applicability limits
        # The adaptive thermal comfort model can only be used
        # if the running mean temperature is higher than 10°C

    Raises
    ------
    ValueError
        Raised if the input are outside the Standard's applicability limits

    """
    check_standard_compliance(standard='ashrae', ta=ta, tr=tr, v=v)

    # Define the variables that will be used throughout the calculation.
    results = dict()

    to = (ta + tr) / 2  # fixme this is not the right way of calculating to

    # See if the running mean temperature is between 10 C and 33.5 C (the range where the adaptive model is supposed to be used)
    if 10.0 <= t_running_mean <= 33.5:

        cooling_effect = 0
        # calculate cooling effect of elevated air speed when top > 25 degC.
        if v >= 0.6 and to >= 25:
            if v < 0.9:
                cooling_effect = 1.2
            elif v < 1.2:
                cooling_effect = 1.8
            else:
                cooling_effect = 2.2

        # Figure out the relation between comfort and outdoor temperature depending on the level of conditioning.
        t_cmf = 0.31 * t_running_mean + 17.8
        tmp_cmf_80_low = t_cmf - 3.5
        t_cmf_90_lower = t_cmf - 2.5
        tmp_cmf_80_up = t_cmf + 3.5 + cooling_effect
        tmp_cmf_90_up = t_cmf + 2.5 + cooling_effect

        def acceptability(t_cmf_lower, t_cmf_upper):
            # See if the conditions are comfortable.
            if t_cmf_lower < to < t_cmf_upper:
                return True
            else:
                return False

        acceptability_80 = acceptability(tmp_cmf_80_low, tmp_cmf_80_up)
        acceptability_90 = acceptability(t_cmf_90_lower, t_cmf_90_lower)

        results = {'tmp_cmf': t_cmf, 'tmp_cmf_80_low': tmp_cmf_80_low, 'tmp_cmf_80_up': tmp_cmf_80_up,
                   'tmp_cmf_90_low': t_cmf_90_lower, 'tmp_cmf_90_up': tmp_cmf_90_up,
                   'acceptability_80': acceptability_80, 'acceptability_90': acceptability_90, }

    else:
        raise ValueError("The running mean is outside the standards applicability limits")

    return results


def utci(ta, tr, rh, v):
    """ Determines the Universal Thermal Climate Index (UTCI)

    Parameters
    ----------
    ta : float
        dry bulb air temperature, [C]
    tr : float
        mean radiant temperature, [C]
    rh: float
        relative humidity, [%]
    v : float
        wind speed, [m/s]


    Returns
    -------
    utci : float
         Universal Thermal Climate Index, [C]

    Notes
    -----
    You can use this function to calculate the Universal Thermal Climate Index (`UTCI`)
    The applicability wind speed value must be between 0.5 and 17 m/s.

    .. _UTCI: http://www.utci.org/utcineu/utcineu.php

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import adaptive_ashrae
        >>> results = adaptive_ashrae(ta=25, tr=25, t_running_mean=20, v=0.1)
        >>> print(results)
        {'tmp_cmf': 24.0, 'tmp_cmf_80_low': 20.5, 'tmp_cmf_80_up': 27.5, 'tmp_cmf_90_low': 21.5, 'tmp_cmf_90_up': 26.5, 'acceptability_80': True, 'acceptability_90': False}

        >>> print(results['acceptability_80'])
        True
        # The conditions you entered are considered to be comfortable for by 80% of the occupants

        >>> results = adaptive_ashrae(ta=25, tr=25, t_running_mean=9, v=0.1)
        ValueError: The running mean is outside the standards applicability limits
        # The adaptive thermal comfort model can only be used
        # if the running mean temperature is higher than 10°C

    Raises
    ------
    ValueError
        Raised if the input are outside the Standard's applicability limits

    """
    check_standard_compliance(standard='utci', ta=ta, tr=tr, v=v)

    def es(ta):
        g = [
            -2836.5744, -6028.076559, 19.54263612,
            -0.02737830188, 0.000016261698,
            (7.0229056 * (10 ** (-10))), (-1.8680009 * (10 ** (-13)))]
        tk = ta + 273.15  # air temp in K
        es = 2.7150305 * math.log1p(tk)
        for count, i in enumerate(g):
            es = es + (i * (tk ** (count - 2)))
        es = math.exp(es) * 0.01  # convert Pa to hPa
        return es

    # Do a series of checks to be sure that the input values are within
    # the bounds accepted by the model.
    # check = (ta < 50.0 or ta > 50.0 or tr - ta < -30.0 or tr - ta > 70.0)  # fixme if this does not pass raise error
    check = True
    if v < 0.5:  # todo decide if it is good to continue with the code
        v = 0.5
    elif v > 17:
        v = 17

    # If everything is good, run the data through the model below to get
    # the UTCI.
    # This is a python version of the UTCI_approx function
    # Version a 0.002, October 2009
    # Ta: air temperature, degrees Celsius
    # ehPa: water vapour presure, hPa=hecto Pascal
    # Tmrt: mean radiant temperature, degrees Celsius
    # va10m: wind speed 10m above ground level in m/s

    if check == True:
        ehPa = es(ta) * (rh / 100.0)
        D_Tmrt = tr - ta
        Pa = ehPa / 10.0  # convert vapour pressure to kPa

        UTCI_approx = (ta +
                       (0.607562052) +
                       (-0.0227712343) * ta +
                       (8.06470249 * (10 ** (-4))) * ta * ta +
                       (-1.54271372 * (10 ** (-4))) * ta * ta * ta +
                       (-3.24651735 * (10 ** (-6))) * ta * ta * ta * ta +
                       (7.32602852 * (10 ** (-8))) * ta * ta * ta * ta * ta +
                       (1.35959073 * (10 ** (-9))) * ta * ta * ta * ta * ta * ta +
                       (-2.25836520) * v +
                       (0.0880326035) * ta * v +
                       (0.00216844454) * ta * ta * v +
                       (-1.53347087 * (10 ** (-5))) * ta * ta * ta * v +
                       (-5.72983704 * (10 ** (-7))) * ta * ta * ta * ta * v +
                       (-2.55090145 * (10 ** (-9))) * ta * ta * ta * ta * ta * v +
                       (-0.751269505) * v * v +
                       (-0.00408350271) * ta * v * v +
                       (-5.21670675 * (10 ** (-5))) * ta * ta * v * v +
                       (1.94544667 * (10 ** (-6))) * ta * ta * ta * v * v +
                       (1.14099531 * (10 ** (-8))) * ta * ta * ta * ta * v * v +
                       (0.158137256) * v * v * v +
                       (-6.57263143 * (10 ** (-5))) * ta * v * v * v +
                       (2.22697524 * (10 ** (-7))) * ta * ta * v * v * v +
                       (-4.16117031 * (10 ** (-8))) * ta * ta * ta * v * v * v +
                       (-0.0127762753) * v * v * v * v +
                       (9.66891875 * (10 ** (-6))) * ta * v * v * v * v +
                       (2.52785852 * (10 ** (-9))) * ta * ta * v * v * v * v +
                       (4.56306672 * (10 ** (-4))) * v * v * v * v * v +
                       (-1.74202546 * (10 ** (-7))) * ta * v * v * v * v * v +
                       (-5.91491269 * (10 ** (-6))) * v * v * v * v * v * v +
                       (0.398374029) * D_Tmrt +
                       (1.83945314 * (10 ** (-4))) * ta * D_Tmrt +
                       (-1.73754510 * (10 ** (-4))) * ta * ta * D_Tmrt +
                       (-7.60781159 * (10 ** (-7))) * ta * ta * ta * D_Tmrt +
                       (3.77830287 * (10 ** (-8))) * ta * ta * ta * ta * D_Tmrt +
                       (5.43079673 * (10 ** (-10))) * ta * ta * ta * ta * ta * D_Tmrt +
                       (-0.0200518269) * v * D_Tmrt +
                       (8.92859837 * (10 ** (-4))) * ta * v * D_Tmrt +
                       (3.45433048 * (10 ** (-6))) * ta * ta * v * D_Tmrt +
                       (-3.77925774 * (10 ** (-7))) * ta * ta * ta * v * D_Tmrt +
                       (-1.69699377 * (10 ** (-9))) * ta * ta * ta * ta * v * D_Tmrt +
                       (1.69992415 * (10 ** (-4))) * v * v * D_Tmrt +
                       (-4.99204314 * (10 ** (-5))) * ta * v * v * D_Tmrt +
                       (2.47417178 * (10 ** (-7))) * ta * ta * v * v * D_Tmrt +
                       (1.07596466 * (10 ** (-8))) * ta * ta * ta * v * v * D_Tmrt +
                       (8.49242932 * (10 ** (-5))) * v * v * v * D_Tmrt +
                       (1.35191328 * (10 ** (-6))) * ta * v * v * v * D_Tmrt +
                       (-6.21531254 * (10 ** (-9))) * ta * ta * v * v * v * D_Tmrt +
                       (-4.99410301 * (10 ** (-6))) * v * v * v * v * D_Tmrt +
                       (-1.89489258 * (10 ** (-8))) * ta * v * v * v * v * D_Tmrt +
                       (8.15300114 * (10 ** (-8))) * v * v * v * v * v * D_Tmrt +
                       (7.55043090 * (10 ** (-4))) * D_Tmrt * D_Tmrt +
                       (-5.65095215 * (10 ** (-5))) * ta * D_Tmrt * D_Tmrt +
                       (-4.52166564 * (10 ** (-7))) * ta * ta * D_Tmrt * D_Tmrt +
                       (2.46688878 * (10 ** (-8))) * ta * ta * ta * D_Tmrt * D_Tmrt +
                       (2.42674348 * (10 ** (-10))) * ta * ta * ta * ta * D_Tmrt * D_Tmrt +
                       (1.54547250 * (10 ** (-4))) * v * D_Tmrt * D_Tmrt +
                       (5.24110970 * (10 ** (-6))) * ta * v * D_Tmrt * D_Tmrt +
                       (-8.75874982 * (10 ** (-8))) * ta * ta * v * D_Tmrt * D_Tmrt +
                       (-1.50743064 * (10 ** (-9))) * ta * ta * ta * v * D_Tmrt * D_Tmrt +
                       (-1.56236307 * (10 ** (-5))) * v * v * D_Tmrt * D_Tmrt +
                       (-1.33895614 * (10 ** (-7))) * ta * v * v * D_Tmrt * D_Tmrt +
                       (2.49709824 * (10 ** (-9))) * ta * ta * v * v * D_Tmrt * D_Tmrt +
                       (6.51711721 * (10 ** (-7))) * v * v * v * D_Tmrt * D_Tmrt +
                       (1.94960053 * (10 ** (-9))) * ta * v * v * v * D_Tmrt * D_Tmrt +
                       (-1.00361113 * (10 ** (-8))) * v * v * v * v * D_Tmrt * D_Tmrt +
                       (-1.21206673 * (10 ** (-5))) * D_Tmrt * D_Tmrt * D_Tmrt +
                       (-2.18203660 * (10 ** (-7))) * ta * D_Tmrt * D_Tmrt * D_Tmrt +
                       (7.51269482 * (10 ** (-9))) * ta * ta * D_Tmrt * D_Tmrt * D_Tmrt +
                       (9.79063848 * (10 ** (-11))) * ta * ta * ta * D_Tmrt * D_Tmrt * D_Tmrt +
                       (1.25006734 * (10 ** (-6))) * v * D_Tmrt * D_Tmrt * D_Tmrt +
                       (-1.81584736 * (10 ** (-9))) * ta * v * D_Tmrt * D_Tmrt * D_Tmrt +
                       (-3.52197671 * (10 ** (-10))) * ta * ta * v * D_Tmrt * D_Tmrt * D_Tmrt +
                       (-3.36514630 * (10 ** (-8))) * v * v * D_Tmrt * D_Tmrt * D_Tmrt +
                       (1.35908359 * (10 ** (-10))) * ta * v * v * D_Tmrt * D_Tmrt * D_Tmrt +
                       (4.17032620 * (10 ** (-10))) * v * v * v * D_Tmrt * D_Tmrt * D_Tmrt +
                       (-1.30369025 * (10 ** (-9))) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt +
                       (4.13908461 * (10 ** (-10))) * ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt +
                       (9.22652254 * (10 ** (-12))) * ta * ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt +
                       (-5.08220384 * (10 ** (-9))) * v * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt +
                       (-2.24730961 * (10 ** (-11))) * ta * v * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt +
                       (1.17139133 * (10 ** (-10))) * v * v * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt +
                       (6.62154879 * (10 ** (-10))) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt +
                       (4.03863260 * (10 ** (-13))) * ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt +
                       (1.95087203 * (10 ** (-12))) * v * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt +
                       (-4.73602469 * (10 ** (-12))) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt +
                       (5.12733497) * Pa +
                       (-0.312788561) * ta * Pa +
                       (-0.0196701861) * ta * ta * Pa +
                       (9.99690870 * (10 ** (-4))) * ta * ta * ta * Pa +
                       (9.51738512 * (10 ** (-6))) * ta * ta * ta * ta * Pa +
                       (-4.66426341 * (10 ** (-7))) * ta * ta * ta * ta * ta * Pa +
                       (0.548050612) * v * Pa +
                       (-0.00330552823) * ta * v * Pa +
                       (-0.00164119440) * ta * ta * v * Pa +
                       (-5.16670694 * (10 ** (-6))) * ta * ta * ta * v * Pa +
                       (9.52692432 * (10 ** (-7))) * ta * ta * ta * ta * v * Pa +
                       (-0.0429223622) * v * v * Pa +
                       (0.00500845667) * ta * v * v * Pa +
                       (1.00601257 * (10 ** (-6))) * ta * ta * v * v * Pa +
                       (-1.81748644 * (10 ** (-6))) * ta * ta * ta * v * v * Pa +
                       (-1.25813502 * (10 ** (-3))) * v * v * v * Pa +
                       (-1.79330391 * (10 ** (-4))) * ta * v * v * v * Pa +
                       (2.34994441 * (10 ** (-6))) * ta * ta * v * v * v * Pa +
                       (1.29735808 * (10 ** (-4))) * v * v * v * v * Pa +
                       (1.29064870 * (10 ** (-6))) * ta * v * v * v * v * Pa +
                       (-2.28558686 * (10 ** (-6))) * v * v * v * v * v * Pa +
                       (-0.0369476348) * D_Tmrt * Pa +
                       (0.00162325322) * ta * D_Tmrt * Pa +
                       (-3.14279680 * (10 ** (-5))) * ta * ta * D_Tmrt * Pa +
                       (2.59835559 * (10 ** (-6))) * ta * ta * ta * D_Tmrt * Pa +
                       (-4.77136523 * (10 ** (-8))) * ta * ta * ta * ta * D_Tmrt * Pa +
                       (8.64203390 * (10 ** (-3))) * v * D_Tmrt * Pa +
                       (-6.87405181 * (10 ** (-4))) * ta * v * D_Tmrt * Pa +
                       (-9.13863872 * (10 ** (-6))) * ta * ta * v * D_Tmrt * Pa +
                       (5.15916806 * (10 ** (-7))) * ta * ta * ta * v * D_Tmrt * Pa +
                       (-3.59217476 * (10 ** (-5))) * v * v * D_Tmrt * Pa +
                       (3.28696511 * (10 ** (-5))) * ta * v * v * D_Tmrt * Pa +
                       (-7.10542454 * (10 ** (-7))) * ta * ta * v * v * D_Tmrt * Pa +
                       (-1.24382300 * (10 ** (-5))) * v * v * v * D_Tmrt * Pa +
                       (-7.38584400 * (10 ** (-9))) * ta * v * v * v * D_Tmrt * Pa +
                       (2.20609296 * (10 ** (-7))) * v * v * v * v * D_Tmrt * Pa +
                       (-7.32469180 * (10 ** (-4))) * D_Tmrt * D_Tmrt * Pa +
                       (-1.87381964 * (10 ** (-5))) * ta * D_Tmrt * D_Tmrt * Pa +
                       (4.80925239 * (10 ** (-6))) * ta * ta * D_Tmrt * D_Tmrt * Pa +
                       (-8.75492040 * (10 ** (-8))) * ta * ta * ta * D_Tmrt * D_Tmrt * Pa +
                       (2.77862930 * (10 ** (-5))) * v * D_Tmrt * D_Tmrt * Pa +
                       (-5.06004592 * (10 ** (-6))) * ta * v * D_Tmrt * D_Tmrt * Pa +
                       (1.14325367 * (10 ** (-7))) * ta * ta * v * D_Tmrt * D_Tmrt * Pa +
                       (2.53016723 * (10 ** (-6))) * v * v * D_Tmrt * D_Tmrt * Pa +
                       (-1.72857035 * (10 ** (-8))) * ta * v * v * D_Tmrt * D_Tmrt * Pa +
                       (-3.95079398 * (10 ** (-8))) * v * v * v * D_Tmrt * D_Tmrt * Pa +
                       (-3.59413173 * (10 ** (-7))) * D_Tmrt * D_Tmrt * D_Tmrt * Pa +
                       (7.04388046 * (10 ** (-7))) * ta * D_Tmrt * D_Tmrt * D_Tmrt * Pa +
                       (-1.89309167 * (10 ** (-8))) * ta * ta * D_Tmrt * D_Tmrt * D_Tmrt * Pa +
                       (-4.79768731 * (10 ** (-7))) * v * D_Tmrt * D_Tmrt * D_Tmrt * Pa +
                       (7.96079978 * (10 ** (-9))) * ta * v * D_Tmrt * D_Tmrt * D_Tmrt * Pa +
                       (1.62897058 * (10 ** (-9))) * v * v * D_Tmrt * D_Tmrt * D_Tmrt * Pa +
                       (3.94367674 * (10 ** (-8))) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa +
                       (-1.18566247 * (10 ** (-9))) * ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa +
                       (3.34678041 * (10 ** (-10))) * v * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa +
                       (-1.15606447 * (10 ** (-10))) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa +
                       (-2.80626406) * Pa * Pa +
                       (0.548712484) * ta * Pa * Pa +
                       (-0.00399428410) * ta * ta * Pa * Pa +
                       (-9.54009191 * (10 ** (-4))) * ta * ta * ta * Pa * Pa +
                       (1.93090978 * (10 ** (-5))) * ta * ta * ta * ta * Pa * Pa +
                       (-0.308806365) * v * Pa * Pa +
                       (0.0116952364) * ta * v * Pa * Pa +
                       (4.95271903 * (10 ** (-4))) * ta * ta * v * Pa * Pa +
                       (-1.90710882 * (10 ** (-5))) * ta * ta * ta * v * Pa * Pa +
                       (0.00210787756) * v * v * Pa * Pa +
                       (-6.98445738 * (10 ** (-4))) * ta * v * v * Pa * Pa +
                       (2.30109073 * (10 ** (-5))) * ta * ta * v * v * Pa * Pa +
                       (4.17856590 * (10 ** (-4))) * v * v * v * Pa * Pa +
                       (-1.27043871 * (10 ** (-5))) * ta * v * v * v * Pa * Pa +
                       (-3.04620472 * (10 ** (-6))) * v * v * v * v * Pa * Pa +
                       (0.0514507424) * D_Tmrt * Pa * Pa +
                       (-0.00432510997) * ta * D_Tmrt * Pa * Pa +
                       (8.99281156 * (10 ** (-5))) * ta * ta * D_Tmrt * Pa * Pa +
                       (-7.14663943 * (10 ** (-7))) * ta * ta * ta * D_Tmrt * Pa * Pa +
                       (-2.66016305 * (10 ** (-4))) * v * D_Tmrt * Pa * Pa +
                       (2.63789586 * (10 ** (-4))) * ta * v * D_Tmrt * Pa * Pa +
                       (-7.01199003 * (10 ** (-6))) * ta * ta * v * D_Tmrt * Pa * Pa +
                       (-1.06823306 * (10 ** (-4))) * v * v * D_Tmrt * Pa * Pa +
                       (3.61341136 * (10 ** (-6))) * ta * v * v * D_Tmrt * Pa * Pa +
                       (2.29748967 * (10 ** (-7))) * v * v * v * D_Tmrt * Pa * Pa +
                       (3.04788893 * (10 ** (-4))) * D_Tmrt * D_Tmrt * Pa * Pa +
                       (-6.42070836 * (10 ** (-5))) * ta * D_Tmrt * D_Tmrt * Pa * Pa +
                       (1.16257971 * (10 ** (-6))) * ta * ta * D_Tmrt * D_Tmrt * Pa * Pa +
                       (7.68023384 * (10 ** (-6))) * v * D_Tmrt * D_Tmrt * Pa * Pa +
                       (-5.47446896 * (10 ** (-7))) * ta * v * D_Tmrt * D_Tmrt * Pa * Pa +
                       (-3.59937910 * (10 ** (-8))) * v * v * D_Tmrt * D_Tmrt * Pa * Pa +
                       (-4.36497725 * (10 ** (-6))) * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa +
                       (1.68737969 * (10 ** (-7))) * ta * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa +
                       (2.67489271 * (10 ** (-8))) * v * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa +
                       (3.23926897 * (10 ** (-9))) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa +
                       (-0.0353874123) * Pa * Pa * Pa +
                       (-0.221201190) * ta * Pa * Pa * Pa +
                       (0.0155126038) * ta * ta * Pa * Pa * Pa +
                       (-2.63917279 * (10 ** (-4))) * ta * ta * ta * Pa * Pa * Pa +
                       (0.0453433455) * v * Pa * Pa * Pa +
                       (-0.00432943862) * ta * v * Pa * Pa * Pa +
                       (1.45389826 * (10 ** (-4))) * ta * ta * v * Pa * Pa * Pa +
                       (2.17508610 * (10 ** (-4))) * v * v * Pa * Pa * Pa +
                       (-6.66724702 * (10 ** (-5))) * ta * v * v * Pa * Pa * Pa +
                       (3.33217140 * (10 ** (-5))) * v * v * v * Pa * Pa * Pa +
                       (-0.00226921615) * D_Tmrt * Pa * Pa * Pa +
                       (3.80261982 * (10 ** (-4))) * ta * D_Tmrt * Pa * Pa * Pa +
                       (-5.45314314 * (10 ** (-9))) * ta * ta * D_Tmrt * Pa * Pa * Pa +
                       (-7.96355448 * (10 ** (-4))) * v * D_Tmrt * Pa * Pa * Pa +
                       (2.53458034 * (10 ** (-5))) * ta * v * D_Tmrt * Pa * Pa * Pa +
                       (-6.31223658 * (10 ** (-6))) * v * v * D_Tmrt * Pa * Pa * Pa +
                       (3.02122035 * (10 ** (-4))) * D_Tmrt * D_Tmrt * Pa * Pa * Pa +
                       (-4.77403547 * (10 ** (-6))) * ta * D_Tmrt * D_Tmrt * Pa * Pa * Pa +
                       (1.73825715 * (10 ** (-6))) * v * D_Tmrt * D_Tmrt * Pa * Pa * Pa +
                       (-4.09087898 * (10 ** (-7))) * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa * Pa +
                       (0.614155345) * Pa * Pa * Pa * Pa +
                       (-0.0616755931) * ta * Pa * Pa * Pa * Pa +
                       (0.00133374846) * ta * ta * Pa * Pa * Pa * Pa +
                       (0.00355375387) * v * Pa * Pa * Pa * Pa +
                       (-5.13027851 * (10 ** (-4))) * ta * v * Pa * Pa * Pa * Pa +
                       (1.02449757 * (10 ** (-4))) * v * v * Pa * Pa * Pa * Pa +
                       (-0.00148526421) * D_Tmrt * Pa * Pa * Pa * Pa +
                       (-4.11469183 * (10 ** (-5))) * ta * D_Tmrt * Pa * Pa * Pa * Pa +
                       (-6.80434415 * (10 ** (-6))) * v * D_Tmrt * Pa * Pa * Pa * Pa +
                       (-9.77675906 * (10 ** (-6))) * D_Tmrt * D_Tmrt * Pa * Pa * Pa * Pa +
                       (0.0882773108) * Pa * Pa * Pa * Pa * Pa +
                       (-0.00301859306) * ta * Pa * Pa * Pa * Pa * Pa +
                       (0.00104452989) * v * Pa * Pa * Pa * Pa * Pa +
                       (2.47090539 * (10 ** (-4))) * D_Tmrt * Pa * Pa * Pa * Pa * Pa +
                       (0.00148348065) * Pa * Pa * Pa * Pa * Pa * Pa)

        cmf = UTCI_approx > 9 and UTCI_approx < 26

        if UTCI_approx < -14.0:
            stress_range = -2
        elif UTCI_approx < 9.0:
            stress_range = -1
        elif UTCI_approx < 26.0:
            stress_range = 0
        elif UTCI_approx < 32.0:
            stress_range = 1
        else:
            stress_range = 2
    else:
        UTCI_approx = None
        cmf = None
        stress_range = None

    return {'utci': round(UTCI_approx, 1), 'cmf': cmf, 'stress_range': stress_range}
