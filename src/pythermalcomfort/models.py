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
    PMV : float
        Predicted Mean Vote
    PPD : float
        Predicted Percentage of Dissatisfied occupants, [%]

    Notes
    -----
    You can use this function to calculate the `PMV`_. and `PPD`_. in accordance with the ASHRAE 55 2017 Standard [1].
    More information about the `PMV`_. model are available in the Wikipedia page.

    .. _PMV: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method
    .. _PPD: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method

    References
    --------
    [1]  ANSI, & ASHRAE. (2017). Thermal Environmental Conditions for Human Occupancy. Atlanta.

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import pmv_ppd
        >>> results = pmv_ppd(25, 25, 0.1, 50, 1.2, .5, wme=0)
        >>> print(results)
        {'pmv': 0.08425176342008413, 'ppd': 5.146986265266861}

        >>> print(results['pmv'])
        0.08425176342008413
    """

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
            print('Max iterations exceeded')
            return 1

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

    return {'pmv': pmv, 'ppd': ppd}


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
    You can use this function to calculate the `PMV`_. in accordance with the ASHRAE 55 2017 Standard [1].
    More information about the `PMV`_. model are available in the Wikipedia page.

    .. _PMV: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method

    References
    --------
    [1]  ANSI, & ASHRAE. (2017). Thermal Environmental Conditions for Human Occupancy. Atlanta.

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import pmv
        >>> pmv(25, 25, 0.1, 50, 1.2, .5, wme=0)
        0.08425176342008413
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
        relative air velocity, [m/s]
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
    You can use this function to calculate the `SET`_. temperature in accordance with the ASHRAE 55 2017 Standard [1].
    More information about the `SET`_. model are available in the Wikipedia page.

    .. _SET: https://en.wikipedia.org/wiki/Thermal_comfort#Standard_effective_temperature

    References
    --------
    [1]  ANSI, & ASHRAE. (2017). Thermal Environmental Conditions for Human Occupancy. Atlanta.

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import set_tmp
        >>> set_tmp(ta=25, tr=25, v=0.1, rh=50, met=1.2, clo=.5)
        25.31276389616353
    """
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
    return set
