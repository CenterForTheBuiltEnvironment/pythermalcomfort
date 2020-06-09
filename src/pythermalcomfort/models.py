from pythermalcomfort.psychrometrics import *
from pythermalcomfort.utilities import *
import math


def cooling_effect(tdb, tr, vr, rh, met, clo, wme=0, units='SI'):
    """
    Returns the value of the Cooling Effect (`CE`_) calculated in compliance with the ASHRAE 55 2017 Standard [1]_. The `CE`_ of the elevated air speed is the value that,
    when subtracted equally from both the average air temperature and the mean radiant temperature, yields the same `SET`_ under still air as in the first `SET`_ calculation
    under elevated air speed.

    .. _CE: https://en.wikipedia.org/wiki/Thermal_comfort#Cooling_Effect

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    vr : float
        relative air velocity, default in [m/s] in [fps] if `units` = 'IP'

        Note: vr is the relative air velocity caused by body movement and not the air speed measured by the air velocity sensor.
        It can be calculate using the function :py:meth:`pythermalcomfort.psychrometrics.v_relative`.
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]
    wme : float
        external work, [met] default 0
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    ce
        Cooling Effect, default in [°C] in [°F] if `units` = 'IP'

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import cooling_effect
        >>> ce = cooling_effect(tdb=25, tr=25, vr=0.3, rh=50, met=1.2, clo=0.5)
        >>> print(ce)
        1.64

        >>> # for users who wants to use the IP system
        >>> ce = cooling_effect(tdb=77, tr=77, vr=1.64, rh=50, met=1, clo=0.6, units="IP")
        >>> print(ce)
        3.74

    Raises
    ------
    ValueError
        If the cooling effect could not be calculated
    """

    if units.lower() == 'ip':
        tdb, tr, vr = units_converter(tdb=tdb, tr=tr, v=vr)

    still_air_threshold = 0.1

    warnings.simplefilter("ignore")
    # ce = secant(lambda x: set_tmp(tdb - x, tr - x, v=still_air_threshold, rh=rh, met=met, clo=clo, wme=wme) - set_tmp(tdb=tdb, tr=tr, v=vr, rh=rh, met=met, clo=clo, wme=wme), 0, 15, 150)
    ce = bisection(lambda x: set_tmp(tdb - x, tr - x, v=still_air_threshold, rh=rh, met=met, clo=clo) - set_tmp(tdb=tdb, tr=tr, v=vr, rh=rh, met=met, clo=clo), 0.0, 15.0, 150)
    if ce is None:
        raise ValueError("It could not calculate the cooling effect")
    warnings.simplefilter("always")

    if units.lower() == 'ip':
        ce = ce / 1.8 * 3.28

    return round(ce, 2)


def pmv_ppd(tdb, tr, vr, rh, met, clo, wme=0, standard='ISO', units='SI'):
    """
    Returns Predicted Mean Vote (`PMV`_) and Predicted Percentage of Dissatisfied (`PPD`_) calculated in accordance to main thermal comfort Standards. The `PMV`_ is an index that
    predicts the mean value of the thermal sensation votes (self-reported perceptions) of a large group of people on a sensation scale expressed from –3 to +3 corresponding to
    the categories \"cold,\" \"cool,\" \"slightly cool,\" \"neutral,\" \"slightly warm,\" \"warm,\" and \"hot.\"[1]_. The `PPD`_ is an index that establishes a quantitative
    prediction of the percentage of thermally dissatisfied people determined from `PMV`_ [1]_.

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    vr : float
        relative air velocity, default in [m/s] in [fps] if `units` = 'IP'

        Note: vr is the relative air velocity caused by body movement and not the air speed measured by the air velocity sensor.
        The relative air velocity can be calculate using the function :py:meth:`pythermalcomfort.psychrometrics.v_relative`.
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]

        Note: The ASHRAE 55 Standard suggests that the dynamic clothing insulation is used as input in the PMV model.
        The dynamic clothing insulation can be calculated using the function :py:meth:`pythermalcomfort.psychrometrics.clo_dynamic`.
    wme : float
        external work, [met] default 0
    standard: str (default="ISO")
        comfort standard used for calculation

        - If "ISO", then the ISO Equation is used
        - If "ASHRAE", then the ASHRAE Equation is used

        Note: While the PMV equation is the same for both the ISO and ASHRAE standards,
        the ASHRAE Standard Use of the PMV model is limited to air speeds below 0.20 m/s (40 fpm).
        When air speeds exceed 0.20 m/s (40 fpm), the comfort zone boundaries are adjusted based on the SET model.
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    PMV
        Predicted Mean Vote
    PPD
        Predicted Percentage of Dissatisfied occupants, [%]

    Notes
    -----
    You can use this function to calculate the `PMV`_ and `PPD`_ in accordance with either the ASHRAE 55 2017 Standard [1]_ or the ISO 7730 Standard [2]_.

    .. _PMV: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method
    .. _PPD: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import pmv_ppd
        >>> results = pmv_ppd(tdb=25, tr=25, vr=0.1, rh=50, met=1.2, clo=0.5, wme=0, standard="ISO")
        >>> print(results)
        {'pmv': 0.08, 'ppd': 5.1}

        >>> print(results['pmv'])
        0.08

        >>> # for users who wants to use the IP system
        >>> results_ip = pmv_ppd(tdb=77, tr=77, vr=0.4, rh=50, met=1.2, clo=0.5, units="IP")
        >>> print(results_ip)
        {'pmv': 0.01, 'ppd': 5.0}

    Raises
    ------
    StopIteration
        Raised if the number of iterations exceeds the threshold
    ValueError
        The 'standard' function input parameter can only be 'ISO' or 'ASHRAE'
    """
    if units.lower() == 'ip':
        tdb, tr, vr = units_converter(tdb=tdb, tr=tr, v=vr)

    standard = standard.lower()
    if standard not in ['iso', 'ashrae']:
        raise ValueError("PMV calculations can only be performed in compliance with ISO or ASHRAE Standards")

    check_standard_compliance(standard=standard, tdb=tdb, tr=tr, v=vr, rh=rh, met=met, clo=clo)

    # if the relative air velocity is higher than 0.2 then follow methodology ASHRAE Appendix H, H3
    if standard == 'ashrae' and vr >= 0.2:
        # calculate the cooling effect
        ce = cooling_effect(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, wme=wme)

        tdb = tdb - ce
        tr = tr - ce
        vr = 0.1

    pa = rh * 10 * math.exp(16.6536 - 4030.183 / (tdb + 235))

    icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
    m = met * 58.15  # metabolic rate in W/M2
    w = wme * 58.15  # external work in W/M2
    mw = m - w  # internal heat production in the human body
    # calculation of the clothing area factor
    if icl <= 0.078:
        fcl = 1 + (1.29 * icl)
    else:
        fcl = 1.05 + (0.645 * icl)

    # heat transfer coefficient by forced convection
    hcf = 12.1 * math.sqrt(vr)
    taa = tdb + 273
    tra = tr + 273
    tcla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

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
        if hcf > hcn:
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
    hl4 = 0.0014 * m * (34 - tdb)
    # heat loss by radiation
    hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100.0, 4))
    # heat loss by convection
    hl6 = fcl * hc * (tcl - tdb)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pow(pmv, 4.0) - 0.2179 * pow(pmv, 2.0))

    return {'pmv': round(pmv, 2), 'ppd': round(ppd, 1)}


def pmv(tdb, tr, vr, rh, met, clo, wme=0, standard='ISO', units='SI'):
    """
    Returns Predicted Mean Vote (`PMV`_) calculated in accordance to main thermal comfort Standards. The PMV is an index that predicts the mean value of the thermal sensation votes
    (self-reported perceptions) of a large group of people on a sensation scale expressed from –3 to +3 corresponding to the categories \"cold,\" \"cool,\" \"slightly cool,\"
    \"neutral,\" \"slightly warm,\" \"warm,\" and \"hot.\" [1]_

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    vr : float
        relative air velocity, default in [m/s] in [fps] if `units` = 'IP'

        Note: vr is the relative air velocity caused by body movement and not the air speed measured by the air velocity sensor.
        It can be calculate using the function :py:meth:`pythermalcomfort.psychrometrics.v_relative`.
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]

        Note: The ASHRAE 55 Standard suggests that the dynamic clothing insulation is used as input in the PMV model.
        The dynamic clothing insulation can be calculated using the function :py:meth:`pythermalcomfort.psychrometrics.clo_dynamic`.
    wme : float
        external work, [met] default 0
    standard: str (default="ISO")
        comfort standard used for calculation

        - If "ISO", then the ISO Equation is used
        - If "ASHRAE", then the ASHRAE Equation is used

        Note: While the PMV equation is the same for both the ISO and ASHRAE standards,
        the ASHRAE Standard Use of the PMV model is limited to air speeds below 0.20 m/s (40 fpm).
        When air speeds exceed 0.20 m/s (40 fpm), the comfort zone boundaries are adjusted based on the SET model.
        See ASHRAE 55 2017 Appendix H for more information [1]_.
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    PMV : float
        Predicted Mean Vote

    Notes
    -----
    You can use this function to calculate the `PMV`_ [1]_ [2]_.

    .. _PMV: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import pmv
        >>> pmv(25, 25, 0.1, 50, 1.2, .5, wme=0)
        0.08
    """

    return pmv_ppd(tdb, tr, vr, rh, met, clo, wme, standard=standard, units=units)['pmv']


def set_tmp(tdb, tr, v, rh, met, clo, wme=0, body_surface_area=1.8258, patm=101325, units='SI'):
    """
    Calculates the Standard Effective Temperature (SET). The SET is the temperature of an imaginary environment at 50% (rh), <0.1 m/s (20 fpm) average air speed (v), and tr = tdb ,
    in which the total heat loss from the skin of an imaginary occupant with an activity level of 1.0 met and a clothing level of 0.6 clo is the same as that
    from a person in the actual environment with actual clothing and activity level.

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    v : float
        air velocity, default in [m/s] in [fps] if `units` = 'IP'
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]
    wme : float
        external work, [met] default 0
    body_surface_area : float
        body surface area, default value 1.8258 [m2] in [ft2] if `units` = 'IP'
    patm : float
        atmospheric pressure, default value 101325 [Pa] in [atm] if `units` = 'IP'
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    SET : float
        Standard effective temperature, [°C]

    Notes
    -----
    You can use this function to calculate the `SET`_ temperature in accordance with the ASHRAE 55 2017 Standard [1]_.

    .. _SET: https://en.wikipedia.org/wiki/Thermal_comfort#Standard_effective_temperature

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import set_tmp
        >>> set_tmp(tdb=25, tr=25, v=0.1, rh=50, met=1.2, clo=.5)
        25.3

        >>> # for users who wants to use the IP system
        >>> set_tmp(tdb=77, tr=77, v=0.328, rh=50, met=1.2, clo=.5, units='IP')
        77.6

    """
    if units.lower() == 'ip':
        if body_surface_area == 1.8258:
            body_surface_area = 19.65
        if patm == 101325:
            patm = 1
        tdb, tr, v, body_surface_area, patm = units_converter(tdb=tdb, tr=tr, v=v, area=body_surface_area, pressure=patm)

    check_standard_compliance(standard='ashrae', tdb=tdb, tr=tr, v=v, rh=rh, met=met, clo=clo)

    # Initial variables as defined in the ASHRAE 55-2017
    vapor_pressure = rh * p_sat_torr(tdb) / 100
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

    pressure_in_atmospheres = patm / 101325
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
    TOP = (CHR * tr + CHC * tdb) / CTC
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
                TOP = (CHR * tr + CHC * tdb) / CTC
            TCL = (RA * temp_skin + RCL * TOP) / (RA + RCL)
            flag = True
        flag = False
        DRY = (temp_skin - TOP) / (RA + RCL)
        HFCS = (temp_core - temp_skin) * (5.28 + 1.163 * skin_blood_flow)
        ERES = 0.0023 * M * (44.0 - vapor_pressure)
        CRES = 0.0014 * M * (34.0 - tdb)
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
        CHCS = 5.66 * math.pow((met - 0.85), 0.39)
    if CHCS < 3.0:
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

    if units.lower() == 'ip':
        set = units_converter(tmp=set, from_units='si')[0]

    return round(set, 1)


def adaptive_ashrae(tdb, tr, t_running_mean, v, units='SI'):
    """
    Determines the adaptive thermal comfort based on ASHRAE 55. The adaptive model relates indoor design temperatures or acceptable temperature ranges to outdoor meteorological
    or climatological parameters.

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    t_running_mean: float
        running mean temperature, default in [°C] in [°C] in [°F] if `units` = 'IP'
    v : float
        air velocity, default in [m/s] in [fps] if `units` = 'IP'
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    tmp_cmf : float
        Comfort temperature a that specific running mean temperature, default in [°C] or in [°F]
    tmp_cmf_80_low : float
        Lower acceptable comfort temperature for 80% occupants, default in [°C] or in [°F]
    tmp_cmf_80_up : float
        Upper acceptable comfort temperature for 80% occupants, default in [°C] or in [°F]
    tmp_cmf_90_low : float
        Lower acceptable comfort temperature for 90% occupants, default in [°C] or in [°F]
    tmp_cmf_90_up : float
        Upper acceptable comfort temperature for 90% occupants, default in [°C] or in [°F]
    acceptability_80 : bol
        Acceptability for 80% occupants
    acceptability_90 : bol
        Acceptability for 90% occupants

    Notes
    -----
    You can use this function to calculate if your conditions are within the `adaptive thermal comfort region`.
    Calculations with comply with the ASHRAE 55 2017 Standard [1]_.

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import adaptive_ashrae
        >>> results = adaptive_ashrae(tdb=25, tr=25, t_running_mean=20, v=0.1)
        >>> print(results)
        {'tmp_cmf': 24.0, 'tmp_cmf_80_low': 20.5, 'tmp_cmf_80_up': 27.5, 'tmp_cmf_90_low': 21.5, 'tmp_cmf_90_up': 26.5, 'acceptability_80': True, 'acceptability_90': False}

        >>> print(results['acceptability_80'])
        True
        # The conditions you entered are considered to be comfortable for by 80% of the occupants

        >>> # for users who wants to use the IP system
        >>> results = adaptive_ashrae(tdb=77, tr=77, t_running_mean=68, v=0.3, units='ip')
        >>> print(results)
        {'tmp_cmf': 75.2, 'tmp_cmf_80_low': 68.9, 'tmp_cmf_80_up': 81.5, 'tmp_cmf_90_low': 70.7, 'tmp_cmf_90_up': 79.7, 'acceptability_80': True, 'acceptability_90': False}

        >>> results = adaptive_ashrae(tdb=25, tr=25, t_running_mean=9, v=0.1)
        ValueError: The running mean is outside the standards applicability limits
        # The adaptive thermal comfort model can only be used
        # if the running mean temperature is higher than 10°C

    Raises
    ------
    ValueError
        Raised if the input are outside the Standard's applicability limits

    """
    if units.lower() == 'ip':
        tdb, tr, t_running_mean, vr = units_converter(tdb=tdb, tr=tr, tmp_running_mean=t_running_mean, v=v)

    check_standard_compliance(standard='ashrae', tdb=tdb, tr=tr, v=v)

    # Define the variables that will be used throughout the calculation.
    results = dict()

    to = t_o(tdb, tr, v)

    # See if the running mean temperature is between 10 °C and 33.5 °C (the range where the adaptive model is supposed to be used)
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
        tmp_cmf_90_low = t_cmf - 2.5
        tmp_cmf_80_up = t_cmf + 3.5 + cooling_effect
        tmp_cmf_90_up = t_cmf + 2.5 + cooling_effect

        def acceptability(t_cmf_lower, t_cmf_upper):
            # See if the conditions are comfortable.
            if t_cmf_lower < to < t_cmf_upper:
                return True
            else:
                return False

        acceptability_80 = acceptability(tmp_cmf_80_low, tmp_cmf_80_up)
        acceptability_90 = acceptability(tmp_cmf_90_low, tmp_cmf_90_up)

        if units.lower() == 'ip':
            t_cmf, tmp_cmf_80_low, tmp_cmf_80_up, tmp_cmf_90_low, tmp_cmf_90_up = units_converter(from_units='si', tmp_cmf=t_cmf, tmp_cmf_80_low=tmp_cmf_80_low,
                                                                                                  tmp_cmf_80_up=tmp_cmf_80_up, tmp_cmf_90_low=tmp_cmf_90_low,
                                                                                                  tmp_cmf_90_up=tmp_cmf_90_up)

        results = {'tmp_cmf': t_cmf, 'tmp_cmf_80_low': tmp_cmf_80_low, 'tmp_cmf_80_up': tmp_cmf_80_up,
                   'tmp_cmf_90_low': tmp_cmf_90_low, 'tmp_cmf_90_up': tmp_cmf_90_up,
                   'acceptability_80': acceptability_80, 'acceptability_90': acceptability_90, }

    else:
        raise ValueError("The running mean is outside the standards applicability limits")

    return results


def adaptive_en(tdb, tr, t_running_mean, v, units='SI'):
    """ Determines the adaptive thermal comfort based on EN 16798-1 2019 [3]_

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    t_running_mean: float
        running mean temperature, default in [°C] in [°C] in [°F] if `units` = 'IP'
    v : float
        air velocity, default in [m/s] in [fps] if `units` = 'IP'

        Note: Indoor operative temperature correction is applicable for buildings equipped
        with fans or personal systems providing building occupants with personal control over air speed at occupant level.
        For operative temperatures above 25°C the comfort zone upper limit can be increased by 1.2 °C (0.6 < v < 0.9 m/s), 1.8 °C (0.9 < v < 1.2 m/s), 2.2 °C ( v > 1.2 m/s)
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    tmp_cmf : float
        Comfort temperature at that specific running mean temperature, default in [°C] or in [°F]
    acceptability_cat_i : bol
        If the indoor conditions comply with comfort category I
    acceptability_cat_ii : bol
        If the indoor conditions comply with comfort category II
    acceptability_cat_iii : bol
        If the indoor conditions comply with comfort category III
    tmp_cmf_cat_i_up : float
        Upper acceptable comfort temperature for category I, default in [°C] or in [°F]
    tmp_cmf_cat_ii_up : float
        Upper acceptable comfort temperature for category II, default in [°C] or in [°F]
    tmp_cmf_cat_iii_up : float
        Upper acceptable comfort temperature for category III, default in [°C] or in [°F]
    tmp_cmf_cat_i_low : float
        Lower acceptable comfort temperature for category I, default in [°C] or in [°F]
    tmp_cmf_cat_ii_low : float
        Lower acceptable comfort temperature for category II, default in [°C] or in [°F]
    tmp_cmf_cat_iii_low : float
        Lower acceptable comfort temperature for category III, default in [°C] or in [°F]

    Notes
    -----
    You can use this function to calculate if your conditions are within the EN adaptive thermal comfort region.
    Calculations with comply with the EN 16798-1 2019 [1]_.

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import adaptive_en
        >>> results = adaptive_en(tdb=25, tr=25, t_running_mean=20, v=0.1)
        >>> print(results)
        {'tmp_cmf': 25.4, 'acceptability_cat_i': True, 'acceptability_cat_ii': True, 'acceptability_cat_iii': True, ... }

        >>> print(results['acceptability_cat_i'])
        True
        # The conditions you entered are considered to comply with Category I

        >>> # for users who wants to use the IP system
        >>> results = adaptive_en(tdb=77, tr=77, t_running_mean=68, v=0.3, units='ip')
        >>> print(results)
        {'tmp_cmf': 77.7, 'acceptability_cat_i': True, 'acceptability_cat_ii': True, 'acceptability_cat_iii': True, ... }

        >>> results = adaptive_en(tdb=25, tr=25, t_running_mean=9, v=0.1)
        ValueError: The running mean is outside the standards applicability limits
        # The adaptive thermal comfort model can only be used
        # if the running mean temperature is between 10 °C and 30 °C

    Raises
    ------
    ValueError
        Raised if the input are outside the Standard's applicability limits

    """

    if units.lower() == 'ip':
        tdb, tr, t_running_mean, vr = units_converter(tdb=tdb, tr=tr, tmp_running_mean=t_running_mean, v=v)

    if (t_running_mean < 10) or (t_running_mean > 30):
        raise ValueError("The running mean is outside the standards applicability limits")

    to = t_o(tdb, tr, v)

    cooling_effect = 0
    # calculate cooling effect of elevated air speed when top > 25 degC.
    if v >= 0.6 and to >= 25:
        if v < 0.9:
            cooling_effect = 1.2
        elif v < 1.2:
            cooling_effect = 1.8
        else:
            cooling_effect = 2.2

    t_cmf = 0.33 * t_running_mean + 18.8

    t_cmf_i_lower = t_cmf - 3
    t_cmf_ii_lower = t_cmf - 4
    t_cmf_iii_lower = t_cmf - 5
    t_cmf_i_upper = t_cmf + 2 + cooling_effect
    t_cmf_ii_upper = t_cmf + 3 + cooling_effect
    t_cmf_iii_upper = t_cmf + 4 + cooling_effect

    def between(val, low, high):
        return low < val < high

    if between(to, t_cmf_i_lower, t_cmf_i_upper):
        acceptability_i, acceptability_ii, acceptability_iii = True, True, True
    elif between(to, t_cmf_ii_lower, t_cmf_ii_upper):
        acceptability_ii, acceptability_iii = True, True
        acceptability_i = False
    elif between(to, t_cmf_iii_lower, t_cmf_iii_upper):
        acceptability_iii = True
        acceptability_i, acceptability_ii = False, False
    else:
        acceptability_i, acceptability_ii, acceptability_iii = False, False, False

    if units.lower() == 'ip':
        t_cmf, t_cmf_i_upper, t_cmf_ii_upper, t_cmf_iii_upper = units_converter(from_units='si', tmp_cmf=t_cmf, tmp_cmf_cat_i_up=t_cmf_i_upper, tmp_cmf_cat_ii_up=t_cmf_ii_upper,
                                                                                tmp_cmf_cat_iii_up=t_cmf_iii_upper)
        t_cmf_i_lower, t_cmf_ii_lower, t_cmf_iii_lower = units_converter(from_units='si', tmp_cmf_cat_i_low=t_cmf_i_lower, tmp_cmf_cat_ii_low=t_cmf_ii_lower,
                                                                         tmp_cmf_cat_iii_low=t_cmf_iii_lower)

    results = {'tmp_cmf': round(t_cmf, 1), 'acceptability_cat_i': acceptability_i, 'acceptability_cat_ii': acceptability_ii, 'acceptability_cat_iii': acceptability_iii,
               'tmp_cmf_cat_i_up': round(t_cmf_i_upper, 1), 'tmp_cmf_cat_ii_up': round(t_cmf_ii_upper, 1), 'tmp_cmf_cat_iii_up': round(t_cmf_iii_upper, 1),
               'tmp_cmf_cat_i_low': round(t_cmf_i_lower, 1), 'tmp_cmf_cat_ii_low': round(t_cmf_ii_lower, 1), 'tmp_cmf_cat_iii_low': round(t_cmf_iii_lower, 1)}

    return results


def utci(tdb, tr, v, rh, units='SI'):
    """ Determines the Universal Thermal Climate Index (UTCI). The UTCI is the equivalent temperature for the environment derived from a reference environment.
    It is defined as the air temperature of the reference environment which produces the same strain index value in comparison with the reference individual's response to the real
    environment. It is regarded as one of the most comprehensive indices for calculating heat stress in outdoor spaces. The parameters that are taken into account for calculating
    UTCI involve dry-bulb temperature, mean radiation temperature, the pressure of water vapor or relative humidity, and wind speed (at the elevation of 10 m) [7]_.

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    v : float
        relative air velocity, default in [m/s] in [fps] if `units` = 'IP'
    rh : float
        relative humidity, [%]
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    utci : float
         Universal Thermal Climate Index, [°C] or in [°F]

    Notes
    -----
    You can use this function to calculate the Universal Thermal Climate Index (`UTCI`)
    The applicability wind speed value must be between 0.5 and 17 m/s.

    .. _UTCI: http://www.utci.org/utcineu/utcineu.php

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import utci
        >>> utci(tdb=25, tr=25, v=1.0, rh=50)
        24.6

        >>> # for users who wants to use the IP system
        >>> utci(tdb=77, tr=77, v=3.28, rh=50, units='ip')
        76.4

    Raises
    ------
    ValueError
        Raised if the input are outside the Standard's applicability limits

    """

    if units.lower() == 'ip':
        tdb, tr, v = units_converter(tdb=tdb, tr=tr, v=v)

    check_standard_compliance(standard='utci', tdb=tdb, tr=tr, v=v)

    def es(tdb):
        g = [
            -2836.5744, -6028.076559, 19.54263612,
            -0.02737830188, 0.000016261698,
            (7.0229056 * (10 ** (-10))), (-1.8680009 * (10 ** (-13)))]
        tk = tdb + 273.15  # air temp in K
        es = 2.7150305 * math.log1p(tk)
        for count, i in enumerate(g):
            es = es + (i * (tk ** (count - 2)))
        es = math.exp(es) * 0.01  # convert Pa to hPa
        return es

    # Do a series of checks to be sure that the input values are within the bounds accepted by the model.
    if (tdb < -50.0) or (tdb > 50.0) or (tr - tdb < -30.0) or (tr - tdb > 70.0) or (v < 0.5) or (v > 17):
        raise ValueError("The value you entered are outside the equation applicability limits")

    # This is a python version of the UTCI_approx function
    # Version a 0.002, October 2009
    # tdb: air temperature, degrees Celsius
    # ehPa: water vapour presure, hPa=hecto Pascal
    # Tmrt: mean radiant temperature, degrees Celsius
    # va10m: wind speed 10m above ground level in m/s

    ehPa = es(tdb) * (rh / 100.0)
    delta_t_tr = tr - tdb
    Pa = ehPa / 10.0  # convert vapour pressure to kPa

    utci_approx = (tdb +
                   (0.607562052) +
                   (-0.0227712343) * tdb +
                   (8.06470249 * (10 ** (-4))) * tdb * tdb +
                   (-1.54271372 * (10 ** (-4))) * tdb * tdb * tdb +
                   (-3.24651735 * (10 ** (-6))) * tdb * tdb * tdb * tdb +
                   (7.32602852 * (10 ** (-8))) * tdb * tdb * tdb * tdb * tdb +
                   (1.35959073 * (10 ** (-9))) * tdb * tdb * tdb * tdb * tdb * tdb +
                   (-2.25836520) * v +
                   (0.0880326035) * tdb * v +
                   (0.00216844454) * tdb * tdb * v +
                   (-1.53347087 * (10 ** (-5))) * tdb * tdb * tdb * v +
                   (-5.72983704 * (10 ** (-7))) * tdb * tdb * tdb * tdb * v +
                   (-2.55090145 * (10 ** (-9))) * tdb * tdb * tdb * tdb * tdb * v +
                   (-0.751269505) * v * v +
                   (-0.00408350271) * tdb * v * v +
                   (-5.21670675 * (10 ** (-5))) * tdb * tdb * v * v +
                   (1.94544667 * (10 ** (-6))) * tdb * tdb * tdb * v * v +
                   (1.14099531 * (10 ** (-8))) * tdb * tdb * tdb * tdb * v * v +
                   (0.158137256) * v * v * v +
                   (-6.57263143 * (10 ** (-5))) * tdb * v * v * v +
                   (2.22697524 * (10 ** (-7))) * tdb * tdb * v * v * v +
                   (-4.16117031 * (10 ** (-8))) * tdb * tdb * tdb * v * v * v +
                   (-0.0127762753) * v * v * v * v +
                   (9.66891875 * (10 ** (-6))) * tdb * v * v * v * v +
                   (2.52785852 * (10 ** (-9))) * tdb * tdb * v * v * v * v +
                   (4.56306672 * (10 ** (-4))) * v * v * v * v * v +
                   (-1.74202546 * (10 ** (-7))) * tdb * v * v * v * v * v +
                   (-5.91491269 * (10 ** (-6))) * v * v * v * v * v * v +
                   (0.398374029) * delta_t_tr +
                   (1.83945314 * (10 ** (-4))) * tdb * delta_t_tr +
                   (-1.73754510 * (10 ** (-4))) * tdb * tdb * delta_t_tr +
                   (-7.60781159 * (10 ** (-7))) * tdb * tdb * tdb * delta_t_tr +
                   (3.77830287 * (10 ** (-8))) * tdb * tdb * tdb * tdb * delta_t_tr +
                   (5.43079673 * (10 ** (-10))) * tdb * tdb * tdb * tdb * tdb * delta_t_tr +
                   (-0.0200518269) * v * delta_t_tr +
                   (8.92859837 * (10 ** (-4))) * tdb * v * delta_t_tr +
                   (3.45433048 * (10 ** (-6))) * tdb * tdb * v * delta_t_tr +
                   (-3.77925774 * (10 ** (-7))) * tdb * tdb * tdb * v * delta_t_tr +
                   (-1.69699377 * (10 ** (-9))) * tdb * tdb * tdb * tdb * v * delta_t_tr +
                   (1.69992415 * (10 ** (-4))) * v * v * delta_t_tr +
                   (-4.99204314 * (10 ** (-5))) * tdb * v * v * delta_t_tr +
                   (2.47417178 * (10 ** (-7))) * tdb * tdb * v * v * delta_t_tr +
                   (1.07596466 * (10 ** (-8))) * tdb * tdb * tdb * v * v * delta_t_tr +
                   (8.49242932 * (10 ** (-5))) * v * v * v * delta_t_tr +
                   (1.35191328 * (10 ** (-6))) * tdb * v * v * v * delta_t_tr +
                   (-6.21531254 * (10 ** (-9))) * tdb * tdb * v * v * v * delta_t_tr +
                   (-4.99410301 * (10 ** (-6))) * v * v * v * v * delta_t_tr +
                   (-1.89489258 * (10 ** (-8))) * tdb * v * v * v * v * delta_t_tr +
                   (8.15300114 * (10 ** (-8))) * v * v * v * v * v * delta_t_tr +
                   (7.55043090 * (10 ** (-4))) * delta_t_tr * delta_t_tr +
                   (-5.65095215 * (10 ** (-5))) * tdb * delta_t_tr * delta_t_tr +
                   (-4.52166564 * (10 ** (-7))) * tdb * tdb * delta_t_tr * delta_t_tr +
                   (2.46688878 * (10 ** (-8))) * tdb * tdb * tdb * delta_t_tr * delta_t_tr +
                   (2.42674348 * (10 ** (-10))) * tdb * tdb * tdb * tdb * delta_t_tr * delta_t_tr +
                   (1.54547250 * (10 ** (-4))) * v * delta_t_tr * delta_t_tr +
                   (5.24110970 * (10 ** (-6))) * tdb * v * delta_t_tr * delta_t_tr +
                   (-8.75874982 * (10 ** (-8))) * tdb * tdb * v * delta_t_tr * delta_t_tr +
                   (-1.50743064 * (10 ** (-9))) * tdb * tdb * tdb * v * delta_t_tr * delta_t_tr +
                   (-1.56236307 * (10 ** (-5))) * v * v * delta_t_tr * delta_t_tr +
                   (-1.33895614 * (10 ** (-7))) * tdb * v * v * delta_t_tr * delta_t_tr +
                   (2.49709824 * (10 ** (-9))) * tdb * tdb * v * v * delta_t_tr * delta_t_tr +
                   (6.51711721 * (10 ** (-7))) * v * v * v * delta_t_tr * delta_t_tr +
                   (1.94960053 * (10 ** (-9))) * tdb * v * v * v * delta_t_tr * delta_t_tr +
                   (-1.00361113 * (10 ** (-8))) * v * v * v * v * delta_t_tr * delta_t_tr +
                   (-1.21206673 * (10 ** (-5))) * delta_t_tr * delta_t_tr * delta_t_tr +
                   (-2.18203660 * (10 ** (-7))) * tdb * delta_t_tr * delta_t_tr * delta_t_tr +
                   (7.51269482 * (10 ** (-9))) * tdb * tdb * delta_t_tr * delta_t_tr * delta_t_tr +
                   (9.79063848 * (10 ** (-11))) * tdb * tdb * tdb * delta_t_tr * delta_t_tr * delta_t_tr +
                   (1.25006734 * (10 ** (-6))) * v * delta_t_tr * delta_t_tr * delta_t_tr +
                   (-1.81584736 * (10 ** (-9))) * tdb * v * delta_t_tr * delta_t_tr * delta_t_tr +
                   (-3.52197671 * (10 ** (-10))) * tdb * tdb * v * delta_t_tr * delta_t_tr * delta_t_tr +
                   (-3.36514630 * (10 ** (-8))) * v * v * delta_t_tr * delta_t_tr * delta_t_tr +
                   (1.35908359 * (10 ** (-10))) * tdb * v * v * delta_t_tr * delta_t_tr * delta_t_tr +
                   (4.17032620 * (10 ** (-10))) * v * v * v * delta_t_tr * delta_t_tr * delta_t_tr +
                   (-1.30369025 * (10 ** (-9))) * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr +
                   (4.13908461 * (10 ** (-10))) * tdb * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr +
                   (9.22652254 * (10 ** (-12))) * tdb * tdb * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr +
                   (-5.08220384 * (10 ** (-9))) * v * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr +
                   (-2.24730961 * (10 ** (-11))) * tdb * v * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr +
                   (1.17139133 * (10 ** (-10))) * v * v * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr +
                   (6.62154879 * (10 ** (-10))) * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr +
                   (4.03863260 * (10 ** (-13))) * tdb * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr +
                   (1.95087203 * (10 ** (-12))) * v * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr +
                   (-4.73602469 * (10 ** (-12))) * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr +
                   (5.12733497) * Pa +
                   (-0.312788561) * tdb * Pa +
                   (-0.0196701861) * tdb * tdb * Pa +
                   (9.99690870 * (10 ** (-4))) * tdb * tdb * tdb * Pa +
                   (9.51738512 * (10 ** (-6))) * tdb * tdb * tdb * tdb * Pa +
                   (-4.66426341 * (10 ** (-7))) * tdb * tdb * tdb * tdb * tdb * Pa +
                   (0.548050612) * v * Pa +
                   (-0.00330552823) * tdb * v * Pa +
                   (-0.00164119440) * tdb * tdb * v * Pa +
                   (-5.16670694 * (10 ** (-6))) * tdb * tdb * tdb * v * Pa +
                   (9.52692432 * (10 ** (-7))) * tdb * tdb * tdb * tdb * v * Pa +
                   (-0.0429223622) * v * v * Pa +
                   (0.00500845667) * tdb * v * v * Pa +
                   (1.00601257 * (10 ** (-6))) * tdb * tdb * v * v * Pa +
                   (-1.81748644 * (10 ** (-6))) * tdb * tdb * tdb * v * v * Pa +
                   (-1.25813502 * (10 ** (-3))) * v * v * v * Pa +
                   (-1.79330391 * (10 ** (-4))) * tdb * v * v * v * Pa +
                   (2.34994441 * (10 ** (-6))) * tdb * tdb * v * v * v * Pa +
                   (1.29735808 * (10 ** (-4))) * v * v * v * v * Pa +
                   (1.29064870 * (10 ** (-6))) * tdb * v * v * v * v * Pa +
                   (-2.28558686 * (10 ** (-6))) * v * v * v * v * v * Pa +
                   (-0.0369476348) * delta_t_tr * Pa +
                   (0.00162325322) * tdb * delta_t_tr * Pa +
                   (-3.14279680 * (10 ** (-5))) * tdb * tdb * delta_t_tr * Pa +
                   (2.59835559 * (10 ** (-6))) * tdb * tdb * tdb * delta_t_tr * Pa +
                   (-4.77136523 * (10 ** (-8))) * tdb * tdb * tdb * tdb * delta_t_tr * Pa +
                   (8.64203390 * (10 ** (-3))) * v * delta_t_tr * Pa +
                   (-6.87405181 * (10 ** (-4))) * tdb * v * delta_t_tr * Pa +
                   (-9.13863872 * (10 ** (-6))) * tdb * tdb * v * delta_t_tr * Pa +
                   (5.15916806 * (10 ** (-7))) * tdb * tdb * tdb * v * delta_t_tr * Pa +
                   (-3.59217476 * (10 ** (-5))) * v * v * delta_t_tr * Pa +
                   (3.28696511 * (10 ** (-5))) * tdb * v * v * delta_t_tr * Pa +
                   (-7.10542454 * (10 ** (-7))) * tdb * tdb * v * v * delta_t_tr * Pa +
                   (-1.24382300 * (10 ** (-5))) * v * v * v * delta_t_tr * Pa +
                   (-7.38584400 * (10 ** (-9))) * tdb * v * v * v * delta_t_tr * Pa +
                   (2.20609296 * (10 ** (-7))) * v * v * v * v * delta_t_tr * Pa +
                   (-7.32469180 * (10 ** (-4))) * delta_t_tr * delta_t_tr * Pa +
                   (-1.87381964 * (10 ** (-5))) * tdb * delta_t_tr * delta_t_tr * Pa +
                   (4.80925239 * (10 ** (-6))) * tdb * tdb * delta_t_tr * delta_t_tr * Pa +
                   (-8.75492040 * (10 ** (-8))) * tdb * tdb * tdb * delta_t_tr * delta_t_tr * Pa +
                   (2.77862930 * (10 ** (-5))) * v * delta_t_tr * delta_t_tr * Pa +
                   (-5.06004592 * (10 ** (-6))) * tdb * v * delta_t_tr * delta_t_tr * Pa +
                   (1.14325367 * (10 ** (-7))) * tdb * tdb * v * delta_t_tr * delta_t_tr * Pa +
                   (2.53016723 * (10 ** (-6))) * v * v * delta_t_tr * delta_t_tr * Pa +
                   (-1.72857035 * (10 ** (-8))) * tdb * v * v * delta_t_tr * delta_t_tr * Pa +
                   (-3.95079398 * (10 ** (-8))) * v * v * v * delta_t_tr * delta_t_tr * Pa +
                   (-3.59413173 * (10 ** (-7))) * delta_t_tr * delta_t_tr * delta_t_tr * Pa +
                   (7.04388046 * (10 ** (-7))) * tdb * delta_t_tr * delta_t_tr * delta_t_tr * Pa +
                   (-1.89309167 * (10 ** (-8))) * tdb * tdb * delta_t_tr * delta_t_tr * delta_t_tr * Pa +
                   (-4.79768731 * (10 ** (-7))) * v * delta_t_tr * delta_t_tr * delta_t_tr * Pa +
                   (7.96079978 * (10 ** (-9))) * tdb * v * delta_t_tr * delta_t_tr * delta_t_tr * Pa +
                   (1.62897058 * (10 ** (-9))) * v * v * delta_t_tr * delta_t_tr * delta_t_tr * Pa +
                   (3.94367674 * (10 ** (-8))) * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * Pa +
                   (-1.18566247 * (10 ** (-9))) * tdb * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * Pa +
                   (3.34678041 * (10 ** (-10))) * v * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * Pa +
                   (-1.15606447 * (10 ** (-10))) * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * Pa +
                   (-2.80626406) * Pa * Pa +
                   (0.548712484) * tdb * Pa * Pa +
                   (-0.00399428410) * tdb * tdb * Pa * Pa +
                   (-9.54009191 * (10 ** (-4))) * tdb * tdb * tdb * Pa * Pa +
                   (1.93090978 * (10 ** (-5))) * tdb * tdb * tdb * tdb * Pa * Pa +
                   (-0.308806365) * v * Pa * Pa +
                   (0.0116952364) * tdb * v * Pa * Pa +
                   (4.95271903 * (10 ** (-4))) * tdb * tdb * v * Pa * Pa +
                   (-1.90710882 * (10 ** (-5))) * tdb * tdb * tdb * v * Pa * Pa +
                   (0.00210787756) * v * v * Pa * Pa +
                   (-6.98445738 * (10 ** (-4))) * tdb * v * v * Pa * Pa +
                   (2.30109073 * (10 ** (-5))) * tdb * tdb * v * v * Pa * Pa +
                   (4.17856590 * (10 ** (-4))) * v * v * v * Pa * Pa +
                   (-1.27043871 * (10 ** (-5))) * tdb * v * v * v * Pa * Pa +
                   (-3.04620472 * (10 ** (-6))) * v * v * v * v * Pa * Pa +
                   (0.0514507424) * delta_t_tr * Pa * Pa +
                   (-0.00432510997) * tdb * delta_t_tr * Pa * Pa +
                   (8.99281156 * (10 ** (-5))) * tdb * tdb * delta_t_tr * Pa * Pa +
                   (-7.14663943 * (10 ** (-7))) * tdb * tdb * tdb * delta_t_tr * Pa * Pa +
                   (-2.66016305 * (10 ** (-4))) * v * delta_t_tr * Pa * Pa +
                   (2.63789586 * (10 ** (-4))) * tdb * v * delta_t_tr * Pa * Pa +
                   (-7.01199003 * (10 ** (-6))) * tdb * tdb * v * delta_t_tr * Pa * Pa +
                   (-1.06823306 * (10 ** (-4))) * v * v * delta_t_tr * Pa * Pa +
                   (3.61341136 * (10 ** (-6))) * tdb * v * v * delta_t_tr * Pa * Pa +
                   (2.29748967 * (10 ** (-7))) * v * v * v * delta_t_tr * Pa * Pa +
                   (3.04788893 * (10 ** (-4))) * delta_t_tr * delta_t_tr * Pa * Pa +
                   (-6.42070836 * (10 ** (-5))) * tdb * delta_t_tr * delta_t_tr * Pa * Pa +
                   (1.16257971 * (10 ** (-6))) * tdb * tdb * delta_t_tr * delta_t_tr * Pa * Pa +
                   (7.68023384 * (10 ** (-6))) * v * delta_t_tr * delta_t_tr * Pa * Pa +
                   (-5.47446896 * (10 ** (-7))) * tdb * v * delta_t_tr * delta_t_tr * Pa * Pa +
                   (-3.59937910 * (10 ** (-8))) * v * v * delta_t_tr * delta_t_tr * Pa * Pa +
                   (-4.36497725 * (10 ** (-6))) * delta_t_tr * delta_t_tr * delta_t_tr * Pa * Pa +
                   (1.68737969 * (10 ** (-7))) * tdb * delta_t_tr * delta_t_tr * delta_t_tr * Pa * Pa +
                   (2.67489271 * (10 ** (-8))) * v * delta_t_tr * delta_t_tr * delta_t_tr * Pa * Pa +
                   (3.23926897 * (10 ** (-9))) * delta_t_tr * delta_t_tr * delta_t_tr * delta_t_tr * Pa * Pa +
                   (-0.0353874123) * Pa * Pa * Pa +
                   (-0.221201190) * tdb * Pa * Pa * Pa +
                   (0.0155126038) * tdb * tdb * Pa * Pa * Pa +
                   (-2.63917279 * (10 ** (-4))) * tdb * tdb * tdb * Pa * Pa * Pa +
                   (0.0453433455) * v * Pa * Pa * Pa +
                   (-0.00432943862) * tdb * v * Pa * Pa * Pa +
                   (1.45389826 * (10 ** (-4))) * tdb * tdb * v * Pa * Pa * Pa +
                   (2.17508610 * (10 ** (-4))) * v * v * Pa * Pa * Pa +
                   (-6.66724702 * (10 ** (-5))) * tdb * v * v * Pa * Pa * Pa +
                   (3.33217140 * (10 ** (-5))) * v * v * v * Pa * Pa * Pa +
                   (-0.00226921615) * delta_t_tr * Pa * Pa * Pa +
                   (3.80261982 * (10 ** (-4))) * tdb * delta_t_tr * Pa * Pa * Pa +
                   (-5.45314314 * (10 ** (-9))) * tdb * tdb * delta_t_tr * Pa * Pa * Pa +
                   (-7.96355448 * (10 ** (-4))) * v * delta_t_tr * Pa * Pa * Pa +
                   (2.53458034 * (10 ** (-5))) * tdb * v * delta_t_tr * Pa * Pa * Pa +
                   (-6.31223658 * (10 ** (-6))) * v * v * delta_t_tr * Pa * Pa * Pa +
                   (3.02122035 * (10 ** (-4))) * delta_t_tr * delta_t_tr * Pa * Pa * Pa +
                   (-4.77403547 * (10 ** (-6))) * tdb * delta_t_tr * delta_t_tr * Pa * Pa * Pa +
                   (1.73825715 * (10 ** (-6))) * v * delta_t_tr * delta_t_tr * Pa * Pa * Pa +
                   (-4.09087898 * (10 ** (-7))) * delta_t_tr * delta_t_tr * delta_t_tr * Pa * Pa * Pa +
                   (0.614155345) * Pa * Pa * Pa * Pa +
                   (-0.0616755931) * tdb * Pa * Pa * Pa * Pa +
                   (0.00133374846) * tdb * tdb * Pa * Pa * Pa * Pa +
                   (0.00355375387) * v * Pa * Pa * Pa * Pa +
                   (-5.13027851 * (10 ** (-4))) * tdb * v * Pa * Pa * Pa * Pa +
                   (1.02449757 * (10 ** (-4))) * v * v * Pa * Pa * Pa * Pa +
                   (-0.00148526421) * delta_t_tr * Pa * Pa * Pa * Pa +
                   (-4.11469183 * (10 ** (-5))) * tdb * delta_t_tr * Pa * Pa * Pa * Pa +
                   (-6.80434415 * (10 ** (-6))) * v * delta_t_tr * Pa * Pa * Pa * Pa +
                   (-9.77675906 * (10 ** (-6))) * delta_t_tr * delta_t_tr * Pa * Pa * Pa * Pa +
                   (0.0882773108) * Pa * Pa * Pa * Pa * Pa +
                   (-0.00301859306) * tdb * Pa * Pa * Pa * Pa * Pa +
                   (0.00104452989) * v * Pa * Pa * Pa * Pa * Pa +
                   (2.47090539 * (10 ** (-4))) * delta_t_tr * Pa * Pa * Pa * Pa * Pa +
                   (0.00148348065) * Pa * Pa * Pa * Pa * Pa * Pa)

    # cmf = utci_approx > 9 and utci_approx < 26
    #
    # if utci_approx < -14.0:
    #     stress_range = -2
    # elif utci_approx < 9.0:
    #     stress_range = -1
    # elif utci_approx < 26.0:
    #     stress_range = 0
    # elif utci_approx < 32.0:
    #     stress_range = 1
    # else:
    #     stress_range = 2

    # return {'utci': round(UTCI_approx, 1), 'cmf': cmf, 'stress_range': stress_range}
    if units.lower() == 'ip':
        utci_approx = units_converter(tmp=utci_approx, from_units='si')[0]

    return round(utci_approx, 1)


def clo_tout(tout, units='SI'):
    """ Representative clothing insulation Icl as a function of outdoor air temperature at 06:00 a.m [4]_.

    Parameters
    ----------
    tout : float
        outdoor air temperature at 06:00 a.m., default in [°C] in [°F] if `units` = 'IP'
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    clo : float
         Representative clothing insulation Icl, [clo]

    Notes
    -----
    The ASHRAE 55 2017 states that it is acceptable to determine the clothing insulation Icl using this equation in mechanically conditioned buildings [1]_.

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import clo_tout
        >>> clo_tout(tout=27)
        0.46

    """
    if units.lower() == 'ip':
        tout = units_converter(tmp=tout)[0]

    if tout < -5:
        clo = 1
    elif tout < 5:
        clo = 0.818 - 0.0364 * tout
    elif tout < 26:
        clo = math.pow(10, -0.1635 - 0.0066 * tout)
    else:
        clo = 0.46

    return round(clo, 2)


def vertical_tmp_grad_ppd(tdb, tr, vr, rh, met, clo, vertical_tmp_grad, units='SI'):
    """ Calculates the percentage of thermally dissatisfied people with a vertical temperature gradient between feet and head [1]_.
    This equation is only applicable for vr < 0.2 m/s (40 fps).

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'

        Note: The air temperature is the average value over two heights: 0.6 m (24 in.) and 1.1 m (43 in.) for seated occupants
        and 1.1 m (43 in.) and 1.7 m (67 in.) for standing occupants.
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    vr : float
        relative air velocity, default in [m/s] in [fps] if `units` = 'IP'

        Note: vr is the relative air velocity caused by body movement and not the air speed measured by the air velocity sensor.
        It can be calculate using the function :py:meth:`pythermalcomfort.psychrometrics.v_relative`.
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]
    vertical_tmp_grad : float
        vertical temperature gradient between the feet and the head, default in [°C/m] in [°F/ft] if `units` = 'IP'
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    PPD_vg: float
        Predicted Percentage of Dissatisfied occupants with vertical temperature gradient, [%]
    Acceptability: bol
        The ASHRAE 55 2017 standard defines that the value of air speed at the ankle level is acceptable if PPD_ad is lower or equal than 5 %

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import vertical_tmp_grad_ppd
        >>> results = vertical_tmp_grad_ppd(25, 25, 0.1, 50, 1.2, 0.5, 7)
        >>> print(results)
        {'PPD_vg': 12.6, 'Acceptability': False}

    """
    if units.lower() == 'ip':
        tdb, tr, vr = units_converter(tdb=tdb, tr=tr, v=vr)
        vertical_tmp_grad = vertical_tmp_grad / 1.8 * 3.28

    check_standard_compliance(standard='ashrae', tdb=tdb, tr=tr, v_limited=vr, rh=rh, met=met, clo=clo)

    tsv = pmv(tdb, tr, vr, rh, met, clo, standard="ashrae")
    numerator = math.exp(0.13 * math.pow(tsv - 1.91, 2) + 0.15 * vertical_tmp_grad - 1.6)
    ppd_val = round((numerator / (1 + numerator) - 0.345) * 100, 1)
    acceptability = ppd_val <= 5
    return {"PPD_vg": ppd_val, "Acceptability": acceptability}


def ankle_draft(tdb, tr, vr, rh, met, clo, v_ankle, units='SI'):
    """
    Calculates the percentage of thermally dissatisfied people with the ankle draft (0.1 m) above floor level [1]_.
    This equation is only applicable for vr < 0.2 m/s (40 fps).

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'

        Note: The air temperature is the average value over two heights: 0.6 m (24 in.) and 1.1 m (43 in.) for seated occupants
        and 1.1 m (43 in.) and 1.7 m (67 in.) for standing occupants.
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    vr : float
        relative air velocity, default in [m/s] in [fps] if `units` = 'IP'

        Note: vr is the relative air velocity caused by body movement and not the air speed measured by the air velocity sensor.
        It can be calculate using the function :py:meth:`pythermalcomfort.psychrometrics.v_relative`.
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]
    v_ankle : float
        air speed at the 0.1 m (4 in.) above the floor, default in [m/s] in [fps] if `units` = 'IP'
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    PPD_ad: float
        Predicted Percentage of Dissatisfied occupants with ankle draft, [%]
    Acceptability: bol
        The ASHRAE 55 2017 standard defines that the value of air speed at the ankle level is acceptable if PPD_ad is lower or equal than 20 %

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import ankle_draft
        >>> results = ankle_draft(25, 25, 0.2, 50, 1.2, 0.5, 0.3, units="SI")
        >>> print(results)
        {'PPD_ad': 18.6, 'Acceptability': True}

    """
    if units.lower() == 'ip':
        tdb, tr, vr, v_ankle = units_converter(tdb=tdb, tr=tr, v=vr, vel=v_ankle)

    check_standard_compliance(standard='ashrae', tdb=tdb, tr=tr, v_limited=vr, rh=rh, met=met, clo=clo)

    tsv = pmv(tdb, tr, vr, rh, met, clo, standard="ashrae")
    ppd_val = round(math.exp(-2.58 + 3.05 * v_ankle - 1.06 * tsv) / (1 + math.exp(-2.58 + 3.05 * v_ankle - 1.06 * tsv)) * 100, 1)
    acceptability = ppd_val <= 20
    return {"PPD_ad": ppd_val, "Acceptability": acceptability}


def solar_gain(sol_altitude, sol_azimuth, sol_radiation_dir, sol_transmittance, f_svv, f_bes, asw=0.7, posture='seated', floor_reflectance=0.6):
    """
        Calculates the solar gain to the human body using the Effective Radiant Field (ERF) [1]_. The ERF is a measure of the net energy flux to or from the human body.
        ERF is expressed in W over human body surface area [w/m2]. In addition, it calculates the delta mean radiant temperature. Which is the amount by which the mean radiant
        temperature of the space should be increased if no solar radiation is present.

        Parameters
        ----------
        sol_altitude : float
            Solar altitude, degrees from horizontal [deg]. Ranges between 0 and 90.
        sol_azimuth : float
            Solar azimuth, degrees clockwise from North [deg]. Ranges between 0 and 180.
        posture : str
            Default 'seated' list of available options 'standing', 'supine' or 'seated'
        sol_radiation_dir : float
            Direct-beam solar radiation, [W/m2]. Ranges between 200 and 1000. See Table C2-3 of ASHRAE 55 2017 [1]_.
        sol_transmittance : float
            Total solar transmittance, ranges from 0 to 1. The total solar transmittance of window systems, including glazing unit, blinds, and other façade treatments, shall be determined using one of the following methods:
            i) Provided by manufacturer or from the National Fenestration Rating Council approved Lawrence Berkeley National Lab International Glazing Database.
            ii) Glazing unit plus venetian blinds or other complex or unique shades shall be calculated using National Fenestration Rating Council approved software or Lawrence Berkeley National Lab Complex Glazing Database.
        f_svv : float
            Fraction of sky vault exposed to body, ranges from 0 to 1.
        f_bes : float
            Fraction of the possible body surface exposed to sun, ranges from 0 to 1. See Table C2-2 and equation C-7 ASHRAE 55 2017 [1]_.
        asw: float
            The average short-wave absorptivity of the occupant. It will range widely, depending on the color of the occupant’s skin as well as the color and amount of clothing covering the body.
            A value of 0.7 shall be used unless more specific information about the clothing or skin color of the occupants is available.
            Note: Short-wave absorptivity typically ranges from 0.57 to 0.84, depending on skin and clothing color. More information is available in Blum (1945).
        floor_reflectance: float
            Floor refectance. It is assumed to be constant and equal to 0.6.

        Notes
        -----
        More information on the calculation procedure can be found in Appendix C of [1]_.

        Returns
        -------
        erf: float
            Solar gain to the human body using the Effective Radiant Field [W/m2]
        delta_mrt: float
            Delta mean radiant temperature. The amount by which the mean radiant temperature of the space should be increased if no solar radiation is present.

        Examples
        --------
        .. code-block:: python

            >>> from pythermalcomfort.models import solar_gain
            >>> results = solar_gain(sol_altitude=0, sol_azimuth=120, sol_radiation_dir=800, sol_transmittance=0.5, f_svv=0.5, f_bes=0.5, asw=0.7, posture='seated')
            >>> print(results)
            {'erf': 42.9, 'delta_mrt': 10.3}

        """

    posture = posture.lower()
    if posture not in ['standing', 'supine', 'seated']:
        raise ValueError("Posture has to be either standing, supine or seated")

    def find_span(arr, x):
        for i in range(0, len(arr)):
            if arr[i + 1] >= x >= arr[i]:
                return i
        return -1

    deg_to_rad = 0.0174532925
    hr = 6
    i_diff = 0.2 * sol_radiation_dir

    fp_table = [[0.25, 0.25, 0.23, 0.19, 0.15, 0.10, 0.06],
                [0.25, 0.25, 0.23, 0.18, 0.15, 0.10, 0.06],
                [0.24, 0.24, 0.22, 0.18, 0.14, 0.10, 0.06],
                [0.22, 0.22, 0.20, 0.17, 0.13, 0.09, 0.06],
                [0.21, 0.21, 0.18, 0.15, 0.12, 0.08, 0.06],
                [0.18, 0.18, 0.17, 0.14, 0.11, 0.08, 0.06],
                [0.17, 0.17, 0.16, 0.13, 0.11, 0.08, 0.06],
                [0.18, 0.18, 0.16, 0.13, 0.11, 0.08, 0.06],
                [0.20, 0.20, 0.18, 0.15, 0.12, 0.08, 0.06],
                [0.22, 0.22, 0.20, 0.16, 0.13, 0.09, 0.06],
                [0.24, 0.24, 0.21, 0.17, 0.13, 0.09, 0.06],
                [0.25, 0.25, 0.22, 0.18, 0.14, 0.09, 0.06],
                [0.25, 0.25, 0.22, 0.18, 0.14, 0.09, 0.06]]
    if posture == 'seated':
        fp_table = [[0.20, 0.23, 0.21, 0.21, 0.18, 0.16, 0.12],
                    [0.20, 0.23, 0.20, 0.20, 0.19, 0.16, 0.12],
                    [0.20, 0.23, 0.21, 0.20, 0.18, 0.15, 0.12],
                    [0.19, 0.23, 0.20, 0.20, 0.18, 0.15, 0.12],
                    [0.18, 0.21, 0.19, 0.19, 0.17, 0.14, 0.12],
                    [0.16, 0.20, 0.18, 0.18, 0.16, 0.13, 0.12],
                    [0.15, 0.18, 0.17, 0.17, 0.15, 0.13, 0.12],
                    [0.16, 0.18, 0.16, 0.16, 0.14, 0.13, 0.12],
                    [0.18, 0.18, 0.16, 0.14, 0.14, 0.12, 0.12],
                    [0.19, 0.18, 0.15, 0.13, 0.13, 0.12, 0.12],
                    [0.21, 0.18, 0.14, 0.12, 0.12, 0.12, 0.12],
                    [0.21, 0.17, 0.13, 0.11, 0.11, 0.12, 0.12],
                    [0.21, 0.17, 0.12, 0.11, 0.11, 0.11, 0.12]]

    if posture == 'supine':
        alt_temp = sol_altitude
        sol_altitude = math.abs(90 - sol_azimuth)
        sol_azimuth = alt_temp

    alt_range = [0, 15, 30, 45, 60, 75, 90]
    az_range = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    alt_i = find_span(alt_range, sol_altitude)
    az_i = find_span(az_range, sol_azimuth)
    fp11 = fp_table[az_i][alt_i]
    fp12 = fp_table[az_i][alt_i + 1]
    fp21 = fp_table[az_i + 1][alt_i]
    fp22 = fp_table[az_i + 1][alt_i + 1]
    az1 = az_range[az_i]
    az2 = az_range[az_i + 1]
    alt1 = alt_range[alt_i]
    alt2 = alt_range[alt_i + 1]
    fp = fp11 * (az2 - sol_azimuth) * (alt2 - sol_altitude)
    fp += fp21 * (sol_azimuth - az1) * (alt2 - sol_altitude)
    fp += fp12 * (az2 - sol_azimuth) * (sol_altitude - alt1)
    fp += fp22 * (sol_azimuth - az1) * (sol_altitude - alt1)
    fp /= (az2 - az1) * (alt2 - alt1)

    f_eff = 0.725
    if posture == 'seated':
        f_eff = 0.696

    sw_abs = asw
    lw_abs = 0.95

    e_diff = f_eff * f_svv * 0.5 * sol_transmittance * i_diff
    e_direct = fp * sol_transmittance * f_bes * sol_radiation_dir
    e_refl = f_eff * f_svv * 0.5 * sol_transmittance * (sol_radiation_dir * math.sin(sol_altitude * deg_to_rad) + i_diff) * floor_reflectance

    e_solar = e_diff + e_direct + e_refl
    erf = e_solar * (sw_abs / lw_abs)
    d_mrt = erf / (hr * f_eff)

    # print(fp, e_diff, e_direct, e_refl, e_solar, erf, d_mrt)

    return {"erf": round(erf, 1), "delta_mrt": round(d_mrt, 1)}


# add the following models:
# todo radiant_tmp_asymmetry
# todo draft
# todo floor_surface_tmp
# todo effective_tmp
# more info here: https://www.rdocumentation.org/packages/comf/versions/0.1.9
# more info here: https://rdrr.io/cran/comf/man/

if __name__ == "__main__":
    pass
