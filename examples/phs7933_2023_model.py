import math
import numpy as np

def PHS79332023(Ta, Tr, Pa, Va, M, posture, Icl, Ap, Fr, defspeed, Walksp, defdir, THETA, accl):
    ConstTeq = math.exp(-1/10)
    ConstTsk = math.exp(-1/3)
    ConstSW = math.exp(-1/10)

    weight = 75
    height = 1.8
    DRINK = 1

    Adu = 0.202 * weight**0.425 * height**0.725
    aux = 3490 * weight / Adu

    SWmax = 400
    Wmax = 0.85
    if accl == 1:
        SWmax = 500
        Wmax = 1

    if DRINK == 0:
        Dmax = 0.03 * weight * 1000
    else:
        Dmax = 0.05 * weight * 1000

    Duration = 480
    Met = M / Adu
    Work = 0
    imst = 0.38

    if posture == 1:
        Ardu = 0.77
    elif posture == 2:
        Ardu = 0.7
    else:
        Ardu = 0.67

    Ecl = 0.97

    Iclst = Icl * 0.155
    fcl = 1 + 0.28 * Icl
    Iast = 0.111
    Itotst = Iclst + Iast / fcl

    if defspeed > 0:
        if defdir == 1:
            Var = abs(Va - Walksp * math.cos(math.pi * THETA / 180))
        else:
            Var = Walksp if Va < Walksp else Va
    else:
        Walksp = 0.0052 * (Met - 58)
        Walksp = min(Walksp, 0.7)
        Var = Va

    Vaux = min(Var, 3)
    Waux = min(Walksp, 1.5)

    CORcl = 1.044 * math.exp((0.066 * Vaux - 0.398) * Vaux + (0.094 * Waux - 0.378) * Waux)
    CORcl = min(CORcl, 1)
    CORia = math.exp((0.047 * Vaux - 0.472) * Vaux + (0.117 * Waux - 0.342) * Waux)
    CORia = min(CORia, 1)
    CORtot = CORcl if Icl > 0.6 else ((0.6 - Icl) * CORia + Icl * CORcl) / 0.6
    Itotdyn = Itotst * CORtot
    Iadyn = CORia * Iast
    Icldyn = Itotdyn - Iadyn / fcl

    CORe = (2.6 * CORtot - 6.5) * CORtot + 4.9
    imdyn = min(imst * CORe, 0.9)
    Rtdyn = Itotdyn / imdyn / 16.7

    Tre, Tcr, Tsk = 36.8, 36.8, 34.1
    Tcreq, TskTcrwg = 36.8, 0.3
    SWp, SWtot = 0, 0
    Dlimtcr, Dlimloss = 999, 999

    for Time in range(1, Duration + 1):
        Tre0, Tcr0, Tsk0, Tcreq0, TskTcrwg0 = Tre, Tcr, Tsk, Tcreq, TskTcrwg
        Tcreqm = 0.0036 * (Met - 55) + 36.8
        Tcreq = Tcreq0 * ConstTeq + Tcreqm * (1 - ConstTeq)
        dStoreq = (aux / 60) * (Tcreq - Tcreq0) * (1 - TskTcrwg0)

        Tskeqcl = 12.165 + 0.02017 * Ta + 0.04361 * Tr + 0.19354 * Pa - 0.25315 * Va + 0.005346 * Met + 0.51274 * Tre
        Tskeqnu = 7.191 + 0.064 * Ta + 0.061 * Tr + 0.198 * Pa - 0.348 * Va + 0.616 * Tre

        if Icl >= 0.6:
            Tskeq = Tskeqcl
        elif Icl <= 0.2:
            Tskeq = Tskeqnu
        else:
            Tskeq = Tskeqnu + 2.5 * (Tskeqcl - Tskeqnu) * (Icl - 0.2)

        Tsk = Tsk0 * ConstTsk + Tskeq * (1 - ConstTsk)
        if Time == 1:
            Tsk = Tskeq

        Psk = 0.6105 * math.exp(17.27 * Tsk / (Tsk + 237.3))
        Z = 3.5 + 5.2 * Var if Var <= 1 else 8.7 * Var ** 0.6
        auxR = 5.67e-8 * Ardu
        EclR = (1 - Ap) * Ecl + Ap * (1 - Fr)
        Tcl = Tr + 0.1
        Hc = max(2.38 * abs(Tsk - Ta) ** 0.25, Z)
        Hr = EclR * auxR * ((Tcl + 273) ** 4 - (Tr + 273) ** 4) / (Tcl - Tr)
        Tcl1 = (fcl * (Hc * Ta + Hr * Tr) + Tsk / Icldyn) / (fcl * (Hc + Hr) + 1 / Icldyn)
        while abs(Tcl - Tcl1) > 0.001:
            Tcl = (Tcl + Tcl1) / 2
            Hr = EclR * auxR * ((Tcl + 273) ** 4 - (Tr + 273) ** 4) / (Tcl - Tr)
            Tcl1 = (fcl * (Hc * Ta + Hr * Tr) + Tsk / Icldyn) / (fcl * (Hc + Hr) + 1 / Icldyn)

        Texp = 28.56 + 0.115 * Ta + 0.641 * Pa
        Cres = 0.001516 * Met * (Texp - Ta)
        Eres = 0.00127 * Met * (59.34 + 0.53 * Ta - 11.63 * Pa)
        Conv = fcl * Hc * (Tcl - Ta)
        Rad = fcl * Hr * (Tcl - Tr)
        Emax = (Psk - Pa) / Rtdyn
        Ereq = Met - dStoreq - Work - Cres - Eres - Conv - Rad

        if Ereq <= 0:
            SWreq = 0
        elif Emax <= 0:
            SWreq = SWmax
        else:
            wreq = Ereq / Emax
            if wreq >= 1.7:
                SWreq = SWmax
            else:
                Eveff = (2 - wreq) ** 2 / 2 if wreq > 1 else 1 - wreq ** 2 / 2
                SWreq = min(Ereq / Eveff, SWmax)

        SWp = SWp * ConstSW + SWreq * (1 - ConstSW)
        if SWp <= 0:
            Ep = 0
            SWp = 0
        else:
            k = Emax / SWp
            wp = -k + math.sqrt(k * k + 2) if k >= 0.5 else 1
            wp = min(wp, Wmax)
            Ep = wp * Emax

        dStorage = Ereq - Ep + dStoreq
        Tcr1 = Tcr0
        while True:
            TskTcrwg = max(min(0.3 - 0.09 * (Tcr1 - 36.8), 0.3), 0.1)
            Tcr = dStorage / (aux / 60) + Tsk0 * TskTcrwg0 / 2 - Tsk * TskTcrwg / 2
            Tcr = (Tcr + Tcr0 * (1 - TskTcrwg0 / 2)) / (1 - TskTcrwg / 2)
            if abs(Tcr - Tcr1) <= 0.001:
                break
            Tcr1 = (Tcr + Tcr1) / 2

        Tre = Tre0 + (2 * Tcr - 1.962 * Tre0 - 1.31) / 9
        SWtot += SWp + Eres
        SWtotg = SWtot * 2.67 * Adu / 1.8 / 60

        if Dlimloss == 999 and SWtotg >= Dmax:
            Dlimloss = Time
        if Dlimtcr == 999 and Tre >= 38:
            Dlimtcr = Time

    return Tre, SWp, SWtotg, Dlimtcr, Dlimloss
