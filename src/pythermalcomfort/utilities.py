import warnings

warnings.simplefilter("always")


def check_standard_compliance(standard, **kwargs):
    params = dict()
    params['standard'] = standard
    for key, value in kwargs.items():
        params[key] = value

    if params['standard'] == 'utci':
        for key, value in params.items():
            if key == 'v':
                if value > 17:
                    warnings.warn("UTCI wind speed applicability limits between 0.5 and 17 m/s", UserWarning)
                elif value < 0.5:
                    warnings.warn("UTCI wind speed applicability limits between 0.5 and 17 m/s", UserWarning)

    elif params['standard'] == 'ashrae':  # based on table 7.3.4 ashrae 55 2017
        for key, value in params.items():
            if key in ['tdb', 'tr']:
                if value > 40 or value < 10:
                    warnings.warn("ASHRAE air temperature applicability limits between 10 and 40 °C", UserWarning)
            if key in ['v', 'vr']:
                if value > 2 or value < 0:
                    warnings.warn("ASHRAE air velocity applicability limits between 0 and 2 m/s", UserWarning)
            if key == 'met':
                if value > 2 or value < 1:
                    warnings.warn("ASHRAE met applicability limits between 1.0 and 2.0 met", UserWarning)
            if key == 'clo':
                if value > 1.5 or value < 0:
                    warnings.warn("ASHRAE clo applicability limits between 0.0 and 1.5 clo", UserWarning)
            if key == 'v_limited':
                if value > 0.2:
                    raise ValueError("This equation is only applicable for air speed lower than 0.2 m/s")

    elif params['standard'] == 'iso':  # based on ISO 7730:2005 page 3
        for key, value in params.items():
            if key == 'tdb':
                if value > 30 or value < 10:
                    warnings.warn("ISO air temperature applicability limits between 10 and 30 °C", UserWarning)
            if key == 'tr':
                if value > 40 or value < 10:
                    warnings.warn("ISO air temperature applicability limits between 10 and 40 °C", UserWarning)
            if key in ['v', 'vr']:
                if value > 1 or value < 0:
                    warnings.warn("ISO air velocity applicability limits between 0 and 1 m/s", UserWarning)
            if key == 'met':
                if value > 4 or value < 0.8:
                    warnings.warn("ISO met applicability limits between 0.8 and 4.0 met", UserWarning)
            if key == 'clo':
                if value > 2 or value < 0:
                    warnings.warn("ISO clo applicability limits between 0.0 and 2 clo", UserWarning)


def secant(f, a, b, n):
    '''Approximate solution of f(x)=0 on interval [a,b] by the secant method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    n : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    m_N : number
        The x intercept of the secant line on the the Nth interval
            m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))
        The initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0
        for some intercept m_n then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iterations, the secant method fails and return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> secant(f,1,2,5)
    1.6180257510729614
    '''
    if f(a) * f(b) >= 0:
        print("Secant method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1, n + 1):
        m_n = a_n - f(a_n) * (b_n - a_n) / (f(b_n) - f(a_n))
        f_m_n = f(m_n)
        if f(a_n) * f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n) * f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            # print("Found exact solution.")
            return m_n
        else:
            # print("Secant method fails.")
            return None
    return a_n - f(a_n) * (b_n - a_n) / (f(b_n) - f(a_n))


def bisection(f, a, b, max_iterations, error=0.0001):
    """Approximate solution of f(x)=0 on interval [a,b] by the bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    max_iterations : (positive) integer
        The number of iterations to implement.
    error: (positive) float
        The max error that is accepted

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> bisection(f,1,2,25)
    1.618033990263939
    >>> f = lambda x: (2*x - 1)*(x - 3)
    >>> bisection(f,0,1,10)
    0.5
    """
    if f(a) * f(b) >= 0:
        # print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    n = 0
    while error < abs(f(a_n) - f(b_n)):
        m_n = (a_n + b_n) / 2
        f_m_n = f(m_n)

        n += 1
        if n > max_iterations:
            return None

        if error > abs(f(a_n) - f(b_n)):
            return (a_n + b_n) / 2
        elif f(a_n) * f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n) * f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            # print("Found exact solution.")
            return m_n
        else:
            # print("Bisection method fails.")
            return None
    return (a_n + b_n) / 2


#: This dictionary contains the met values of typical tasks.
met_typical_tasks = {
    'Sleeping': 0.7,
    'Reclining': 0.8,
    'Seated, quiet': 1.0,
    'Reading, seated': 1.0,
    'Writing': 1.0,
    'Typing': 1.1,
    'Standing, relaxed': 1.2,
    'Filing, seated': 1.2,
    'Flying aircraft, routine': 1.2,
    'Filing, standing': 1.4,
    'Driving a car': 1.5,
    'Walking about': 1.7,
    'Cooking': 1.8,
    'Table sawing': 1.8,
    'Walking 2mph (3.2kmh)': 2.0,
    'Lifting/packing': 2.1,
    'Seated, heavy limb movement': 2.2,
    'Light machine work': 2.2,
    'Flying aircraft, combat': 2.4,
    'Walking 3mph (4.8kmh)': 2.6,
    'House cleaning': 2.7,
    'Driving, heavy vehicle': 3.2,
    'Dancing': 3.4,
    'Calisthenics': 3.5,
    'Walking 4mph (6.4kmh)': 3.8,
    'Tennis': 3.8,
    'Heavy machine work': 4.0,
    'Handling 100lb (45 kg) bags': 4.0,
    'Pick and shovel work': 4.4,
    'Basketball': 6.3,
    'Wrestling': 7.8,
}

#: This dictionary contains the total clothing insulation of typical typical ensembles.
clo_typical_ensembles = {
    'Walking shorts, short-sleeve shirt': 0.36,
    'Typical summer indoor clothing': 0.5,
    'Knee-length skirt, short-sleeve shirt, sandals, underwear': 0.54,
    'Trousers, short-sleeve shirt, socks, shoes, underwear': 0.57,
    'Trousers, long-sleeve shirt': 0.61,
    'Knee-length skirt, long-sleeve shirt, full slip': 0.67,
    'Sweat pants, long-sleeve sweatshirt': 0.74,
    'Jacket, Trousers, long-sleeve shirt': 0.96,
    'Typical winter indoor clothing': 1.0,
}

#: This dictionary contains the clo values of individual clothing elements. To calculate the total clothing insulation you need to add these values together.
clo_individual_garments = {
    'Metal chair': 0.00,
    'Bra': 0.01,
    'Wooden stool': 0.01,
    'Ankle socks': 0.02,
    'Shoes or sandals': 0.02,
    'Slippers': 0.03,
    'Panty hose': 0.02,
    'Calf length socks': 0.03,
    'Women\'s underwear': 0.03,
    'Men\'s underwear': 0.04,
    'Knee socks (thick)': 0.06,
    'Short shorts': 0.06,
    'Walking shorts': 0.08,
    'T-shirt': 0.08,
    'Standard office chair': 0.10,
    'Executive chair': 0.15,
    'Boots': 0.1,
    'Sleeveless scoop-neck blouse': 0.12,
    'Half slip': 0.14,
    'Long underwear bottoms': 0.15,
    'Full slip': 0.16,
    'Short-sleeve knit shirt': 0.17,
    'Sleeveless vest (thin)': 0.1,
    'Sleeveless vest (thick)': 0.17,
    'Sleeveless short gown (thin)': 0.18,
    'Short-sleeve dress shirt': 0.19,
    'Sleeveless long gown (thin)': 0.2,
    'Long underwear top': 0.2,
    'Thick skirt': 0.23,
    'Long-sleeve dress shirt': 0.25,
    'Long-sleeve flannel shirt': 0.34,
    'Long-sleeve sweat shirt': 0.34,
    'Short-sleeve hospital gown': 0.31,
    'Short-sleeve short robe (thin)': 0.34,
    'Short-sleeve pajamas': 0.42,
    'Long-sleeve long gown': 0.46,
    'Long-sleeve short wrap robe (thick)': 0.48,
    'Long-sleeve pajamas (thick)': 0.57,
    'Long-sleeve long wrap robe (thick)': 0.69,
    'Thin trousers': 0.15,
    'Thick trousers': 0.24,
    'Sweatpants': 0.28,
    'Overalls': 0.30,
    'Coveralls': 0.49,
    'Thin skirt': 0.14,
    'Long-sleeve shirt dress (thin)': 0.33,
    'Long-sleeve shirt dress (thick)': 0.47,
    'Short-sleeve shirt dress': 0.29,
    'Sleeveless, scoop-neck shirt (thin)': 0.23,
    'Sleeveless, scoop-neck shirt (thick)': 0.27,
    'Long sleeve shirt (thin)': 0.25,
    'Long sleeve shirt (thick)': 0.36,
    'Single-breasted coat (thin)': 0.36,
    'Single-breasted coat (thick)': 0.44,
    'Double-breasted coat (thin)': 0.42,
    'Double-breasted coat (thick)': 0.48,
}
