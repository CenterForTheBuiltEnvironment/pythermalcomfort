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
            if key in ['ta', 'tr']:
                if value > 40 or value < 10:
                    warnings.warn("ASHRAE air temperature applicability limits between 10 and 50 °C", UserWarning)
            if key in ['v', 'vr']:
                if value > 2 or value < 0:
                    warnings.warn("ASHRAE air velocity applicability limits between 0 and 2 m/s", UserWarning)
            if key == 'met':
                if value > 2 or value < 1:
                    warnings.warn("ASHRAE met applicability limits between 1.0 and 2.0 met", UserWarning)
            if key == 'clo':
                if value > 1.5 or value < 0:
                    warnings.warn("ASHRAE clo applicability limits between 0.0 and 1.5 clo", UserWarning)

    elif params['standard'] == 'iso':  # based on table 7.3.4 ashrae 55 2017
        for key, value in params.items():
            if key == 'ta':
                if value > 30 or value < 10:
                    warnings.warn("ISO air temperature applicability limits between 10 and 50 °C", UserWarning)
            if key == 'tr':
                if value > 40 or value < 10:
                    warnings.warn("ISO air temperature applicability limits between 10 and 50 °C", UserWarning)
            if key in ['v', 'vr']:
                if value > 1 or value < 0:
                    warnings.warn("ISO air velocity applicability limits between 0 and 2 m/s", UserWarning)
            if key == 'met':
                if value > 4 or value < 0.8:
                    warnings.warn("ISO met applicability limits between 1.0 and 2.0 met", UserWarning)
            if key == 'clo':
                if value > 2 or value < 0:
                    warnings.warn("ISO clo applicability limits between 0.0 and 1.5 clo", UserWarning)


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
    while (error < abs(f(a_n) - f(b_n))) or (n < max_iterations):
        # for n in range(1, max_iterations + 1):
        m_n = (a_n + b_n) / 2
        f_m_n = f(m_n)
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
