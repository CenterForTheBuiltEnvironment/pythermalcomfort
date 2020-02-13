import warnings

warnings.simplefilter("always")


def check_standard_compliance(standard, **kwargs):  # todo enter all the limits
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

    elif params['standard'] == 'ashrae':
        for key, value in params.items():
            if key == 'met':
                if value > 2:
                    warnings.warn("ASHRAE met applicability limits between 1.0 and 2.0 met", UserWarning)
                elif value < 1:
                    warnings.warn("ASHRAE met applicability limits between 1.0 and 2.0 met", UserWarning)
            if key == 'clo':
                if value > 1.5:
                    warnings.warn("ASHRAE clo applicability limits between 0.0 and 1.5 clo", UserWarning)
