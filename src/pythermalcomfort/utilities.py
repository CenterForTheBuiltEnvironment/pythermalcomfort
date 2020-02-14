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
