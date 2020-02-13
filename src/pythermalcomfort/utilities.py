import warnings

warnings.simplefilter("always")


def check_standard_compliance(**kwargs):  # todo enter all the limits
    for key, value in kwargs.items():
        if key == 'met':
            if value > 2:
                warnings.warn("ASHRAE met applicability limits between 1.0 and 2.0 clo", UserWarning)
            elif value < 1:
                warnings.warn("ASHRAE met applicability limits between 1.0 and 2.0 clo", UserWarning)
        if key == 'clo':
            if value > 1.5:
                warnings.warn("ASHRAE clo applicability limits between 0.0 and 1.5 clo", UserWarning)
