from collections.abc import Callable
from enum import Enum

"""
This module handles the main Mean Radiant Temperature MRT functions inlcuding helper methods to calucalte longwave radiation
"""

# copy over 4th power formular called: forth_power, this mrt.forth_power(...)
# add


class MRTMethod(Enum):
    ForthPower = "4Power"
    NoGeometry = "NoGeometry"
    AnotherDifferentMehtod = "AnotherDifferentMehtod"

    # Need to define this, but that is the idea.


class MRT:
    """Manage multiple Mean Radiant Temperature calculation methods."""

    _methods: dict[str, Callable[..., float]] = {}

    # MRT methods
    @staticmethod
    def mrt_simple(tdb: float, tr: float, v: float) -> float:
        return 0.5 * (tdb + tr)

    @staticmethod
    def mrt_asw(asw: float, tdb: float, tr: float) -> float:
        return asw * tdb + (1 - asw) * tr

    _methods: dict[MRTMethod, Callable[..., float]] = {
        MRTMethod.ForthPower: mrt_simple.__func__,
        MRTMethod.NoGeometry: mrt_asw.__func__,
    }

    @classmethod
    def calculate(cls, method: MRTMethod, **kwargs) -> float:
        if method not in cls._methods:
            raise ValueError(f"Unknown MRT method '{method}'")
        return cls._methods[method](**kwargs)


# ----------------------
# Testing:
# ----------------------
mrt_val = MRT.calculate(MRTMethod.ForthPower, tdb=25, tr=30, v=0.1)
mrt_val2 = MRT.calculate(MRTMethod.NoGeometry, tdb=25, tr=30, asw=0.6)

print(mrt_val, mrt_val2)
