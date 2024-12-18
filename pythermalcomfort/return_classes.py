from dataclasses import dataclass
from typing import Union, List


@dataclass(frozen=True)
class SetTmp:
    """
    Dataclass to represent the Standard Effective Temperature (SET).

    Attributes
    ----------
    set : float or list of floats
        Standard effective temperature, [°C].
    """

    set: Union[float, List[float]]

    def __getitem__(self, item):
        return getattr(self, item)
