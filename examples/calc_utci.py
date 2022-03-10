from pythermalcomfort.models import utci
import numpy as np

utci_val = utci(tdb=29, tr=30, v=1, rh=60)
print(utci_val)

utci_val = utci(tdb=29, tr=30, v=2, rh=60)
print(utci_val)

utci_val = utci(tdb=77, tr=77, v=6.56168, rh=60, units="IP")
print(utci_val)

# numpy examples
utci_val = utci(
    tdb=np.array([29, 29, 25]),
    tr=np.array([30, 30, 25]),
    v=np.array([1, 2, 1]),
    rh=np.array([60, 60, 50]),
)
print(utci_val)

utci_val = utci(
    tdb=np.array([29, 29, 25]),
    tr=np.array([30, 30, 25]),
    v=np.array([1, 2, 1]),
    rh=np.array([60, 60, 50]),
    return_stress_category=True,
)
print(utci_val)
