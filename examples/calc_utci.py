import time

import numpy as np

from pythermalcomfort.models import utci

utci_val = utci(tdb=29, tr=30, v=1, rh=60)
print(utci_val)

utci_val = utci(tdb=60, tr=30, v=1, rh=60)
print(utci_val)
print(utci(tdb=60, tr=30, v=1, rh=60, limit_inputs=False))

utci_val = utci(tdb=77, tr=77, v=6.56168, rh=60, units="IP")
print(utci_val)

# numpy examples
utci_val = utci(
    tdb=[29, 29, 25],
    tr=[30, 30, 25],
    v=[1, 2, 1],
    rh=[60, 60, 50],
)
print(utci_val.utci)

utci_val = utci(
    tdb=[29, 29, 25],
    tr=[30, 30, 25],
    v=[1, 2, 1],
    rh=[60, 60, 50],
)
print(utci_val)
print(utci_val["utci"])
print(utci_val["stress_category"])

iterations = 100000
tdb = np.empty(iterations)
tdb.fill(25)
start = time.time()
utci_val = utci(tdb=tdb, tr=30, v=1, rh=60)
end = time.time()
print(end - start)
