from pythermalcomfort.models import utci

utci_val = utci(tdb=29, tr=30, v=1, rh=60)
print(utci_val)

utci_val = utci(tdb=29, tr=30, v=2, rh=60)
print(utci_val)

utci_val = utci(tdb=77, tr=77, v=6.56168, rh=60, units="IP")
print(utci_val)
