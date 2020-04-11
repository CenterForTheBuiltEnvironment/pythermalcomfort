from pythermalcomfort.models import adaptive_ashrae
from pprint import pprint

result = adaptive_ashrae(tdb=25, tr=25, t_running_mean=23, v=0.2)

pprint(result)
