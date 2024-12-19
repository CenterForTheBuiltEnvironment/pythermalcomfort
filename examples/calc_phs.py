import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pythermalcomfort.models import phs

plt.close("all")

t_array = range(30, 52, 1)
rh_array = range(0, 100, 1)

df = pd.DataFrame(list(itertools.product(t_array, rh_array)), columns=["tdb", "rh"])

results = phs(
    tdb=df.tdb.values,
    tr=df.tdb.values,
    rh=df.rh.values,
    v=0.3,
    met=2,
    clo=0.5,
    posture="sitting",
)

df["t_re"] = results.t_re

plt.figure()
pivot = df.pivot(columns="tdb", index="rh", values="t_re")
pivot = pivot.sort_index(ascending=False)
ax = sns.heatmap(pivot)
plt.tight_layout()
plt.show()
