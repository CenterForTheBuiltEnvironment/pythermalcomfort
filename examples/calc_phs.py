import matplotlib.pyplot as plt

from pythermalcomfort.models import phs
import pandas as pd
import seaborn as sns

plt.close("all")

for v in [0.2, 0.8]:

    results = []
    for tdb in range(30, 52, 1):
        for rh in range(0, 100, 1):
            result = phs(tdb=tdb, tr=tdb, rh=rh, v=v, met=55, clo=0.5, posture=2)
            result["tdb"] = tdb
            result["rh"] = rh
            results.append(result)

    df = pd.DataFrame.from_dict(results)

    plt.figure()
    pivot = df.pivot("tdb", "rh", "t_re")
    pivot = pivot.sort_index(ascending=False)
    ax = sns.heatmap(pivot)
    plt.title(f"velocity {v}")
    plt.tight_layout()
    plt.show()
