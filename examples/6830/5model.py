
import os
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

SEED   = 42
SEASON = 12
np.random.seed(SEED)


VAL_START = pd.Timestamp("2022-09-01")   
VAL_END   = pd.Timestamp("2024-02-01")   
TEST_START = pd.Timestamp("2024-03-01")  
TEST_END   = pd.Timestamp("2025-08-01")  
VAL_INDEX  = pd.date_range(VAL_START,  VAL_END,  freq="MS")  
TEST_INDEX = pd.date_range(TEST_START, TEST_END, freq="MS")  
H = len(TEST_INDEX)


DATE_COL, VALUE_COL = None, None

def r4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return float(np.round(x, 4))


def resolve_data_path(preferred_name="datacleaned.csv"):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", type=str, default=None)
    args, _ = parser.parse_known_args()
    here = Path(__file__).resolve().parent
    if args.data:
        p = Path(args.data).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--data 文件不存在：{p}")
        print(f"[INFO] Using data file from --data: {p}")
        return p
    for p in [Path.cwd()/preferred_name, here/preferred_name]:
        if p.exists():
            print(f"[INFO] Using data file: {p.resolve()}")
            return p.resolve()
    found = list(here.rglob(preferred_name))
    if found:
        print(f"[INFO] Using data file (found by rglob): {found[0].resolve()}")
        return found[0].resolve()
    raise FileNotFoundError("未找到数据文件，请提供 datacleaned.csv 或使用 --data 指定。")

DATA_PATH = resolve_data_path("datacleaned.csv")


def infer_columns(df: pd.DataFrame, date_col=None, value_col=None):
    if date_col and value_col:
        return date_col, value_col
    cols = list(df.columns)
    date_candidates  = [c for c in cols if c.lower() in ["date","ds","month","time","timestamp"]]
    value_candidates = [c for c in cols if c.lower() in ["value","y","count","activity","transport",
                                                         "ridership","passengers","vol","volume","cnt"]]
    dcol = date_col  or (date_candidates[0]  if date_candidates  else cols[0])
    vcol = value_col or (value_candidates[0] if value_candidates else cols[1])
    return dcol, vcol

df = pd.read_csv(DATA_PATH)
dcol, vcol = infer_columns(df, DATE_COL, VALUE_COL)
s_raw = pd.Series(df[vcol].values, index=pd.to_datetime(df[dcol])).sort_index()
dup = int(s_raw.index.duplicated().sum())

start = s_raw.index.min().to_period("M").start_time
end   = s_raw.index.max().to_period("M").start_time
idx_full = pd.date_range(start, end, freq="MS")
s = s_raw.groupby(s_raw.index).last().reindex(idx_full)
missing_total = int(s.isna().sum())
s = s.ffill().bfill()
y = s.asfreq("MS")

print(f"[INFO] data span: {y.index.min().date()} → {y.index.max().date()} (n={len(y)})")


if y.index.max() < VAL_END:
    raise ValueError(f"数据最晚到 {y.index.max().date()}，不足以覆盖验证窗至 2024-02-01。")


eda = {
    "min_date": str(y.index.min().date()),
    "max_date": str(y.index.max().date()),
    "n_obs": int(len(y)),
    "duplicates_in_raw_index": dup,
    "missing_months_filled_total": missing_total,
    "mean": r4(float(y.mean())),
    "std":  r4(float(y.std())),
    "min":  r4(float(y.min())),
    "max":  r4(float(y.max())),
}
print("\n=== EDA SUMMARY (no plots) ===")
for k,v in eda.items():
    print(f"{k:30s}: {v}")


TRAIN_END_FOR_VAL = VAL_START - pd.offsets.MonthBegin()  # 2022-08-01
train_for_val = y.loc[:TRAIN_END_FOR_VAL]
val_true      = y.reindex(VAL_INDEX)  
if len(val_true.dropna()) != len(VAL_INDEX):
    print("The original data of the verification window is missing, which has been conservatively filled in the previous step;")


FINAL_TRAIN_END = VAL_END  
final_train = y.loc[:FINAL_TRAIN_END]

print(f"\n[Split] train_for_val={len(train_for_val)} (≤ {TRAIN_END_FOR_VAL.date()}) ; "
      f"val={len(val_true)} ({VAL_START.date()}→{VAL_END.date()}) ; "
      f"final_train={len(final_train)} (≤ {FINAL_TRAIN_END.date()})")


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    idx = y_true.index.intersection(y_pred.index)
    return float(np.sqrt(np.mean((y_true.loc[idx].values - y_pred.loc[idx].values)**2)))

def seasonal_naive(y_tr: pd.Series, h: int, season: int = SEASON) -> pd.Series:
    last = y_tr.iloc[-season:]
    reps = int(np.ceil(h/season))
    vals = np.tile(last.values, reps)[:h]
    idx  = pd.date_range(y_tr.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    return pd.Series(vals, index=idx)


rows = []

# 1) Seasonal Naïve
fc = seasonal_naive(train_for_val, len(VAL_INDEX), season=SEASON).reindex(VAL_INDEX)
rows.append({"model":"snaive", "RMSE": r4(rmse(val_true, fc)), "AIC": np.nan, "BIC": np.nan})

# 2) Holt
holt_fit_val = ExponentialSmoothing(train_for_val, trend="add", seasonal=None)\
                 .fit(optimized=True, use_brute=False)
fc = holt_fit_val.forecast(len(VAL_INDEX)).reindex(VAL_INDEX)
rows.append({"model":"holt", "RMSE": r4(rmse(val_true, fc)), "AIC": r4(holt_fit_val.aic), "BIC": r4(holt_fit_val.bic)})

# 3) Holt-Winters
hw_fit_val = ExponentialSmoothing(train_for_val, trend="add", seasonal="add", seasonal_periods=SEASON)\
               .fit(optimized=True, use_brute=False)
fc = hw_fit_val.forecast(len(VAL_INDEX)).reindex(VAL_INDEX)
rows.append({"model":"holtwinters", "RMSE": r4(rmse(val_true, fc)), "AIC": r4(hw_fit_val.aic), "BIC": r4(hw_fit_val.bic)})

# 4) ARIMA(1,1,1)
arima_fit_val = SARIMAX(train_for_val, order=(1,1,1), seasonal_order=(0,0,0,0), trend="n").fit(disp=False)
fc = arima_fit_val.get_forecast(len(VAL_INDEX)).predicted_mean.reindex(VAL_INDEX)
rows.append({"model":"arima111", "RMSE": r4(rmse(val_true, fc)), "AIC": r4(arima_fit_val.aic), "BIC": r4(arima_fit_val.bic)})

# 5) SARIMA(1,1,1)(0,1,1)[12]
sarima_fit_val = SARIMAX(train_for_val, order=(1,1,1), seasonal_order=(0,1,1,SEASON), trend="n").fit(disp=False)
fc = sarima_fit_val.get_forecast(len(VAL_INDEX)).predicted_mean.reindex(VAL_INDEX)
rows.append({"model":"sarima", "RMSE": r4(rmse(val_true, fc)), "AIC": r4(sarima_fit_val.aic), "BIC": r4(sarima_fit_val.bic)})

summary = pd.DataFrame(rows).set_index("model").sort_values("RMSE")
print("\n=== MODEL COMPARISON on validation window (2022-09 → 2024-02) ===")
print(summary.to_string())

best = summary.index[0]
print(f"\nBest model by RMSE on validation: {best}")


def fit_and_forecast(model_name: str, y_tr: pd.Series, steps: int) -> pd.Series:
    if model_name == "snaive":
        return seasonal_naive(y_tr, steps, season=SEASON)
    elif model_name == "holt":
        m = ExponentialSmoothing(y_tr, trend="add", seasonal=None).fit(optimized=True, use_brute=False)
        return m.forecast(steps)
    elif model_name == "holtwinters":
        m = ExponentialSmoothing(y_tr, trend="add", seasonal="add", seasonal_periods=SEASON).fit(optimized=True, use_brute=False)
        return m.forecast(steps)
    elif model_name == "arima111":
        m = SARIMAX(y_tr, order=(1,1,1), seasonal_order=(0,0,0,0), trend="n").fit(disp=False)
        return m.get_forecast(steps).predicted_mean
    elif model_name == "sarima":
        m = SARIMAX(y_tr, order=(1,1,1), seasonal_order=(0,1,1,SEASON), trend="n").fit(disp=False)
        return m.get_forecast(steps).predicted_mean
    else:
        raise ValueError("unknown model")

fc_best = fit_and_forecast(best, final_train, H).reindex(TEST_INDEX)


out = pd.DataFrame({
    "date": fc_best.index.strftime("%Y-%m-01"),
    "forecast": fc_best.values
})
out_path = Path(__file__).resolve().parent / "MyForecast_202403_202508.csv"
out.to_csv(out_path, index=False, encoding="utf-8")
print(f"\n[OK] Saved 18-month forecast (2024-03→2025-08) -> {out_path}")
