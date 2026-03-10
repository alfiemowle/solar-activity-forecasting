"""
Solar Activity Long-Term Forecasting Pipeline (144-month / 12-year horizon).
Loads pre-engineered monthly features from data/features_monthly.parquet.
Direct multi-step XGBoost ensemble, two outputs per month:
  predicted_point -- trained on raw monthly ISN targets
  predicted_trend -- trained on 13-month trailing smoothed targets
"""
import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# --- Configuration -------------------------------------------------------
FORECAST_HORIZON    = 144   # months (12 years)
VALIDATION_WINDOW   = 144   # months (12 years)
ENSEMBLE_SEEDS      = [42, 7, 123, 1, 2, 3, 99]
HORIZONS            = [1, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144]
CAL_WINDOW          = 120   # months (10 years) for calibration

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MONTHLY_PARQUET = os.path.join(BASE_DIR, "data", "features_monthly.parquet")
OUT_DIR         = os.path.join(BASE_DIR, "outputs")

FEATURE_COLS = (
    [f"lag_{l}" for l in [1, 2, 3, 6, 12, 24, 48, 120, 132]]
    + [f"roll_{s}_{w}" for w in [12, 24, 60, 120] for s in ["mean", "std", "max"]]
    + [f"roll_min_{w}" for w in [60, 120]]
    + [f"ema_{sp}" for sp in [12, 24, 60]]
    + [f"mom_{l}" for l in [12, 24, 48, 120, 132]]
    + ["cycle_sin", "cycle_cos", "gleissberg_sin", "gleissberg_cos"]
)


# --- Load features -------------------------------------------------------
def load_features():
    if not os.path.isfile(MONTHLY_PARQUET):
        raise FileNotFoundError(
            "features_monthly.parquet not found. Run data_prepare.py first."
        )
    feat_df = pd.read_parquet(MONTHLY_PARQUET)
    feat_df["Date"] = pd.to_datetime(feat_df["Date"])
    last_data_month = feat_df["Date"].max()
    print("Loaded features_monthly.parquet")
    print(f"  Monthly rows   : {len(feat_df)}")
    print(f"  Date range     : {feat_df['Date'].iloc[0].date()} -> {feat_df['Date'].iloc[-1].date()}")
    print(f"  Last month     : {last_data_month.date()}")
    return feat_df, last_data_month


# --- Ensemble training ---------------------------------------------------
def train_ensemble(X_train, y_train):
    models = []
    for seed in ENSEMBLE_SEEDS:
        m = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        m.fit(X_train, y_train)
        models.append(m)
    return models


def predict_ensemble(models, X):
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


# --- Calibrate uncertainty -----------------------------------------------
def calibrate_sigma(horizon_models, feat_df, cutoff_dt, target_isn=None):
    """target_isn: if None uses raw MonthlyISN; else the supplied series."""
    train_rows = feat_df[feat_df["Date"] < cutoff_dt]
    cal_rows   = train_rows.iloc[-CAL_WINDOW:] if len(train_rows) > CAL_WINDOW else train_rows

    base = target_isn if target_isn is not None else feat_df["MonthlyISN"]
    cal_sigma = {}
    for h, models in horizon_models.items():
        target = base.shift(-h)
        idx = cal_rows.index
        valid_mask = target.loc[idx].notna()
        valid_idx  = idx[valid_mask]
        if len(valid_idx) == 0:
            cal_sigma[h] = 25.0
            continue
        X_cal = feat_df.loc[valid_idx, FEATURE_COLS].values
        y_cal = target.loc[valid_idx].values
        mean_pred, _ = predict_ensemble(models, X_cal)
        rmse = float(np.sqrt(np.mean((y_cal - mean_pred) ** 2)))
        # Floor scales linearly: 5.0 for h<=24, rising to 40.0 at h=144
        if h <= 24:
            floor = 5.0
        else:
            floor = 25.0 + (h - 24) / (144 - 24) * (40.0 - 25.0)
        cal_sigma[h] = max(rmse, floor)
    return cal_sigma


# --- Run mode (train + predict) ------------------------------------------
def run_mode(feat_df, cutoff_date, pred_base_date=None):
    if pred_base_date is None:
        pred_base_date = cutoff_date - pd.DateOffset(months=1)

    smooth_isn = feat_df["SmoothedISN"].copy()

    cutoff_dt    = pd.Timestamp(cutoff_date)
    pred_base_dt = pd.Timestamp(pred_base_date)

    train_mask = feat_df["Date"] < cutoff_dt

    print("  Training point models...")
    point_models = {}
    for h in HORIZONS:
        target    = feat_df["MonthlyISN"].shift(-h)
        valid_idx = train_mask & target.notna()
        X_tr = feat_df.loc[valid_idx, FEATURE_COLS].values
        y_tr = target[valid_idx].values
        if len(X_tr) == 0:
            raise RuntimeError(f"No training data for h={h}")
        point_models[h] = train_ensemble(X_tr, y_tr)
        print(f"    h={h:4d}  ({len(X_tr)} rows)")

    print("  Training trend models...")
    trend_models = {}
    for h in HORIZONS:
        target    = smooth_isn.shift(-h)
        valid_idx = train_mask & target.notna()
        X_tr = feat_df.loc[valid_idx, FEATURE_COLS].values
        y_tr = target[valid_idx].values
        if len(X_tr) == 0:
            raise RuntimeError(f"No training data for h={h}")
        trend_models[h] = train_ensemble(X_tr, y_tr)
        print(f"    h={h:4d}  ({len(X_tr)} rows)")

    print("  Calibrating sigma...")
    point_sigma = calibrate_sigma(point_models, feat_df, cutoff_dt, target_isn=None)
    trend_sigma = calibrate_sigma(trend_models, feat_df, cutoff_dt, target_isn=smooth_isn)
    for h in HORIZONS:
        print(f"    h={h:4d}  pt={point_sigma[h]:.1f}  tr={trend_sigma[h]:.1f}")

    anchor_rows = feat_df[feat_df["Date"] <= pred_base_dt]
    if anchor_rows.empty:
        raise RuntimeError("No anchor row at or before pred_base")
    X_pred = anchor_rows.iloc[[-1]][FEATURE_COLS].values

    sorted_horizons = sorted(HORIZONS)
    results = []

    for mo in range(1, FORECAST_HORIZON + 1):
        target_date = pred_base_date + pd.DateOffset(months=mo)

        h_lo = sorted_horizons[0]
        h_hi = sorted_horizons[-1]
        for h in sorted_horizons:
            if h <= mo:
                h_lo = h
            if h >= mo and h_hi >= h:
                h_hi = h

        if h_lo == h_hi or mo <= sorted_horizons[0]:
            h_use = min(HORIZONS, key=lambda h: abs(h - mo))
            pp, _ = predict_ensemble(point_models[h_use], X_pred)
            tp, _ = predict_ensemble(trend_models[h_use], X_pred)
            p_pred, t_pred = float(pp[0]), float(tp[0])
            p_sig,  t_sig  = point_sigma[h_use], trend_sigma[h_use]
        else:
            pp_lo, _ = predict_ensemble(point_models[h_lo], X_pred)
            pp_hi, _ = predict_ensemble(point_models[h_hi], X_pred)
            tp_lo, _ = predict_ensemble(trend_models[h_lo], X_pred)
            tp_hi, _ = predict_ensemble(trend_models[h_hi], X_pred)
            alpha  = (mo - h_lo) / (h_hi - h_lo)
            p_pred = float(pp_lo[0] * (1 - alpha) + pp_hi[0] * alpha)
            t_pred = float(tp_lo[0] * (1 - alpha) + tp_hi[0] * alpha)
            p_sig  = point_sigma[h_lo] * (1 - alpha) + point_sigma[h_hi] * alpha
            t_sig  = trend_sigma[h_lo] * (1 - alpha) + trend_sigma[h_hi] * alpha

        p_pred = max(0.0, p_pred)
        t_pred = max(0.0, t_pred)
        results.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_point": round(p_pred, 2),
            "predicted_trend": round(t_pred, 2),
            "lower_bound_point": round(max(0.0, p_pred - 1.28 * p_sig), 2),
            "upper_bound_point": round(p_pred + 1.28 * p_sig, 2),
            "lower_bound_trend": round(max(0.0, t_pred - 1.28 * t_sig), 2),
            "upper_bound_trend": round(t_pred + 1.28 * t_sig, 2),
        })

    return results


# --- Validation ----------------------------------------------------------
def run_validation(feat_df, last_data_month):
    cutoff_date  = last_data_month - pd.DateOffset(months=VALIDATION_WINDOW)
    pred_base    = last_data_month - pd.DateOffset(months=VALIDATION_WINDOW + 1)
    window_start = last_data_month - pd.DateOffset(months=VALIDATION_WINDOW - 1)

    print(f"\n[Validation]  cutoff={cutoff_date.date()}  "
          f"window={window_start.date()} -> {last_data_month.date()}")

    preds = run_mode(feat_df, cutoff_date, pred_base_date=pred_base)

    raw_map  = dict(zip(feat_df["Date"], feat_df["MonthlyISN"]))
    prov_map = dict(zip(feat_df["Date"], feat_df["provisional"]))

    records = []
    y_true = []
    y_pred_point, y_pred_trend = [], []

    for p in preds:
        d   = pd.Timestamp(p["date"])
        raw = raw_map.get(d)
        if raw is not None and not math.isnan(float(raw)):
            y_true.append(float(raw))
            y_pred_point.append(p["predicted_point"])
            y_pred_trend.append(p["predicted_trend"])
        records.append({
            "date": p["date"],
            "actual": round(float(raw), 2) if raw is not None and not math.isnan(float(raw)) else None,
            "predicted_point": p["predicted_point"],
            "predicted_trend": p["predicted_trend"],
            "lower_bound_point": p["lower_bound_point"],
            "upper_bound_point": p["upper_bound_point"],
            "lower_bound_trend": p["lower_bound_trend"],
            "upper_bound_trend": p["upper_bound_trend"],
            "provisional": bool(prov_map.get(d, False)),
        })

    r2_point   = r2_score(y_true, y_pred_point)                      if len(y_true) > 1 else float("nan")
    r2_trend   = r2_score(y_true, y_pred_trend)                      if len(y_true) > 1 else float("nan")
    mae_point  = mean_absolute_error(y_true, y_pred_point)           if y_true else float("nan")
    rmse_point = math.sqrt(mean_squared_error(y_true, y_pred_point)) if y_true else float("nan")
    mae_trend  = mean_absolute_error(y_true, y_pred_trend)           if y_true else float("nan")
    rmse_trend = math.sqrt(mean_squared_error(y_true, y_pred_trend)) if y_true else float("nan")

    print(f"  r2_point={r2_point:.4f}  mae_point={mae_point:.2f}  rmse_point={rmse_point:.2f}")
    print(f"  r2_trend={r2_trend:.4f}  mae_trend={mae_trend:.2f}  rmse_trend={rmse_trend:.2f}")
    print("  (all vs raw monthly ISN)")

    with open(os.path.join(OUT_DIR, "long_validation.json"), "w") as f:
        json.dump(records, f, indent=2)
    return r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend


# --- Forecast ------------------------------------------------------------
def run_forecast(feat_df, last_data_month):
    train_cutoff = last_data_month + pd.DateOffset(months=1)
    first_fc     = last_data_month + pd.DateOffset(months=1)
    print(f"\n[Forecast]  trained through={last_data_month.date()}  "
          f"first forecast month={first_fc.date()}")

    preds = run_mode(feat_df, train_cutoff, pred_base_date=last_data_month)

    records = [
        {
            "date": p["date"],
            "predicted_point": p["predicted_point"],
            "predicted_trend": p["predicted_trend"],
            "lower_bound_point": p["lower_bound_point"],
            "upper_bound_point": p["upper_bound_point"],
            "lower_bound_trend": p["lower_bound_trend"],
            "upper_bound_trend": p["upper_bound_trend"],
        }
        for p in preds
    ]

    with open(os.path.join(OUT_DIR, "long_forecast.json"), "w") as f:
        json.dump(records, f, indent=2)
    return preds


# --- Update shared metadata.json -----------------------------------------
def update_metadata(r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend):
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    meta["long"] = {
        "model_name":               "xgboost_long_v2",
        "data_granularity":         "monthly",
        "horizon_months":           FORECAST_HORIZON,
        "validation_window_months": VALIDATION_WINDOW,
        "horizons_trained":         HORIZONS,
        "ensemble_size":            len(ENSEMBLE_SEEDS),
        "r2_point":           round(r2_point,   6),
        "r2_trend":           round(r2_trend,   6),
        "mae_point":          round(mae_point,  4),
        "rmse_point":         round(rmse_point, 4),
        "mae_trend":          round(mae_trend,  4),
        "rmse_trend":         round(rmse_trend, 4),
        "uncertainty_method": "calibrated RMSE, 80pct PI (1.28 sigma)",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("  metadata.json updated (long key).")


# --- Main ----------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    feat_df, last_data_month = load_features()

    r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend = \
        run_validation(feat_df, last_data_month)
    run_forecast(feat_df, last_data_month)
    update_metadata(r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend)

    print("\n=== Long-term pipeline complete ===")
    print(f"  r2_point  : {r2_point:.4f}")
    print(f"  r2_trend  : {r2_trend:.4f}")
    print(f"  mae_point : {mae_point:.2f}  rmse_point : {rmse_point:.2f}")
    print(f"  mae_trend : {mae_trend:.2f}  rmse_trend : {rmse_trend:.2f}")
