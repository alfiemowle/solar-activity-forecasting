"""
Solar Activity Long-Term Forecasting Pipeline (144-month / 12-year horizon).
Loads pre-engineered monthly features from data/features_monthly.parquet.
Direct multi-step XGBoost ensemble, two outputs per month:
  predicted_point -- trained on raw monthly ISN targets
  predicted_trend -- trained on 13-month trailing smoothed targets (SmoothedISN)

Models are trained once on the full dataset (cutoff = last_data_month + 1 month)
and reused for both the validation back-check and the live forecast.
"""

import os
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from config import ENSEMBLE_SEEDS, LONG as CFG
from utils import (
    predict_ensemble, calibrate_sigma, load_parquet,
    build_predictions, update_metadata,
)

# --- Paths -------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MONTHLY_PARQUET = os.path.join(BASE_DIR, "data", "features_monthly.parquet")
OUT_DIR         = os.path.join(BASE_DIR, "outputs")
META_PATH       = os.path.join(OUT_DIR, "metadata.json")

# --- Unpack config -----------------------------------------------------------
FORECAST_HORIZON  = CFG["FORECAST_HORIZON"]
VALIDATION_WINDOW = CFG["VALIDATION_WINDOW"]
CAL_WINDOW        = CFG["CAL_WINDOW"]
HORIZONS          = CFG["HORIZONS"]
XGB_PARAMS        = CFG["XGB_PARAMS"]

# --- Feature columns ---------------------------------------------------------
FEATURE_COLS = (
    [f"lag_{l}" for l in [1, 2, 3, 6, 12, 24, 48, 120, 132]]
    + [f"roll_{s}_{w}" for w in [12, 24, 60, 120] for s in ["mean", "std", "max"]]
    + [f"roll_min_{w}" for w in [60, 120]]
    + [f"ema_{sp}" for sp in [12, 24, 60]]
    + [f"mom_{l}" for l in [12, 24, 48, 120, 132]]
    + ["cycle_sin", "cycle_cos", "gleissberg_sin", "gleissberg_cos"]
)


# --- Horizon-scaled sigma floor (long pipeline only) -------------------------

def _sigma_floor(h):
    """Minimum sigma scales linearly: 5.0 for h<=24, up to 40.0 at h=144."""
    if h <= 24:
        return 5.0
    return 25.0 + (h - 24) / (144 - 24) * (40.0 - 25.0)


# --- Training ----------------------------------------------------------------

def _train_ensemble(X_train, y_train):
    models = []
    for seed in ENSEMBLE_SEEDS:
        m = XGBRegressor(**XGB_PARAMS, random_state=seed, n_jobs=-1, verbosity=0)
        m.fit(X_train, y_train)
        models.append(m)
    return models


def train_models(feat_df, cutoff_dt):
    """Train point + trend ensembles on data strictly before cutoff_dt.

    Returns (smooth_isn, point_models, trend_models, point_sigma, trend_sigma).
    """
    smooth_isn = feat_df["SmoothedISN"].copy()
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
        point_models[h] = _train_ensemble(X_tr, y_tr)
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
        trend_models[h] = _train_ensemble(X_tr, y_tr)
        print(f"    h={h:4d}  ({len(X_tr)} rows)")

    print("  Calibrating sigma...")
    point_sigma = calibrate_sigma(
        point_models, feat_df, cutoff_dt, FEATURE_COLS, CAL_WINDOW,
        isn_col="MonthlyISN", sigma_floor=_sigma_floor, default_sigma=25.0,
    )
    trend_sigma = calibrate_sigma(
        trend_models, feat_df, cutoff_dt, FEATURE_COLS, CAL_WINDOW,
        target_isn=smooth_isn, sigma_floor=_sigma_floor, default_sigma=25.0,
    )
    for h in HORIZONS:
        print(f"    h={h:4d}  pt={point_sigma[h]:.1f}  tr={trend_sigma[h]:.1f}")

    return smooth_isn, point_models, trend_models, point_sigma, trend_sigma


# --- Validation --------------------------------------------------------------

def run_validation(feat_df, models_tuple, last_data_month):
    """Back-check using the full-dataset models anchored VALIDATION_WINDOW+1 months back."""
    _, point_models, trend_models, point_sigma, trend_sigma = models_tuple

    # pred_base is one month before the first validation target
    pred_base    = last_data_month - pd.DateOffset(months=VALIDATION_WINDOW + 1)
    window_start = last_data_month - pd.DateOffset(months=VALIDATION_WINDOW - 1)

    print(f"\n[Validation]  anchor={pred_base.date()}  "
          f"window={window_start.date()} -> {last_data_month.date()}")

    anchor_rows = feat_df[feat_df["Date"] <= pd.Timestamp(pred_base)]
    if anchor_rows.empty:
        raise RuntimeError("No anchor row for validation")
    X_val = anchor_rows.iloc[[-1]][FEATURE_COLS].values

    preds = build_predictions(
        point_models, trend_models, point_sigma, trend_sigma,
        HORIZONS, X_val, pred_base, VALIDATION_WINDOW, use_months=True,
    )

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
            "date":               p["date"],
            "actual":             round(float(raw), 2)
                                  if raw is not None and not math.isnan(float(raw))
                                  else None,
            "predicted_point":    p["predicted_point"],
            "predicted_trend":    p["predicted_trend"],
            "lower_bound_point":  p["lower_bound_point"],
            "upper_bound_point":  p["upper_bound_point"],
            "lower_bound_trend":  p["lower_bound_trend"],
            "upper_bound_trend":  p["upper_bound_trend"],
            "provisional":        bool(prov_map.get(d, False)),
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


# --- Forecast ----------------------------------------------------------------

def run_forecast(feat_df, models_tuple, last_data_month):
    """Generate the live 144-month forecast anchored to last_data_month."""
    _, point_models, trend_models, point_sigma, trend_sigma = models_tuple

    first_fc = last_data_month + pd.DateOffset(months=1)
    print(f"\n[Forecast]  trained through={last_data_month.date()}  "
          f"first forecast month={first_fc.date()}")

    anchor_rows = feat_df[feat_df["Date"] <= pd.Timestamp(last_data_month)]
    if anchor_rows.empty:
        raise RuntimeError("No anchor row for forecast")
    X_fc = anchor_rows.iloc[[-1]][FEATURE_COLS].values

    preds = build_predictions(
        point_models, trend_models, point_sigma, trend_sigma,
        HORIZONS, X_fc, last_data_month, FORECAST_HORIZON, use_months=True,
    )

    records = [
        {
            "date":              p["date"],
            "predicted_point":   p["predicted_point"],
            "predicted_trend":   p["predicted_trend"],
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


# --- Metadata ----------------------------------------------------------------

def update_meta(r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend):
    data = {
        "model_name":               "xgboost_long_v2",
        "data_granularity":         "monthly",
        "horizon_months":           FORECAST_HORIZON,
        "validation_window_months": VALIDATION_WINDOW,
        "horizons_trained":         HORIZONS,
        "ensemble_size":            len(ENSEMBLE_SEEDS),
        "r2_point":                 round(r2_point,   6),
        "r2_trend":                 round(r2_trend,   6),
        "mae_point":                round(mae_point,  4),
        "rmse_point":               round(rmse_point, 4),
        "mae_trend":                round(mae_trend,  4),
        "rmse_trend":               round(rmse_trend, 4),
        "uncertainty_method":       "calibrated RMSE, 80pct PI (1.28 sigma)",
        "config": {
            "ensemble_seeds": ENSEMBLE_SEEDS,
            "cal_window":     CAL_WINDOW,
            "xgb_params":     XGB_PARAMS,
        },
    }
    update_metadata(META_PATH, key="long", data=data)
    print("  metadata.json updated (long key).")


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    feat_df, last_data_month = load_parquet(MONTHLY_PARQUET)
    print(f"  Last month : {last_data_month.date()}")

    # Train once on the full dataset
    train_cutoff_dt = last_data_month + pd.DateOffset(months=1)
    print(f"\n[Training]  cutoff={train_cutoff_dt.date()}")
    models_tuple = train_models(feat_df, train_cutoff_dt)

    r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend = \
        run_validation(feat_df, models_tuple, last_data_month)
    run_forecast(feat_df, models_tuple, last_data_month)
    update_meta(r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend)

    print("\n=== Long-term pipeline complete ===")
    print(f"  r2_point  : {r2_point:.4f}")
    print(f"  r2_trend  : {r2_trend:.4f}")
    print(f"  mae_point : {mae_point:.2f}  rmse_point : {rmse_point:.2f}")
    print(f"  mae_trend : {mae_trend:.2f}  rmse_trend : {rmse_trend:.2f}")
