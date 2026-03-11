"""
Solar Activity Short-Term Forecasting Pipeline (30-day horizon).
Loads pre-engineered features from data/features_daily.parquet.
Direct multi-step XGBoost ensemble, two outputs per day:
  predicted_point -- trained on raw daily ISN targets
  predicted_trend -- trained on 25-day trailing smoothed targets

Models are trained once on the full dataset (cutoff = last_data_date + 1 day)
and reused for both the validation back-check and the live forecast.
"""

import os
import json
import math
import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from config import ENSEMBLE_SEEDS, SHORT as CFG
from utils import (
    predict_ensemble, calibrate_sigma, load_parquet,
    build_predictions, update_metadata,
)

# --- Paths --------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DAILY_PARQUET = os.path.join(BASE_DIR, "data", "features_daily.parquet")
OUT_DIR       = os.path.join(BASE_DIR, "outputs")
META_PATH     = os.path.join(OUT_DIR, "metadata.json")

# --- Unpack config ------------------------------------------------------------
FORECAST_HORIZON  = CFG["FORECAST_HORIZON"]
VALIDATION_WINDOW = CFG["VALIDATION_WINDOW"]
CAL_WINDOW        = CFG["CAL_WINDOW"]
SMOOTH_WINDOW     = CFG["SMOOTH_WINDOW"]
WEIGHT_HALFLIFE   = CFG["WEIGHT_HALFLIFE"]
HORIZONS          = CFG["HORIZONS"]
XGB_PARAMS        = CFG["XGB_PARAMS"]

# --- Feature columns ----------------------------------------------------------
FEATURE_COLS = (
    [f"lag_{l}" for l in [1, 2, 3, 5, 7, 14, 21, 27, 30, 54, 81]]
    + [f"roll_{stat}_{w}" for w in [7, 14, 27, 30, 60, 90, 180]
       for stat in ["mean", "std", "max"]]
    + [f"roll_min_{w}" for w in [30, 90]]
    + [f"ema_{span}" for span in [7, 30, 90]]
    + [f"mom_{l}" for l in [1, 3, 7, 14, 27]]
    + ["doy_sin", "doy_cos", "cycle_sin", "cycle_cos",
       "carrington_sin", "carrington_cos", "month_sin", "month_cos"]
)


# --- Training -----------------------------------------------------------------

def _train_ensemble(X_train, y_train, sample_weight=None):
    models = []
    for seed in ENSEMBLE_SEEDS:
        m = XGBRegressor(**XGB_PARAMS, random_state=seed, n_jobs=-1, verbosity=0)
        m.fit(X_train, y_train, sample_weight=sample_weight)
        models.append(m)
    return models


def train_models(feat_df, cutoff_dt):
    """Train point + trend ensembles on data strictly before cutoff_dt.

    Returns (smooth_isn, point_models, trend_models, point_sigma, trend_sigma).
    """
    train_mask = feat_df["Date"] < cutoff_dt

    smooth_isn = feat_df["DailyISN"].shift(1).rolling(SMOOTH_WINDOW, min_periods=1).mean()

    days_to_cutoff = (cutoff_dt - feat_df["Date"]).dt.days.clip(lower=0)
    exp_weights    = np.exp(-np.log(2) * days_to_cutoff.values / WEIGHT_HALFLIFE)

    print("  Training point models...")
    point_models = {}
    for h in HORIZONS:
        target    = feat_df["DailyISN"].shift(-h)
        valid_idx = train_mask & target.notna()
        X_tr = feat_df.loc[valid_idx, FEATURE_COLS].values
        y_tr = target[valid_idx].values
        w_tr = exp_weights[valid_idx.values]
        if len(X_tr) == 0:
            raise RuntimeError(f"No training data for h={h}")
        point_models[h] = _train_ensemble(X_tr, y_tr, sample_weight=w_tr)
        print(f"    h={h:2d} ({len(X_tr)} rows)")

    print("  Training trend models...")
    trend_models = {}
    for h in HORIZONS:
        target    = smooth_isn.shift(-h)
        valid_idx = train_mask & target.notna()
        X_tr = feat_df.loc[valid_idx, FEATURE_COLS].values
        y_tr = target[valid_idx].values
        w_tr = exp_weights[valid_idx.values]
        if len(X_tr) == 0:
            raise RuntimeError(f"No training data for h={h}")
        trend_models[h] = _train_ensemble(X_tr, y_tr, sample_weight=w_tr)
        print(f"    h={h:2d} ({len(X_tr)} rows)")

    print("  Calibrating uncertainty bands...")
    point_sigma = calibrate_sigma(
        point_models, feat_df, cutoff_dt, FEATURE_COLS, CAL_WINDOW,
        isn_col="DailyISN",
    )
    trend_sigma = calibrate_sigma(
        trend_models, feat_df, cutoff_dt, FEATURE_COLS, CAL_WINDOW,
        target_isn=smooth_isn,
    )
    for h in HORIZONS:
        print(f"    h={h:2d}  pt={point_sigma[h]:.1f}  tr={trend_sigma[h]:.1f}")

    return smooth_isn, point_models, trend_models, point_sigma, trend_sigma


# --- Validation ---------------------------------------------------------------

def run_validation(feat_df, models_tuple, last_data_date):
    """Back-check using the full-dataset models anchored to val_anchor date."""
    _, point_models, trend_models, point_sigma, trend_sigma = models_tuple

    val_anchor    = last_data_date - datetime.timedelta(days=VALIDATION_WINDOW)
    val_anchor_dt = pd.Timestamp(val_anchor)
    window_start  = val_anchor + datetime.timedelta(days=1)
    print(f"\n[Validation]  anchor={val_anchor}  "
          f"window={window_start} -> {last_data_date}")

    anchor_rows = feat_df[feat_df["Date"] <= val_anchor_dt]
    if anchor_rows.empty:
        raise RuntimeError("No anchor row for validation")
    X_val = anchor_rows.iloc[[-1]][FEATURE_COLS].values

    preds = build_predictions(
        point_models, trend_models, point_sigma, trend_sigma,
        HORIZONS, X_val, val_anchor, VALIDATION_WINDOW,
    )

    actual_map = dict(zip(feat_df["Date"].dt.date, feat_df["DailyISN"]))
    prov_map   = dict(zip(feat_df["Date"].dt.date, feat_df["provisional"]))

    records = []
    y_act, y_pt, y_tr = [], [], []
    for p in preds:
        d      = datetime.date.fromisoformat(p["date"])
        actual = actual_map.get(d)
        if actual is not None and not math.isnan(float(actual)):
            y_act.append(float(actual))
            y_pt.append(p["predicted_point"])
            y_tr.append(p["predicted_trend"])
        records.append({
            "date":               p["date"],
            "actual":             round(float(actual), 2)
                                  if actual is not None and not math.isnan(float(actual))
                                  else None,
            "predicted_point":    p["predicted_point"],
            "predicted_trend":    p["predicted_trend"],
            "lower_bound_point":  p["lower_bound_point"],
            "upper_bound_point":  p["upper_bound_point"],
            "lower_bound_trend":  p["lower_bound_trend"],
            "upper_bound_trend":  p["upper_bound_trend"],
            "provisional":        bool(prov_map.get(d, False)),
        })

    r2_point   = r2_score(y_act, y_pt)                      if len(y_act) > 1 else float("nan")
    r2_trend   = r2_score(y_act, y_tr)                      if len(y_act) > 1 else float("nan")
    mae_point  = mean_absolute_error(y_act, y_pt)           if y_act else float("nan")
    mae_trend  = mean_absolute_error(y_act, y_tr)           if y_act else float("nan")
    rmse_point = math.sqrt(mean_squared_error(y_act, y_pt)) if y_act else float("nan")
    rmse_trend = math.sqrt(mean_squared_error(y_act, y_tr)) if y_act else float("nan")

    print(f"  r2_point={r2_point:.4f}  r2_trend={r2_trend:.4f}  "
          f"mae_pt={mae_point:.2f}  mae_tr={mae_trend:.2f}  (vs raw daily ISN)")

    with open(os.path.join(OUT_DIR, "validation.json"), "w") as f:
        json.dump(records, f, indent=2)

    return r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend


# --- Live Forecast ------------------------------------------------------------

def run_forecast(feat_df, models_tuple, last_data_date):
    """Generate the live forecast anchored to last_data_date."""
    _, point_models, trend_models, point_sigma, trend_sigma = models_tuple

    first_fc = last_data_date + datetime.timedelta(days=1)
    print(f"\n[Forecast]  trained through={last_data_date}  first day={first_fc}")

    last_dt     = pd.Timestamp(last_data_date)
    anchor_rows = feat_df[feat_df["Date"] <= last_dt]
    X_fc        = anchor_rows.iloc[[-1]][FEATURE_COLS].values

    preds = build_predictions(
        point_models, trend_models, point_sigma, trend_sigma,
        HORIZONS, X_fc, last_data_date, FORECAST_HORIZON,
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

    with open(os.path.join(OUT_DIR, "forecast.json"), "w") as f:
        json.dump(records, f, indent=2)

    return preds


# --- Metadata -----------------------------------------------------------------

def save_metadata(feat_df, r2_point, r2_trend, mae_point, rmse_point,
                  mae_trend, rmse_trend, last_data_date):
    data = {
        "date_generated":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_start":             str(feat_df["Date"].iloc[0].date()),
        "data_end":               str(feat_df["Date"].iloc[-1].date()),
        "last_data_date":         str(last_data_date),
        "model_name":             "xgboost_multistep_v2",
        "horizons_trained":       HORIZONS,
        "ensemble_size":          len(ENSEMBLE_SEEDS),
        "forecast_horizon_days":  FORECAST_HORIZON,
        "validation_window_days": VALIDATION_WINDOW,
        "r2_point":               round(r2_point,   6),
        "r2_trend":               round(r2_trend,   6),
        "mae_point":              round(mae_point,  4),
        "rmse_point":             round(rmse_point, 4),
        "mae_trend":              round(mae_trend,  4),
        "rmse_trend":             round(rmse_trend, 4),
        "leakage_check": (
            "direct multi-step -- each horizon model trained independently "
            "on data before cutoff, no autoregressive chaining"
        ),
        "uncertainty_method": (
            "calibrated RMSE on trailing CAL_WINDOW training rows, "
            "80pct prediction interval (1.28 sigma)"
        ),
        "note": (
            "Forecast anchored to last available SILSO data date, not system date. "
            "SILSO daily data typically lags 3-7 days behind current date."
        ),
        "config": {
            "ensemble_seeds":  ENSEMBLE_SEEDS,
            "cal_window":      CAL_WINDOW,
            "smooth_window":   SMOOTH_WINDOW,
            "weight_halflife": WEIGHT_HALFLIFE,
            "xgb_params":      XGB_PARAMS,
        },
    }
    update_metadata(META_PATH, key=None, data=data)
    print("\nMetadata saved.")


# --- Verify -------------------------------------------------------------------

def verify():
    print("\n--- Verification -----------------------------------")
    required = [
        os.path.join(BASE_DIR, "data", "silso_daily.csv"),
        os.path.join(BASE_DIR, "data", "features_daily.parquet"),
        os.path.join(OUT_DIR, "validation.json"),
        os.path.join(OUT_DIR, "forecast.json"),
        os.path.join(OUT_DIR, "metadata.json"),
    ]
    for path in required:
        exists = os.path.isfile(path)
        print(f"  {'OK' if exists else 'MISSING'}  {os.path.relpath(path, BASE_DIR)}")

    with open(os.path.join(OUT_DIR, "validation.json")) as f:
        val = json.load(f)
    with open(os.path.join(OUT_DIR, "forecast.json")) as f:
        fc  = json.load(f)
    with open(os.path.join(OUT_DIR, "metadata.json")) as f:
        meta = json.load(f)

    print(f"\n  validation.json  -- {len(val)} records")
    print(f"    first: {val[0]}")
    print(f"    last : {val[-1]}")
    print(f"\n  forecast.json    -- {len(fc)} records")
    print(f"    first: {fc[0]}")
    print(f"    last : {fc[-1]}")

    print(f"\n  Data range      : {meta['data_start']}  ->  {meta['data_end']}")
    print(f"  LAST_DATA_DATE  : {meta['last_data_date']}")
    print(f"  r2_point  : {meta['r2_point']}")
    print(f"  r2_trend  : {meta['r2_trend']}")
    print(f"  mae_point : {meta['mae_point']}  rmse_point : {meta['rmse_point']}")
    print(f"  mae_trend : {meta['mae_trend']}  rmse_trend : {meta['rmse_trend']}")
    first_fc = fc[0]
    print(f"\n  First forecast day : {first_fc['date']}  "
          f"pt={first_fc['predicted_point']}  tr={first_fc['predicted_trend']}  "
          f"pt_bounds=[{first_fc['lower_bound_point']}, {first_fc['upper_bound_point']}]  "
          f"tr_bounds=[{first_fc['lower_bound_trend']}, {first_fc['upper_bound_trend']}]")


# --- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    feat_df, last_dt = load_parquet(DAILY_PARQUET)
    last_data_date   = last_dt.date()
    print(f"\n  >> LAST_DATA_DATE = {last_data_date} <<")

    # Train once on the full dataset
    train_cutoff_dt = pd.Timestamp(last_data_date) + pd.Timedelta(days=1)
    print(f"\n[Training]  cutoff={train_cutoff_dt.date()}")
    models_tuple = train_models(feat_df, train_cutoff_dt)

    r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend = \
        run_validation(feat_df, models_tuple, last_data_date)

    run_forecast(feat_df, models_tuple, last_data_date)

    save_metadata(feat_df, r2_point, r2_trend, mae_point, rmse_point,
                  mae_trend, rmse_trend, last_data_date)

    verify()
