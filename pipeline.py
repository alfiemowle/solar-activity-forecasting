"""
Solar Activity Short-Term Forecasting Pipeline (30-day horizon).
Loads pre-engineered features from data/features_daily.parquet.
Direct multi-step XGBoost ensemble, two outputs per day:
  predicted_point -- trained on raw daily ISN targets
  predicted_trend -- trained on 25-day trailing smoothed targets
"""

import os
import json
import math
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# --- Configuration ------------------------------------------------------------
FORECAST_HORIZON  = 30
VALIDATION_WINDOW = 30
ENSEMBLE_SEEDS    = [42, 7, 123, 1, 2, 3, 99]
HORIZONS          = [1, 2, 3, 4, 5, 6, 7, 10, 14, 18, 21, 25, 28, 30]
CAL_WINDOW        = 730   # trailing training rows for calibration
SMOOTH_WINDOW     = 25    # trailing rolling mean for trend targets

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DAILY_PARQUET = os.path.join(BASE_DIR, "data", "features_daily.parquet")
OUT_DIR       = os.path.join(BASE_DIR, "outputs")

# Features used by this pipeline (subset of the daily parquet columns)
FEATURE_COLS = (
    [f"lag_{l}" for l in [1, 2, 3, 5, 7, 14, 21, 27, 30, 54, 81]]
    + [f"roll_{stat}_{w}" for w in [7, 14, 27, 30, 60, 90, 180] for stat in ["mean", "std", "max"]]
    + [f"roll_min_{w}" for w in [30, 90]]
    + [f"ema_{span}" for span in [7, 30, 90]]
    + [f"mom_{l}" for l in [1, 3, 7, 14, 27]]
    + ["doy_sin", "doy_cos", "cycle_sin", "cycle_cos",
       "carrington_sin", "carrington_cos", "month_sin", "month_cos"]
)


# --- Load features ------------------------------------------------------------
def load_features() -> tuple[pd.DataFrame, datetime.date]:
    if not os.path.isfile(DAILY_PARQUET):
        raise FileNotFoundError(
            f"features_daily.parquet not found. Run data_prepare.py first."
        )
    feat_df = pd.read_parquet(DAILY_PARQUET)
    feat_df["Date"] = pd.to_datetime(feat_df["Date"])
    last_data_date = feat_df["Date"].max().date()
    print(f"Loaded features_daily.parquet")
    print(f"  {len(feat_df)} rows  "
          f"({feat_df['Date'].iloc[0].date()} -> {feat_df['Date'].iloc[-1].date()})")
    print(f"  LAST_DATA_DATE : {last_data_date}")
    return feat_df, last_data_date


# --- Train & Predict ----------------------------------------------------------
def train_horizon_ensemble(X_train, y_train, sample_weight=None):
    models = []
    for seed in ENSEMBLE_SEEDS:
        m = XGBRegressor(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.5,
            gamma=0.05,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        m.fit(X_train, y_train, sample_weight=sample_weight)
        models.append(m)
    return models


def predict_ensemble(models, X):
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


def calibrate_sigma(horizon_models: dict, feat_df: pd.DataFrame,
                    cutoff_dt: pd.Timestamp, target_isn=None) -> dict:
    """Per-horizon calibrated RMSE on the last CAL_WINDOW training rows.
    target_isn: if None uses raw DailyISN; otherwise the supplied series."""
    train_rows = feat_df[feat_df["Date"] < cutoff_dt]
    cal_rows   = train_rows.iloc[-CAL_WINDOW:] if len(train_rows) > CAL_WINDOW else train_rows

    base = target_isn if target_isn is not None else feat_df["DailyISN"]
    cal_sigma = {}
    for h, models in horizon_models.items():
        target = base.shift(-h)
        idx = cal_rows.index
        valid_mask = target.loc[idx].notna()
        valid_idx  = idx[valid_mask]
        if len(valid_idx) == 0:
            cal_sigma[h] = 30.0
            continue
        X_cal = feat_df.loc[valid_idx, FEATURE_COLS].values
        y_cal = target.loc[valid_idx].values
        mean_pred, _ = predict_ensemble(models, X_cal)
        rmse = float(np.sqrt(np.mean((y_cal - mean_pred) ** 2)))
        cal_sigma[h] = max(rmse, 5.0)
    return cal_sigma


def run_mode(feat_df: pd.DataFrame, cutoff_date: datetime.date,
             pred_base: datetime.date = None):
    """Train two ensembles per horizon on data strictly before cutoff_date.
    Predictions run from pred_base+1 to pred_base+FORECAST_HORIZON."""
    if pred_base is None:
        pred_base = cutoff_date

    cutoff_dt    = pd.Timestamp(cutoff_date)
    pred_base_dt = pd.Timestamp(pred_base)

    train_mask = feat_df["Date"] < cutoff_dt

    # Smoothed ISN for trend targets (trailing, no future values)
    smooth_isn = feat_df["DailyISN"].shift(1).rolling(SMOOTH_WINDOW, min_periods=1).mean()

    # Exponential sample weights (half-life = 90 days, emphasises recent cycles)
    days_to_cutoff = (cutoff_dt - feat_df["Date"]).dt.days.clip(lower=0)
    exp_weights    = np.exp(-np.log(2) * days_to_cutoff.values / 90.0)

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
        point_models[h] = train_horizon_ensemble(X_tr, y_tr, sample_weight=w_tr)
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
        trend_models[h] = train_horizon_ensemble(X_tr, y_tr, sample_weight=w_tr)
        print(f"    h={h:2d} ({len(X_tr)} rows)")

    print("  Calibrating uncertainty bands...")
    point_sigma = calibrate_sigma(point_models, feat_df, cutoff_dt, target_isn=None)
    trend_sigma = calibrate_sigma(trend_models, feat_df, cutoff_dt, target_isn=smooth_isn)
    for h in HORIZONS:
        print(f"    h={h:2d}  pt={point_sigma[h]:.1f}  tr={trend_sigma[h]:.1f}")

    anchor_rows = feat_df[feat_df["Date"] <= pred_base_dt]
    if anchor_rows.empty:
        raise RuntimeError("No feature rows at or before pred_base")
    X_pred = anchor_rows.iloc[[-1]][FEATURE_COLS].values

    results = []
    sorted_horizons = sorted(HORIZONS)

    for day_offset in range(1, FORECAST_HORIZON + 1):
        target_date = pred_base + datetime.timedelta(days=day_offset)

        h_lo = sorted_horizons[0]
        h_hi = sorted_horizons[-1]
        for h in sorted_horizons:
            if h <= day_offset:
                h_lo = h
            if h >= day_offset and h_hi >= h:
                h_hi = h

        if h_lo == h_hi or day_offset <= sorted_horizons[0]:
            h_use = min(HORIZONS, key=lambda h: abs(h - day_offset))
            pp, _ = predict_ensemble(point_models[h_use], X_pred)
            tp, _ = predict_ensemble(trend_models[h_use], X_pred)
            p_pred, t_pred = float(pp[0]), float(tp[0])
            p_sig,  t_sig  = point_sigma[h_use], trend_sigma[h_use]
        else:
            pp_lo, _ = predict_ensemble(point_models[h_lo], X_pred)
            pp_hi, _ = predict_ensemble(point_models[h_hi], X_pred)
            tp_lo, _ = predict_ensemble(trend_models[h_lo], X_pred)
            tp_hi, _ = predict_ensemble(trend_models[h_hi], X_pred)
            alpha  = (day_offset - h_lo) / (h_hi - h_lo)
            p_pred = float(pp_lo[0] * (1 - alpha) + pp_hi[0] * alpha)
            t_pred = float(tp_lo[0] * (1 - alpha) + tp_hi[0] * alpha)
            p_sig  = point_sigma[h_lo] * (1 - alpha) + point_sigma[h_hi] * alpha
            t_sig  = trend_sigma[h_lo] * (1 - alpha) + trend_sigma[h_hi] * alpha

        p_pred = max(0.0, p_pred)
        t_pred = max(0.0, t_pred)
        results.append({
            "date": target_date.isoformat(),
            "predicted_point": round(p_pred, 2),
            "predicted_trend": round(t_pred, 2),
            "lower_bound_point": round(max(0.0, p_pred - 1.28 * p_sig), 2),
            "upper_bound_point": round(p_pred + 1.28 * p_sig, 2),
            "lower_bound_trend": round(max(0.0, t_pred - 1.28 * t_sig), 2),
            "upper_bound_trend": round(t_pred + 1.28 * t_sig, 2),
        })

    return results


# --- Validation ---------------------------------------------------------------
def run_validation(feat_df: pd.DataFrame, last_data_date: datetime.date):
    cutoff      = last_data_date - datetime.timedelta(days=VALIDATION_WINDOW)
    window_start = cutoff + datetime.timedelta(days=1)
    print(f"\n[Mode A -- Validation]  training cutoff={cutoff}  "
          f"window={window_start} -> {last_data_date}")

    preds = run_mode(feat_df, cutoff)

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
            "date": p["date"],
            "actual": round(float(actual), 2) if actual is not None and not math.isnan(float(actual)) else None,
            "predicted_point": p["predicted_point"],
            "predicted_trend": p["predicted_trend"],
            "lower_bound_point": p["lower_bound_point"],
            "upper_bound_point": p["upper_bound_point"],
            "lower_bound_trend": p["lower_bound_trend"],
            "upper_bound_trend": p["upper_bound_trend"],
            "provisional": bool(prov_map.get(d, False)),
        })

    r2_point   = r2_score(y_act, y_pt)                            if len(y_act) > 1 else float("nan")
    r2_trend   = r2_score(y_act, y_tr)                            if len(y_act) > 1 else float("nan")
    mae_point  = mean_absolute_error(y_act, y_pt)                 if y_act else float("nan")
    mae_trend  = mean_absolute_error(y_act, y_tr)                 if y_act else float("nan")
    rmse_point = math.sqrt(mean_squared_error(y_act, y_pt))       if y_act else float("nan")
    rmse_trend = math.sqrt(mean_squared_error(y_act, y_tr))       if y_act else float("nan")

    print(f"  r2_point={r2_point:.4f}  r2_trend={r2_trend:.4f}  "
          f"mae_pt={mae_point:.2f}  mae_tr={mae_trend:.2f}  (vs raw daily ISN)")

    with open(os.path.join(OUT_DIR, "validation.json"), "w") as f:
        json.dump(records, f, indent=2)

    return r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend


# --- Live Forecast ------------------------------------------------------------
def run_forecast(feat_df: pd.DataFrame, last_data_date: datetime.date):
    train_cutoff = last_data_date + datetime.timedelta(days=1)
    first_fc     = last_data_date + datetime.timedelta(days=1)
    print(f"\n[Mode B -- Live Forecast]  trained through={last_data_date}  "
          f"first forecast day={first_fc}")

    preds = run_mode(feat_df, train_cutoff, pred_base=last_data_date)

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

    with open(os.path.join(OUT_DIR, "forecast.json"), "w") as f:
        json.dump(records, f, indent=2)

    return preds


# --- Metadata -----------------------------------------------------------------
def save_metadata(feat_df, r2_point, r2_trend, mae_point, rmse_point,
                  mae_trend, rmse_trend, last_data_date: datetime.date):
    # Preserve any existing medium/long keys written by other pipelines
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    meta.update({
        "date_generated":        datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_start":            str(feat_df["Date"].iloc[0].date()),
        "data_end":              str(feat_df["Date"].iloc[-1].date()),
        "last_data_date":        str(last_data_date),
        "model_name":            "xgboost_multistep_v2",
        "horizons_trained":      HORIZONS,
        "ensemble_size":         len(ENSEMBLE_SEEDS),
        "forecast_horizon_days": FORECAST_HORIZON,
        "validation_window_days": VALIDATION_WINDOW,
        "r2_point":              round(r2_point,   6),
        "r2_trend":              round(r2_trend,   6),
        "mae_point":             round(mae_point,  4),
        "rmse_point":            round(rmse_point, 4),
        "mae_trend":             round(mae_trend,  4),
        "rmse_trend":            round(rmse_trend, 4),
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
    })

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
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
        os.path.join(BASE_DIR, "website", "index.html"),
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

    feat_df, last_data_date = load_features()
    print(f"\n  >> LAST_DATA_DATE = {last_data_date} <<")

    r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend = \
        run_validation(feat_df, last_data_date)

    run_forecast(feat_df, last_data_date)

    save_metadata(feat_df, r2_point, r2_trend, mae_point, rmse_point,
                  mae_trend, rmse_trend, last_data_date)

    verify()
