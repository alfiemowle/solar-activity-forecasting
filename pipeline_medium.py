"""
Solar Activity Medium-Term Forecasting Pipeline (365-day horizon).
Loads pre-engineered features from data/features_daily.parquet.
Direct multi-step XGBoost ensemble, two outputs per day:
  predicted_point -- trained on raw daily ISN targets
  predicted_trend -- trained on 25-day trailing smoothed targets
r2_trend is evaluated against 25-day smoothed actuals.
"""
import os
import json
import math
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# --- Configuration -------------------------------------------------------
FORECAST_HORIZON  = 365
VALIDATION_WINDOW = 365
ENSEMBLE_SEEDS    = [42, 7, 123, 1, 2, 3, 99]
HORIZONS          = [1, 7, 14, 30, 60, 90, 120, 180, 270, 365]
CAL_WINDOW        = 730
SMOOTH_WINDOW     = 25
TRAIN_START_DATE  = "2010-01-01"  # limit to recent cycles for efficiency

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DAILY_PARQUET = os.path.join(BASE_DIR, "data", "features_daily.parquet")
OUT_DIR       = os.path.join(BASE_DIR, "outputs")

# Features used by this pipeline (full medium set from parquet)
FEATURE_COLS = (
    [f"lag_{l}" for l in [1, 2, 3, 5, 7, 14, 21, 27, 30, 54, 81, 180, 270, 365]]
    + [f"roll_{s}_{w}" for w in [7, 14, 27, 30, 60, 90, 180, 365] for s in ["mean", "std", "max"]]
    + [f"roll_min_{w}" for w in [30, 90]]
    + [f"ema_{sp}" for sp in [7, 30, 90]]
    + [f"mom_{l}" for l in [1, 3, 7, 14, 27]]
    + ["doy_sin", "doy_cos", "cycle_sin", "cycle_cos",
       "carrington_sin", "carrington_cos", "month_sin", "month_cos"]
)


# --- Load features -------------------------------------------------------
def load_features() -> tuple[pd.DataFrame, datetime.date]:
    if not os.path.isfile(DAILY_PARQUET):
        raise FileNotFoundError(
            "features_daily.parquet not found. Run data_prepare.py first."
        )
    feat_df = pd.read_parquet(DAILY_PARQUET)
    feat_df["Date"] = pd.to_datetime(feat_df["Date"])
    last_data_date = feat_df["Date"].max().date()
    print("Loaded features_daily.parquet")
    print(f"  {len(feat_df)} rows  "
          f"({feat_df['Date'].iloc[0].date()} -> {feat_df['Date'].iloc[-1].date()})")
    print(f"  Last data date : {last_data_date}")
    return feat_df, last_data_date


# --- Ensemble training ---------------------------------------------------
def train_ensemble(X_train, y_train, sample_weight=None):
    models = []
    for seed in ENSEMBLE_SEEDS:
        m = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
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


# --- Calibrate uncertainty -----------------------------------------------
def calibrate_sigma(horizon_models, feat_df, cutoff_dt, target_isn=None):
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


# --- Run mode ------------------------------------------------------------
def run_mode(feat_df, cutoff_date, pred_base_date=None):
    if pred_base_date is None:
        pred_base_date = cutoff_date

    cutoff_dt    = pd.Timestamp(cutoff_date)
    pred_base_dt = pd.Timestamp(pred_base_date)

    train_start_dt = pd.Timestamp(TRAIN_START_DATE)
    train_mask = (feat_df["Date"] < cutoff_dt) & (feat_df["Date"] >= train_start_dt)

    smooth_isn = feat_df["DailyISN"].shift(1).rolling(SMOOTH_WINDOW, min_periods=1).mean()

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
        point_models[h] = train_ensemble(X_tr, y_tr, sample_weight=w_tr)
        print(f"    h={h:4d} ({len(X_tr)} rows)")

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
        trend_models[h] = train_ensemble(X_tr, y_tr, sample_weight=w_tr)
        print(f"    h={h:4d} ({len(X_tr)} rows)")

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

    for day_offset in range(1, FORECAST_HORIZON + 1):
        target_date = pred_base_date + datetime.timedelta(days=day_offset)

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


# --- Validation ----------------------------------------------------------
def run_validation(feat_df, last_data_date):
    cutoff_date  = last_data_date - datetime.timedelta(days=VALIDATION_WINDOW)
    window_start = cutoff_date + datetime.timedelta(days=1)
    print(f"\n[Validation]  cutoff={cutoff_date}  "
          f"window={window_start} -> {last_data_date}")

    preds = run_mode(feat_df, cutoff_date, pred_base_date=cutoff_date)

    actual_map = dict(zip(feat_df["Date"].dt.date, feat_df["DailyISN"]))
    prov_map   = dict(zip(feat_df["Date"].dt.date, feat_df["provisional"]))

    # Compute smoothed actuals over the validation window (25-day trailing rolling)
    val_dates   = [datetime.date.fromisoformat(p["date"]) for p in preds]
    raw_actuals = [actual_map.get(d) for d in val_dates]
    raw_series  = pd.Series(
        [float(v) if v is not None and not math.isnan(float(v)) else float("nan")
         for v in raw_actuals]
    )
    smoothed_series = raw_series.rolling(SMOOTH_WINDOW, min_periods=5).mean().round(2)

    records = []
    y_act, y_act_sm, y_pt, y_tr = [], [], [], []
    for i, p in enumerate(preds):
        d      = val_dates[i]
        actual = raw_actuals[i]
        act_sm = smoothed_series.iloc[i]
        has_raw = actual is not None and not math.isnan(float(actual))
        has_sm  = not math.isnan(float(act_sm)) if act_sm is not None else False
        if has_raw:
            y_act.append(float(actual))
            y_pt.append(p["predicted_point"])
        if has_raw and has_sm:
            y_act_sm.append(float(act_sm))
            y_tr.append(p["predicted_trend"])
        records.append({
            "date": p["date"],
            "actual": round(float(actual), 2) if has_raw else None,
            "actual_smoothed": round(float(act_sm), 2) if has_sm else None,
            "predicted_point": p["predicted_point"],
            "predicted_trend": p["predicted_trend"],
            "lower_bound_point": p["lower_bound_point"],
            "upper_bound_point": p["upper_bound_point"],
            "lower_bound_trend": p["lower_bound_trend"],
            "upper_bound_trend": p["upper_bound_trend"],
            "provisional": bool(prov_map.get(d, False)),
        })

    r2_point  = r2_score(y_act,    y_pt) if len(y_act)    > 1 else float("nan")
    r2_trend  = r2_score(y_act_sm, y_tr) if len(y_act_sm) > 1 else float("nan")
    mae_point  = mean_absolute_error(y_act,    y_pt) if y_act    else float("nan")
    mae_trend  = mean_absolute_error(y_act_sm, y_tr) if y_act_sm else float("nan")
    rmse_point = math.sqrt(mean_squared_error(y_act,    y_pt)) if y_act    else float("nan")
    rmse_trend = math.sqrt(mean_squared_error(y_act_sm, y_tr)) if y_act_sm else float("nan")

    print(f"  r2_point={r2_point:.4f}  (vs raw daily ISN)")
    print(f"  r2_trend={r2_trend:.4f}  mae_tr={mae_trend:.2f}  (vs smoothed ISN)")

    # Predicted trend vs actual_smoothed comparison table
    print(f"\n  {'Date':>12}  {'Actual':>7}  {'Smoothed':>10}  {'Pred_trend':>12}  {'Error':>8}")
    print("  " + "-" * 56)
    step = max(1, len(records) // 24)
    for rec in records[::step]:
        sm  = f"{rec['actual_smoothed']:.1f}" if rec["actual_smoothed"] is not None else "   n/a"
        pt  = f"{rec['actual']:.1f}"          if rec["actual"]          is not None else "   n/a"
        err = (f"{rec['predicted_trend'] - rec['actual_smoothed']:+.1f}"
               if rec["actual_smoothed"] is not None else "   n/a")
        print(f"  {rec['date']:>12}  {pt:>7}  {sm:>10}  {rec['predicted_trend']:>12.1f}  {err:>8}")

    with open(os.path.join(OUT_DIR, "medium_validation.json"), "w") as f:
        json.dump(records, f, indent=2)
    return r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend


# --- Forecast ------------------------------------------------------------
def run_forecast(feat_df, last_data_date):
    train_cutoff = last_data_date + datetime.timedelta(days=1)
    print(f"\n[Forecast]  trained through={last_data_date}  "
          f"first forecast day={train_cutoff}")

    preds = run_mode(feat_df, train_cutoff, pred_base_date=last_data_date)

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

    with open(os.path.join(OUT_DIR, "medium_forecast.json"), "w") as f:
        json.dump(records, f, indent=2)
    return preds


# --- Update shared metadata.json -----------------------------------------
def update_metadata(r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend):
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    meta["medium"] = {
        "model_name":             "xgboost_medium_v2",
        "data_granularity":       "daily",
        "horizon_days":           FORECAST_HORIZON,
        "validation_window_days": VALIDATION_WINDOW,
        "horizons_trained":       HORIZONS,
        "ensemble_size":          len(ENSEMBLE_SEEDS),
        "smooth_window":          SMOOTH_WINDOW,
        "r2_point":               round(r2_point,   6),
        "r2_trend":               round(r2_trend,   6),
        "mae_point":              round(mae_point,  4),
        "rmse_point":             round(rmse_point, 4),
        "mae_trend":              round(mae_trend,  4),
        "rmse_trend":             round(rmse_trend, 4),
        "uncertainty_method":     "calibrated RMSE, 80pct PI (1.28 sigma)",
        "metric_note":            "r2_point vs raw daily ISN; r2_trend vs 25-day trailing smoothed ISN",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("  metadata.json updated (medium key).")


# --- Main ----------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    feat_df, last_data_date = load_features()

    r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend = \
        run_validation(feat_df, last_data_date)
    run_forecast(feat_df, last_data_date)
    update_metadata(r2_point, r2_trend, mae_point, rmse_point, mae_trend, rmse_trend)

    print("\n=== Medium-term pipeline complete ===")
    print(f"  r2_point  : {r2_point:.4f}")
    print(f"  r2_trend  : {r2_trend:.4f}")
    print(f"  mae_point : {mae_point:.2f}  rmse_point : {rmse_point:.2f}")
    print(f"  mae_trend : {mae_trend:.2f}  rmse_trend : {rmse_trend:.2f}")
