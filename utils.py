"""
Shared utilities for solar forecast pipelines.

Functions:
  predict_ensemble   -- mean + std from an XGBoost ensemble
  calibrate_sigma    -- per-horizon calibrated RMSE on trailing rows
  load_parquet       -- load a feature parquet and return (df, last_date)
  build_predictions  -- interpolate predictions over a horizon from one anchor row
  update_metadata    -- read / merge / write metadata.json
"""

import json
import os
import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------

def predict_ensemble(models, X):
    """Return (mean_pred, std_pred) arrays from an ensemble of models."""
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


# ---------------------------------------------------------------------------
# Uncertainty calibration
# ---------------------------------------------------------------------------

def calibrate_sigma(horizon_models, feat_df, cutoff_dt, feature_cols, cal_window,
                    target_isn=None, isn_col="DailyISN",
                    sigma_floor=5.0, default_sigma=30.0):
    """Per-horizon calibrated RMSE on the last cal_window training rows.

    Parameters
    ----------
    horizon_models  : dict {h: list_of_models}
    feat_df         : full feature DataFrame (with Date column)
    cutoff_dt       : pd.Timestamp -- rows strictly before this are training rows
    feature_cols    : list of feature column names
    cal_window      : int -- number of trailing training rows to calibrate on
    target_isn      : optional pd.Series -- use instead of feat_df[isn_col]
    isn_col         : str -- ISN column name when target_isn is None
    sigma_floor     : float or callable(h) -> float -- minimum sigma per horizon
    default_sigma   : float -- fallback when no valid calibration rows exist
    """
    train_rows = feat_df[feat_df["Date"] < cutoff_dt]
    cal_rows   = train_rows.iloc[-cal_window:] if len(train_rows) > cal_window else train_rows

    base = target_isn if target_isn is not None else feat_df[isn_col]
    cal_sigma = {}
    for h, models in horizon_models.items():
        floor      = sigma_floor(h) if callable(sigma_floor) else sigma_floor
        target     = base.shift(-h)
        idx        = cal_rows.index
        valid_mask = target.loc[idx].notna()
        valid_idx  = idx[valid_mask]
        if len(valid_idx) == 0:
            cal_sigma[h] = default_sigma
            continue
        X_cal = feat_df.loc[valid_idx, feature_cols].values
        y_cal = target.loc[valid_idx].values
        mean_pred, _ = predict_ensemble(models, X_cal)
        rmse = float(np.sqrt(np.mean((y_cal - mean_pred) ** 2)))
        cal_sigma[h] = max(rmse, floor)
    return cal_sigma


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_parquet(path, date_col="Date"):
    """Load a parquet feature file and parse the date column.

    Returns (df, last_date) where last_date is a pd.Timestamp.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{os.path.basename(path)} not found. Run data_prepare.py first."
        )
    df = pd.read_parquet(path)
    df[date_col] = pd.to_datetime(df[date_col])
    last_date = df[date_col].max()
    print(f"Loaded {os.path.basename(path)}")
    print(f"  {len(df)} rows  "
          f"({df[date_col].iloc[0].date()} -> {df[date_col].iloc[-1].date()})")
    return df, last_date


# ---------------------------------------------------------------------------
# Prediction builder (shared interpolation loop)
# ---------------------------------------------------------------------------

def build_predictions(point_models, trend_models, point_sigma, trend_sigma,
                      horizons, X_pred, pred_base, forecast_horizon,
                      use_months=False):
    """Interpolate ensemble predictions over a forecast horizon.

    Parameters
    ----------
    point_models, trend_models : dict {h: list_of_models}
    point_sigma, trend_sigma   : dict {h: float}
    horizons                   : list of int -- trained horizon steps
    X_pred                     : np.ndarray shape (1, n_features) -- anchor row
    pred_base                  : datetime.date (use_months=False) or
                                 pd.Timestamp  (use_months=True)
    forecast_horizon           : int -- number of steps to predict
    use_months                 : bool -- if True offsets are months, else days

    Returns list of dicts with keys: date, predicted_point, predicted_trend,
    lower_bound_point, upper_bound_point, lower_bound_trend, upper_bound_trend.
    """
    sorted_horizons = sorted(horizons)
    results = []

    for offset in range(1, forecast_horizon + 1):
        if use_months:
            target_date = pd.Timestamp(pred_base) + pd.DateOffset(months=offset)
            date_str    = target_date.strftime("%Y-%m-%d")
        else:
            target_date = pred_base + datetime.timedelta(days=offset)
            date_str    = target_date.isoformat()

        # Find the bracketing horizon pair for interpolation
        h_lo = sorted_horizons[0]
        h_hi = sorted_horizons[-1]
        for h in sorted_horizons:
            if h <= offset:
                h_lo = h
            if h >= offset and h_hi >= h:
                h_hi = h

        if h_lo == h_hi or offset <= sorted_horizons[0]:
            h_use = min(horizons, key=lambda hh: abs(hh - offset))
            pp, _ = predict_ensemble(point_models[h_use], X_pred)
            tp, _ = predict_ensemble(trend_models[h_use], X_pred)
            p_pred, t_pred = float(pp[0]), float(tp[0])
            p_sig,  t_sig  = point_sigma[h_use], trend_sigma[h_use]
        else:
            pp_lo, _ = predict_ensemble(point_models[h_lo], X_pred)
            pp_hi, _ = predict_ensemble(point_models[h_hi], X_pred)
            tp_lo, _ = predict_ensemble(trend_models[h_lo], X_pred)
            tp_hi, _ = predict_ensemble(trend_models[h_hi], X_pred)
            alpha  = (offset - h_lo) / (h_hi - h_lo)
            p_pred = float(pp_lo[0] * (1 - alpha) + pp_hi[0] * alpha)
            t_pred = float(tp_lo[0] * (1 - alpha) + tp_hi[0] * alpha)
            p_sig  = point_sigma[h_lo] * (1 - alpha) + point_sigma[h_hi] * alpha
            t_sig  = trend_sigma[h_lo] * (1 - alpha) + trend_sigma[h_hi] * alpha

        p_pred = max(0.0, p_pred)
        t_pred = max(0.0, t_pred)
        results.append({
            "date":               date_str,
            "predicted_point":    round(p_pred, 2),
            "predicted_trend":    round(t_pred, 2),
            "lower_bound_point":  round(max(0.0, p_pred - 1.28 * p_sig), 2),
            "upper_bound_point":  round(p_pred + 1.28 * p_sig, 2),
            "lower_bound_trend":  round(max(0.0, t_pred - 1.28 * t_sig), 2),
            "upper_bound_trend":  round(t_pred + 1.28 * t_sig, 2),
        })

    return results


# ---------------------------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------------------------

def update_metadata(meta_path, key, data):
    """Read existing metadata.json (if any), merge data, then write back.

    If key is None, data is merged into the top level.
    Otherwise data is stored at meta[key].
    """
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    if key is None:
        meta.update(data)
    else:
        meta[key] = data
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
