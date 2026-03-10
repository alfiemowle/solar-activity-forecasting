"""
Build feature-engineered parquet files from data/silso_daily.csv.
Only rebuilds a parquet if silso_daily.csv has been modified more recently.

Outputs:
  data/features_daily.parquet   -- daily features for pipeline.py and pipeline_medium.py
  data/features_monthly.parquet -- monthly features for pipeline_long.py
"""
import os
import numpy as np
import pandas as pd

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
DATA_CSV        = os.path.join(DATA_DIR, "silso_daily.csv")
DAILY_PARQUET   = os.path.join(DATA_DIR, "features_daily.parquet")
MONTHLY_PARQUET = os.path.join(DATA_DIR, "features_monthly.parquet")

# ---------------------------------------------------------------------------
# Daily feature config -- medium pipeline's set (superset of short pipeline)
# ---------------------------------------------------------------------------
DAILY_LAGS         = [1, 2, 3, 5, 7, 14, 21, 27, 30, 54, 81, 180, 270, 365]
DAILY_ROLL_WINDOWS = [7, 14, 27, 30, 60, 90, 180, 365]
DAILY_ROLL_MIN     = [30, 90]
DAILY_EMA_SPANS    = [7, 30, 90]
DAILY_MOM_LAGS     = [1, 3, 7, 14, 27]
DAILY_CYCLE_PERIOD = 4018   # ~11-year solar cycle in days

# ---------------------------------------------------------------------------
# Monthly feature config -- long pipeline
# ---------------------------------------------------------------------------
MONTHLY_LAGS         = [1, 2, 3, 6, 12, 24, 48, 120, 132]
MONTHLY_ROLL_WINDOWS = [12, 24, 60, 120]
MONTHLY_ROLL_MIN     = [60, 120]
MONTHLY_EMA_SPANS    = [12, 24, 60]
MONTHLY_MOM_LAGS     = [12, 24, 48, 120, 132]
MONTHLY_CYCLE_PERIOD = 132   # ~11-year in months
GLEISSBERG_MONTHS    = 960   # ~80-year Gleissberg cycle
SMOOTH_WINDOW_MONTHLY = 13   # 13-month trailing rolling mean (training target)


def needs_rebuild(parquet_path: str) -> bool:
    if not os.path.isfile(parquet_path):
        return True
    csv_mtime     = os.path.getmtime(DATA_CSV)
    parquet_mtime = os.path.getmtime(parquet_path)
    return csv_mtime > parquet_mtime + 1   # 1-second tolerance


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all daily features. Returns DataFrame with Date, DailyISN,
    provisional, and all feature columns. Rows with any NaN are dropped."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    s = df["DailyISN"].shift(1)  # causal anchor: no look-ahead

    for lag in DAILY_LAGS:
        df[f"lag_{lag}"] = df["DailyISN"].shift(lag)

    for w in DAILY_ROLL_WINDOWS:
        rolled = s.rolling(w)
        df[f"roll_mean_{w}"] = rolled.mean()
        df[f"roll_std_{w}"]  = rolled.std()
        df[f"roll_max_{w}"]  = rolled.max()

    for w in DAILY_ROLL_MIN:
        df[f"roll_min_{w}"] = s.rolling(w).min()

    for span in DAILY_EMA_SPANS:
        df[f"ema_{span}"] = s.ewm(span=span, adjust=False).mean()

    for lag in DAILY_MOM_LAGS:
        df[f"mom_{lag}"] = df["DailyISN"].shift(1) - df["DailyISN"].shift(lag + 1)

    doy = df["Date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Time elapsed from dataset start (anchored to 1818-01-01 for consistency)
    t = (df["Date"] - df["Date"].iloc[0]).dt.days
    df["cycle_sin"]      = np.sin(2 * np.pi * t / DAILY_CYCLE_PERIOD)
    df["cycle_cos"]      = np.cos(2 * np.pi * t / DAILY_CYCLE_PERIOD)
    df["carrington_sin"] = np.sin(2 * np.pi * t / 27.27)
    df["carrington_cos"] = np.cos(2 * np.pi * t / 27.27)

    month = df["Date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to monthly, compute all monthly features. Returns DataFrame
    with Date, MonthlyISN, SmoothedISN, provisional, and feature columns."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["YearMonth"] = df["Date"].dt.to_period("M")
    monthly = (
        df.groupby("YearMonth")
        .agg(MonthlyISN=("DailyISN", "mean"),
             provisional=("provisional", "any"))
        .reset_index()
    )
    monthly["Date"] = monthly["YearMonth"].dt.to_timestamp()
    monthly = monthly.drop(columns=["YearMonth"])
    monthly = monthly.sort_values("Date").reset_index(drop=True)

    # 13-month trailing smoothed ISN -- used as trend training target in pipeline_long.py
    monthly["SmoothedISN"] = (
        monthly["MonthlyISN"]
        .shift(1)
        .rolling(SMOOTH_WINDOW_MONTHLY, min_periods=7)
        .mean()
    )

    s = monthly["MonthlyISN"].shift(1)

    for lag in MONTHLY_LAGS:
        monthly[f"lag_{lag}"] = monthly["MonthlyISN"].shift(lag)

    for w in MONTHLY_ROLL_WINDOWS:
        rolled = s.rolling(w)
        monthly[f"roll_mean_{w}"] = rolled.mean()
        monthly[f"roll_std_{w}"]  = rolled.std()
        monthly[f"roll_max_{w}"]  = rolled.max()

    for w in MONTHLY_ROLL_MIN:
        monthly[f"roll_min_{w}"] = s.rolling(w).min()

    for span in MONTHLY_EMA_SPANS:
        monthly[f"ema_{span}"] = s.ewm(span=span, adjust=False).mean()

    for lag in MONTHLY_MOM_LAGS:
        monthly[f"mom_{lag}"] = monthly["MonthlyISN"].shift(1) - monthly["MonthlyISN"].shift(lag + 1)

    # Time elapsed in months from dataset start (anchored to first month in data)
    t_months = (monthly["Date"] - monthly["Date"].iloc[0]).dt.days / 30.4375
    monthly["cycle_sin"]      = np.sin(2 * np.pi * t_months / MONTHLY_CYCLE_PERIOD)
    monthly["cycle_cos"]      = np.cos(2 * np.pi * t_months / MONTHLY_CYCLE_PERIOD)
    monthly["gleissberg_sin"] = np.sin(2 * np.pi * t_months / GLEISSBERG_MONTHS)
    monthly["gleissberg_cos"] = np.cos(2 * np.pi * t_months / GLEISSBERG_MONTHS)

    monthly.dropna(inplace=True)
    monthly.reset_index(drop=True, inplace=True)
    return monthly


def main():
    if not os.path.isfile(DATA_CSV):
        print("ERROR: data/silso_daily.csv not found. Run data_ingest.py first.")
        raise SystemExit(1)

    rebuild_daily   = needs_rebuild(DAILY_PARQUET)
    rebuild_monthly = needs_rebuild(MONTHLY_PARQUET)

    if not rebuild_daily and not rebuild_monthly:
        print("Both parquet files are up to date. Nothing to rebuild.")
        return

    print("Loading silso_daily.csv...")
    df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    print(f"  {len(df)} rows  ({df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()})")

    if rebuild_daily:
        print("Building features_daily.parquet...")
        feat = build_daily_features(df)
        feat.to_parquet(DAILY_PARQUET, index=False)
        n_feat = len([c for c in feat.columns if c not in ("Date", "DailyISN", "provisional")])
        print(f"  {len(feat)} rows x {len(feat.columns)} cols  "
              f"({n_feat} feature cols)  -> features_daily.parquet")
    else:
        print("features_daily.parquet is up to date.")

    if rebuild_monthly:
        print("Building features_monthly.parquet...")
        feat = build_monthly_features(df)
        feat.to_parquet(MONTHLY_PARQUET, index=False)
        n_feat = len([c for c in feat.columns
                      if c not in ("Date", "MonthlyISN", "SmoothedISN", "provisional")])
        print(f"  {len(feat)} rows x {len(feat.columns)} cols  "
              f"({n_feat} feature cols)  -> features_monthly.parquet")
    else:
        print("features_monthly.parquet is up to date.")


if __name__ == "__main__":
    main()
