"""
Fetch latest SILSO daily sunspot data and append new rows to data/silso_daily.csv.
Safe to run at any time -- only appends rows newer than the last date already stored.
"""
import os
import datetime
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_CSV = os.path.join(DATA_DIR, "silso_daily.csv")
SILSO_URL = "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt"


def fetch_silso() -> pd.DataFrame:
    """Download full SILSO daily ISN dataset, return parsed DataFrame."""
    resp = requests.get(SILSO_URL, timeout=60)
    resp.raise_for_status()

    records = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            day   = int(parts[2])
            # parts[3] = DecimalDate, parts[4] = DailyISN
            isn_raw     = parts[4] if len(parts) >= 5 else parts[3]
            provisional = "*" in isn_raw
            isn         = float(isn_raw.replace("*", ""))
        except (ValueError, IndexError):
            continue
        if isn == -1.0:
            isn = float("nan")
        try:
            date = datetime.date(year, month, day)
        except ValueError:
            continue
        records.append({"Date": date, "DailyISN": isn, "provisional": provisional})

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching SILSO daily sunspot data...")
    remote = fetch_silso()
    print(f"  Downloaded {len(remote)} rows  "
          f"({remote['Date'].iloc[0].date()} -> {remote['Date'].iloc[-1].date()})")

    if not os.path.isfile(DATA_CSV):
        # First run -- write full dataset with date-gap filling and interpolation
        full_idx = pd.date_range(remote["Date"].iloc[0], remote["Date"].iloc[-1], freq="D")
        df = remote.set_index("Date").reindex(full_idx).reset_index()
        df.rename(columns={"index": "Date"}, inplace=True)
        df["provisional"] = df["provisional"].fillna(False)
        df["DailyISN"]    = df["DailyISN"].interpolate(method="linear")
        df["Date"]        = df["Date"].dt.date
        df.to_csv(DATA_CSV, index=False)
        print(f"  Created silso_daily.csv with {len(df)} rows.")
        return

    # Incremental update
    existing = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    last_existing = existing["Date"].max()

    new_rows = remote[remote["Date"] > last_existing].copy()
    if new_rows.empty:
        print(f"  Already up to date (last date: {last_existing.date()}). 0 new rows added.")
        return

    # Fill any calendar gaps in the new rows and interpolate
    date_range = pd.date_range(
        last_existing + pd.Timedelta(days=1), new_rows["Date"].iloc[-1], freq="D"
    )
    new_rows = (
        new_rows.set_index("Date")
                .reindex(date_range)
                .reset_index()
                .rename(columns={"index": "Date"})
    )
    new_rows["provisional"] = new_rows["provisional"].fillna(False)
    new_rows["DailyISN"]    = new_rows["DailyISN"].interpolate(method="linear")
    new_rows["Date"]        = new_rows["Date"].dt.date

    combined = pd.concat(
        [existing.assign(Date=pd.to_datetime(existing["Date"]).dt.date), new_rows],
        ignore_index=True,
    )
    combined.sort_values("Date", inplace=True)
    combined.to_csv(DATA_CSV, index=False)
    print(f"  Added {len(new_rows)} new rows. Last date now: {new_rows['Date'].iloc[-1]}")


if __name__ == "__main__":
    main()
