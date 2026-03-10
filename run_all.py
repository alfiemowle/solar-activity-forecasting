"""
Run all pipeline stages in order:
  1. data_ingest.py    -- fetch new SILSO data, append to CSV
  2. data_prepare.py   -- build feature parquets (skips if up to date)
  3. pipeline.py       -- short-term 30-day forecast
  4. pipeline_medium.py -- medium-term 365-day forecast
  5. pipeline_long.py  -- long-term 144-month forecast

A failure in any stage stops execution immediately.
"""
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    "data_ingest.py",
    "data_prepare.py",
    "pipeline.py",
    "pipeline_medium.py",
    "pipeline_long.py",
]

def main():
    for script in STAGES:
        path = os.path.join(BASE_DIR, script)
        print(f"\n{'=' * 60}")
        print(f"  {script}")
        print(f"{'=' * 60}")
        result = subprocess.run([sys.executable, path])
        if result.returncode != 0:
            print(f"\nERROR: {script} failed (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print(f"\n{'=' * 60}")
    print("  All stages complete.")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
