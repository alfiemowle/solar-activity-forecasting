# Solar Activity Forecasting

Time-series forecasting of solar activity (sunspot area) from historical observatory records.

This repository contains a **selection of example models and notebooks** taken from a larger MSc dissertation project in Advanced Aerospace Engineering at the University of Liverpool. The full project explored a wide range of architectures; here I’ve included some representative models for others to inspect and run.

## Project goal

The core objective is to **forecast sunspot area** using historical solar observations, under consistent preprocessing and evaluation settings.

More specifically, the project focuses on:

- Turning raw sunspot area measurements into supervised learning datasets (input–target sequences).
- Comparing different modelling approaches for time-series forecasting.
- Evaluating and visualising how well different models predict future solar activity.

These notebooks are intended as **clean, readable examples**, not a complete dump of every experiment run during the MSc project.

## Data

The notebooks are designed to work with historical solar / sunspot activity records, for example:

- **Royal Greenwich Observatory (RGO)** sunspot area data.
- Daily or aggregated sunspot area time series.

Depending on how the repository is configured, you may find data files (e.g. `Combined_RGO.txt`, `daily_area.txt`) inside a `data/` folder

In all cases, the data is treated as a **univariate time series**, then transformed into supervised learning examples by building sequences of past values to predict future values.

