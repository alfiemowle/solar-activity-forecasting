# Solar Activity Forecasting

Time-series forecasting of solar activity (sunspot area) using deep learning and classical machine learning models in Python.

This project is based on my MSc dissertation in **Advanced Aerospace Engineering** at the University of Liverpool, where I built disciplined, reproducible PyTorch pipelines for comparing multiple architectures on historical solar data.

---

## Project overview

The goal is to forecast sunspot area from historical observatory records under **consistent preprocessing and evaluation settings**.

I compare:

- **LSTM** sequence model in PyTorch  
- **Transformer** with positional encoding  
- **Autoencoder + MLP** approach  
- **XGBoost** regression as a classical ML baseline  

All models share the same:

- train/validation/test splits  
- preprocessing (scaling, lag windows / sequence generation)  
- error metrics (MAE, MSE, RMSE, R²)  

---

## Methods & tools

- **Python / ML stack**
  - PyTorch (LSTM, Transformer, autoencoder)
  - scikit-learn (metrics, model wrappers, Grid/RandomizedSearchCV)
  - XGBoost
  - NumPy, Pandas, Matplotlib

- **ML engineering practices**
  - Custom `Dataset` and `DataLoader` for time series
  - On-the-fly noise augmentation
  - Leak-free scaling and deterministic splits
  - Seed control, gradient clipping, learning-rate scheduling, early stopping
  - Best-model checkpointing and basic throughput/latency benchmarking

- **Signal processing**
  - Resampling and smoothing (e.g. tapered boxcar)
  - Lag-window / sequence generation
  - (Optionally) Kalman filtering for noisy observations

---

## Repository structure

_Example layout (may evolve):_

```text
notebooks/
  01_lstm_rgo_sum.ipynb         # LSTM model on RGO sunspot data
  02_transformer_rgo.ipynb      # Transformer with positional encoding
  03_autoencoder_sunspot.ipynb  # Autoencoder + MLP forecasting
  04_xgboost_baseline.ipynb     # XGBoost baseline model

data/
  Combined_RGO.txt              # Example input data (or see instructions below)
  daily_area.txt

models/
  best_transformer_model.pth    # Saved PyTorch models (optional)

src/                            # (Optional) shared utilities
  datasets.py
  models.py
  train.py

