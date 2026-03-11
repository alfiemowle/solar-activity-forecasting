"""
Central configuration for all solar forecast pipelines.

Sections
--------
ENSEMBLE_SEEDS  -- shared across all pipelines
SHORT           -- short-term 30-day daily pipeline
MEDIUM          -- medium-term 365-day daily pipeline
LONG            -- long-term 144-month monthly pipeline
"""

ENSEMBLE_SEEDS = [42, 7, 123, 1, 2, 3, 99]

# ---------------------------------------------------------------------------
# Short-term pipeline (pipeline.py)
# ---------------------------------------------------------------------------
SHORT = dict(
    FORECAST_HORIZON  = 30,
    VALIDATION_WINDOW = 30,
    CAL_WINDOW        = 730,    # trailing training rows for sigma calibration
    SMOOTH_WINDOW     = 25,     # trailing rolling mean for trend targets
    WEIGHT_HALFLIFE   = 90,     # exponential sample-weight half-life (days)
    HORIZONS          = [1, 2, 3, 4, 5, 6, 7, 10, 14, 18, 21, 25, 28, 30],
    XGB_PARAMS        = dict(
        n_estimators     = 600,
        max_depth        = 4,
        learning_rate    = 0.03,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        reg_alpha        = 0.1,
        reg_lambda       = 1.5,
        gamma            = 0.05,
    ),
)

# ---------------------------------------------------------------------------
# Medium-term pipeline (pipeline_medium.py)
# ---------------------------------------------------------------------------
MEDIUM = dict(
    FORECAST_HORIZON  = 365,
    VALIDATION_WINDOW = 365,
    CAL_WINDOW        = 730,
    SMOOTH_WINDOW     = 25,
    WEIGHT_HALFLIFE   = 90,
    TRAIN_START_DATE  = "2010-01-01",   # limit to recent cycles for efficiency
    HORIZONS          = [1, 7, 14, 30, 60, 90, 120, 180, 270, 365],
    XGB_PARAMS        = dict(
        n_estimators     = 500,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        reg_alpha        = 0.1,
        reg_lambda       = 1.5,
        gamma            = 0.05,
    ),
)

# ---------------------------------------------------------------------------
# Long-term pipeline (pipeline_long.py)
# ---------------------------------------------------------------------------
LONG = dict(
    FORECAST_HORIZON  = 144,    # months (12 years)
    VALIDATION_WINDOW = 144,
    CAL_WINDOW        = 120,    # months (10 years) for calibration
    HORIZONS          = [1, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144],
    XGB_PARAMS        = dict(
        n_estimators     = 500,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        reg_alpha        = 0.1,
        reg_lambda       = 1.5,
    ),
)
