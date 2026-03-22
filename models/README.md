# Local Baseline Training

This module trains and evaluates **local baseline models** for household energy consumption forecasting using the cleaned hourly datasets.

## What this Python module does

The Python script in `models/train_local.py` is responsible for:

- loading the **5 cleaned hourly house files** from the `processed/` folder
- using **`main` as the target variable**
- running **two baseline levels** for each house:
  - **Naive baseline**: predicts the current hour using the previous hour value (`lag_1`)
  - **Linear Regression**: trains a local regression model using all input features except `main`
- splitting each house dataset into **train and test sets in time order**
- computing evaluation metrics for each house:
  - **MAE**
  - **RMSE**
  - **R²**
- generating **actual vs predicted plots** for each house and each model
- saving the final evaluation summary to `results/local_metrics.csv`

## Objective

The goal of this component is to build a **per-house local baseline** that can later be compared with the federated learning model.

We use:

- **Naive baseline**: predict the current value using the previous hour consumption (`lag_1`)
- **Linear Regression**: train a local regression model for each house

## Dataset

This script uses the **5 cleaned hourly files** in the `processed/` folder:

- `house1_hourly_clean.csv`
- `house2_hourly_clean.csv`
- `house3_hourly_clean.csv`
- `house4_hourly_clean.csv`
- `house6_hourly_clean.csv`

## Target Variable

- **Target**: `main`

The `main` column represents the household energy consumption to be predicted.

## Input Features

The model uses all columns except `main` as input features.

Typical features include:

- appliance-level usage:
  - `dish washer`
  - `electric stove`
  - `fridge`
  - `microwave`
  - `washer dryer`
- time-based features:
  - `hour`
  - `day`
- historical features:
  - `lag_1`
  - `rolling_mean`

## Models

### 1. Naive Baseline
A simple baseline where:

```python
y_pred = lag_1