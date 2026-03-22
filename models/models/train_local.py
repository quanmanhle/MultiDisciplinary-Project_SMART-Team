import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "processed"
RESULT_DIR = BASE_DIR / "results"
PLOT_DIR = RESULT_DIR / "plots"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "main"


# =========================
# METRICS
# =========================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# =========================
# PLOT
# =========================
def plot_actual_vs_predicted(timestamps, y_true, y_pred, house_name, model_name):
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, y_true, label="Actual")
    plt.plot(timestamps, y_pred, label="Predicted")
    plt.title(f"{house_name} - {model_name}: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Main")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{house_name}_{model_name}.png", dpi=150)
    plt.close()


# =========================
# SPLIT TIME SERIES
# =========================
def time_split(df, train_ratio=0.8):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


# =========================
# NAIVE BASELINE
# =========================
def run_naive_baseline(test_df, house_name):
    # dùng lag_1 để dự đoán main
    valid_df = test_df.dropna(subset=["lag_1", TARGET_COL]).copy()

    y_true = valid_df[TARGET_COL].values
    y_pred = valid_df["lag_1"].values
    timestamps = valid_df.index

    mae, rmse, r2 = compute_metrics(y_true, y_pred)

    plot_actual_vs_predicted(
        timestamps=timestamps,
        y_true=y_true,
        y_pred=y_pred,
        house_name=house_name,
        model_name="naive_baseline"
    )

    return {
        "model": "naive_baseline",
        "house": house_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# =========================
# LINEAR REGRESSION
# =========================
def run_linear_regression(train_df, test_df, house_name):
    feature_cols = [col for col in train_df.columns if col != TARGET_COL]

    train_valid = train_df.dropna(subset=feature_cols + [TARGET_COL]).copy()
    test_valid = test_df.dropna(subset=feature_cols + [TARGET_COL]).copy()

    X_train = train_valid[feature_cols]
    y_train = train_valid[TARGET_COL]

    X_test = test_valid[feature_cols]
    y_test = test_valid[TARGET_COL]
    timestamps = test_valid.index

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae, rmse, r2 = compute_metrics(y_test, y_pred)

    plot_actual_vs_predicted(
        timestamps=timestamps,
        y_true=y_test,
        y_pred=y_pred,
        house_name=house_name,
        model_name="linear_regression"
    )

    return {
        "model": "linear_regression",
        "house": house_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# =========================
# ONE HOUSE
# =========================
def process_house(csv_path):
    house_name = csv_path.stem.replace("_hourly_clean", "")
    print(f"\n=== Processing {house_name} ===")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{house_name} missing target column: {TARGET_COL}")

    train_df, test_df = time_split(df, train_ratio=0.8)

    naive_result = run_naive_baseline(test_df, house_name)
    lr_result = run_linear_regression(train_df, test_df, house_name)

    print(
        f"[OK] {house_name} | "
        f"Naive MAE={naive_result['mae']:.4f}, RMSE={naive_result['rmse']:.4f}, R2={naive_result['r2']:.4f}"
    )
    print(
        f"[OK] {house_name} | "
        f"LR    MAE={lr_result['mae']:.4f}, RMSE={lr_result['rmse']:.4f}, R2={lr_result['r2']:.4f}"
    )

    return [naive_result, lr_result]


# =========================
# MAIN
# =========================
def main():
    csv_files = sorted(DATA_DIR.glob("house*_hourly_clean.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No clean files found in: {DATA_DIR}")

    all_results = []

    for csv_path in csv_files:
        try:
            results = process_house(csv_path)
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] {csv_path.name}: {e}")

    if not all_results:
        raise RuntimeError("No houses were processed successfully.")

    results_df = pd.DataFrame(all_results)
    results_df = results_df[["model", "house", "mae", "rmse", "r2"]]
    results_df.to_csv(RESULT_DIR / "local_metrics.csv", index=False)

    print("\n=== DONE ===")
    print(results_df)
    print(f"\nSaved metrics to: {RESULT_DIR / 'local_metrics.csv'}")
    print(f"Saved plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()