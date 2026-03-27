import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULT_DIR = BASE_DIR / "resultmlp"
PLOT_DIR = RESULT_DIR / "plots"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "main"

FEATURE_COLS = [
    "dish washer",
    "electric stove",
    "fridge",
    "microwave",
    "washer dryer",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "lag_1",
    "rolling_mean",
]


# =========================
# HELPERS
# =========================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def plot_actual_vs_predicted(timestamps, y_true, y_pred, house_name, model_name):
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, y_true, label="Actual")
    plt.plot(timestamps, y_pred, label="Predicted")
    plt.title(f"{house_name} - {model_name}: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Scaled main")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{house_name}_{model_name}.png", dpi=150)
    plt.close()


def validate_columns(df, required_cols, file_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing columns: {missing}")


def find_house_splits(data_dir):
    """
    Find house train/test files
    """
    train_files = sorted(data_dir.glob("house*_train.csv"))
    house_map = {}

    for train_path in train_files:
        match = re.match(r"(house\d+)_train\.csv", train_path.name)
        if not match:
            continue

        house_name = match.group(1)
        valid_path = data_dir / f"{house_name}_valid.csv"
        test_path = data_dir / f"{house_name}_test.csv"

        if not test_path.exists():
            print(f"[WARNING] Missing test file for {house_name}: {test_path.name}")
            continue

        house_map[house_name] = {
            "train": train_path,
            "valid": valid_path if valid_path.exists() else None,
            "test": test_path,
        }

    return house_map


# =========================
# MODELS
# =========================
def run_naive_baseline(test_df, house_name):
    """
    Naive baseline: predict current target using lag_1.
    """
    validate_columns(test_df, [TARGET_COL, "lag_1"], f"{house_name}_test.csv")

    valid_df = test_df.dropna(subset=[TARGET_COL, "lag_1"]).copy()
    if valid_df.empty:
        raise ValueError(f"{house_name}: no valid rows for naive baseline.")

    y_true = valid_df[TARGET_COL].values
    y_pred = valid_df["lag_1"].values
    timestamps = valid_df.index

    mae, rmse, r2 = compute_metrics(y_true, y_pred)

    plot_actual_vs_predicted(
        timestamps=timestamps,
        y_true=y_true,
        y_pred=y_pred,
        house_name=house_name,
        model_name="naive_baseline",
    )

    return {
        "model": "naive_baseline",
        "house": house_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def run_mlp_regression(train_df, test_df, house_name):
    validate_columns(train_df, FEATURE_COLS + [TARGET_COL], f"{house_name}_train.csv")
    validate_columns(test_df, FEATURE_COLS + [TARGET_COL], f"{house_name}_test.csv")

    train_valid = train_df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    test_valid = test_df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    if train_valid.empty or test_valid.empty:
        raise ValueError(f"{house_name}: empty train/test after dropping missing values.")

    X_train = train_valid[FEATURE_COLS]
    y_train = train_valid[TARGET_COL]

    X_test = test_valid[FEATURE_COLS]
    y_test = test_valid[TARGET_COL]
    timestamps = test_valid.index

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae, rmse, r2 = compute_metrics(y_test, y_pred)

    plot_actual_vs_predicted(
        timestamps=timestamps,
        y_true=y_test,
        y_pred=y_pred,
        house_name=house_name,
        model_name="mlp_regression",
    )

    return {
        "model": "mlp_regression",
        "house": house_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# =========================
# PIPELINE
# =========================
def process_house(house_name, file_dict):
    print(f"\n=== Processing {house_name} ===")

    train_df = pd.read_csv(file_dict["train"], index_col=0, parse_dates=True)
    test_df = pd.read_csv(file_dict["test"], index_col=0, parse_dates=True)

    naive_result = run_naive_baseline(test_df, house_name)
    mlp_result = run_mlp_regression(train_df, test_df, house_name)

    print(
        f"[OK] {house_name} | "
        f"Naive MAE={naive_result['mae']:.6f}, "
        f"RMSE={naive_result['rmse']:.6f}, "
        f"R2={naive_result['r2']:.6f}"
    )
    print(
        f"[OK] {house_name} | "
        f"MLP   MAE={mlp_result['mae']:.6f}, "
        f"RMSE={mlp_result['rmse']:.6f}, "
        f"R2={mlp_result['r2']:.6f}"
    )

    return [naive_result, mlp_result]


def main():
    house_splits = find_house_splits(DATA_DIR)

    if not house_splits:
        raise FileNotFoundError(
            f"No house split files found in {DATA_DIR}. "
            f"Expected files like house1_train.csv and house1_test.csv"
        )

    all_results = []

    for house_name, file_dict in house_splits.items():
        try:
            results = process_house(house_name, file_dict)
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] {house_name}: {e}")

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
