"""
Local MLP Baseline Training - 10 Houses (REDD + UK-DALE)
=========================================================
Train per-house MLP regressors on all 10 cleaned house datasets.

Houses:
  - house1..house5  : REDD dataset
  - house6..house10 : UK-DALE dataset

Usage:
  python models/train_local_mlp.py                # train all houses, no cap
  python models/train_local_mlp.py --balanced      # cap training rows at median
  python models/train_local_mlp.py --cap 5000      # cap training rows at 5000
"""

import argparse
import re
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# =========================
# PATH CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent          # models/
BASE_DIR   = SCRIPT_DIR.parent                        # project root
DATA_DIR   = BASE_DIR / "data" / "processed"
RESULT_DIR = SCRIPT_DIR / "resultmlp"
PLOT_DIR   = RESULT_DIR / "plots"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "main"

# Columns to exclude from features (if present)
EXCLUDE_COLS = {TARGET_COL, "split"}

# Dataset labels by house index
DATASET_LABELS = {
    "house1": "REDD", "house2": "REDD", "house3": "REDD",
    "house4": "REDD", "house5": "REDD",
    "house6": "UK-DALE", "house7": "UK-DALE", "house8": "UK-DALE",
    "house9": "UK-DALE", "house10": "UK-DALE",
}


# =========================
# HELPERS
# =========================
def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R2 between actual and predicted values."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


def plot_actual_vs_predicted(timestamps, y_true, y_pred, house_name, model_name):
    """Save a time-series comparison plot of actual vs predicted values."""
    plt.figure(figsize=(14, 5))
    plt.plot(timestamps, y_true, label="Actual", linewidth=0.8)
    plt.plot(timestamps, y_pred, label="Predicted", linewidth=0.8, alpha=0.8)
    dataset = DATASET_LABELS.get(house_name, "")
    title = f"{house_name} ({dataset}) - {model_name}: Actual vs Predicted"
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Scaled main")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{house_name}_{model_name}.png", dpi=150)
    plt.close()


def detect_feature_cols(df):
    """Auto-detect feature columns by excluding target and meta columns."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def validate_columns(df, required_cols, file_name):
    """Raise if required columns are missing from a DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing columns: {missing}")


def find_house_splits(data_dir):
    """
    Scan data_dir for house*_train.csv / house*_test.csv pairs.
    Returns dict: { house_name: { "train": Path, "test": Path, "valid": Path|None } }
    """
    train_files = sorted(data_dir.glob("house*_train.csv"))
    house_map = {}

    for train_path in train_files:
        match = re.match(r"(house\d+)_train\.csv", train_path.name)
        if not match:
            continue

        house_name = match.group(1)
        valid_path = data_dir / f"{house_name}_valid.csv"
        test_path  = data_dir / f"{house_name}_test.csv"

        if not test_path.exists():
            print(f"[WARNING] Missing test file for {house_name}: {test_path.name}")
            continue

        house_map[house_name] = {
            "train": train_path,
            "valid": valid_path if valid_path.exists() else None,
            "test":  test_path,
        }

    return house_map


def subsample(df, max_rows, random_state=42):
    """Subsample DataFrame to at most max_rows, preserving temporal distribution."""
    if max_rows is None or len(df) <= max_rows:
        return df
    # Evenly spaced indices to preserve time-series distribution
    indices = np.linspace(0, len(df) - 1, max_rows, dtype=int)
    return df.iloc[indices].copy()


# =========================
# MODELS
# =========================
def run_naive_baseline(test_df, house_name, feature_cols):
    """
    Naive baseline: predict current target using lag_1.
    """
    if "lag_1" not in test_df.columns:
        print(f"  [SKIP naive] {house_name}: no 'lag_1' column")
        return None

    valid_df = test_df.dropna(subset=[TARGET_COL, "lag_1"]).copy()
    if valid_df.empty:
        print(f"  [SKIP naive] {house_name}: no valid rows after dropna")
        return None

    y_true     = valid_df[TARGET_COL].values
    y_pred     = valid_df["lag_1"].values
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
        "dataset": DATASET_LABELS.get(house_name, "unknown"),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def run_mlp_regression(train_df, test_df, house_name, feature_cols):
    """
    Train an MLP regressor on train_df and evaluate on test_df.
    """
    validate_columns(train_df, feature_cols + [TARGET_COL], f"{house_name}_train.csv")
    validate_columns(test_df, feature_cols + [TARGET_COL], f"{house_name}_test.csv")

    train_valid = train_df.dropna(subset=feature_cols + [TARGET_COL]).copy()
    test_valid  = test_df.dropna(subset=feature_cols + [TARGET_COL]).copy()

    if train_valid.empty or test_valid.empty:
        raise ValueError(f"{house_name}: empty train/test after dropping missing values.")

    X_train = train_valid[feature_cols]
    y_train = train_valid[TARGET_COL]

    X_test     = test_valid[feature_cols]
    y_test     = test_valid[TARGET_COL]
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
        "dataset": DATASET_LABELS.get(house_name, "unknown"),
        "train_rows": len(train_valid),
        "test_rows": len(test_valid),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# =========================
# PIPELINE
# =========================
def process_house(house_name, file_dict, cap=None):
    """Load data, optionally subsample, and run baselines + MLP for one house."""
    dataset = DATASET_LABELS.get(house_name, "unknown")
    print(f"\n{'='*50}")
    print(f"  {house_name}  ({dataset})")
    print(f"{'='*50}")

    train_df = pd.read_csv(file_dict["train"], index_col=0, parse_dates=True)
    test_df  = pd.read_csv(file_dict["test"],  index_col=0, parse_dates=True)

    original_train_len = len(train_df)

    # Subsample if needed (to balance data between REDD and UK-DALE)
    if cap is not None and len(train_df) > cap:
        train_df = subsample(train_df, cap)
        print(f"  [CAP] Subsampled train: {original_train_len} -> {len(train_df)} rows")
    else:
        print(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")

    # Auto-detect features from common columns between train and test
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    feature_cols = [c for c in common_cols if c not in EXCLUDE_COLS]

    if not feature_cols:
        raise ValueError(f"{house_name}: no feature columns found after excluding {EXCLUDE_COLS}")

    naive_result = run_naive_baseline(test_df, house_name, feature_cols)
    mlp_result   = run_mlp_regression(train_df, test_df, house_name, feature_cols)

    if naive_result:
        print(
            f"  Naive  ->  MAE={naive_result['mae']:.6f}  "
            f"RMSE={naive_result['rmse']:.6f}  "
            f"R2={naive_result['r2']:.6f}"
        )
    print(
        f"  MLP    ->  MAE={mlp_result['mae']:.6f}  "
        f"RMSE={mlp_result['rmse']:.6f}  "
        f"R2={mlp_result['r2']:.6f}"
    )

    results = []
    if naive_result:
        results.append(naive_result)
    results.append(mlp_result)
    return results


def print_data_summary(house_splits):
    """Print a summary table of data sizes to highlight imbalance."""
    print("\n" + "=" * 65)
    print("  DATA SIZE SUMMARY  (train split)")
    print("=" * 65)
    print(f"  {'House':<10} {'Dataset':<10} {'Train Rows':>12} {'Ratio vs Min':>14}")
    print("  " + "-" * 50)

    sizes = {}
    for house_name in sorted(house_splits.keys(), key=lambda h: int(re.search(r'\d+', h).group())):
        train_path = house_splits[house_name]["train"]
        # Quick row count (no full DataFrame load)
        with open(train_path, "r", encoding="utf-8") as f:
            row_count = sum(1 for _ in f) - 1  # minus header
        sizes[house_name] = row_count

    min_size = min(sizes.values()) if sizes else 1

    for house_name, row_count in sizes.items():
        dataset = DATASET_LABELS.get(house_name, "?")
        ratio = row_count / min_size
        marker = " [!]" if ratio > 10 else ""
        print(f"  {house_name:<10} {dataset:<10} {row_count:>12,} {ratio:>13.1f}x{marker}")

    print()
    median_size = int(np.median(list(sizes.values())))
    print(f"  Median size: {median_size:,} rows")
    print(f"  -> Use --balanced or --cap {median_size} to balance data")
    print()

    return sizes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train local MLP baselines for all 10 houses (REDD + UK-DALE)"
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Cap training rows per house at the median size across all houses "
             "(reduce bias from UK-DALE having much more data than REDD)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Cap training rows per house at this exact number",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  LOCAL MLP BASELINE TRAINING")
    print("  10 Houses: REDD (house1-5) + UK-DALE (house6-10)")
    print("=" * 65)

    house_splits = find_house_splits(DATA_DIR)

    if not house_splits:
        raise FileNotFoundError(
            f"No house split files found in {DATA_DIR}. "
            f"Expected files like house1_train.csv and house1_test.csv"
        )

    print(f"\nFound {len(house_splits)} houses: {sorted(house_splits.keys())}")

    # Print data size summary table
    sizes = print_data_summary(house_splits)

    # Calculate cap if --balanced
    cap = args.cap
    if args.balanced and cap is None:
        cap = int(np.median(list(sizes.values())))
        print(f"[BALANCED MODE] Capping training data at median = {cap:,} rows/house\n")
    elif cap is not None:
        print(f"[CAP MODE] Capping training data at {cap:,} rows/house\n")

    all_results = []

    # Sort by house number
    sorted_houses = sorted(
        house_splits.keys(),
        key=lambda h: int(re.search(r'\d+', h).group())
    )

    for house_name in sorted_houses:
        file_dict = house_splits[house_name]
        try:
            results = process_house(house_name, file_dict, cap=cap)
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] {house_name}: {e}")

    if not all_results:
        raise RuntimeError("No houses were processed successfully.")

    # Save results
    results_df = pd.DataFrame(all_results)

    # Ensure dataset column is present
    base_cols = ["model", "house", "dataset", "mae", "rmse", "r2"]
    extra_cols = [c for c in results_df.columns if c not in base_cols]
    results_df = results_df[base_cols + extra_cols]

    results_df.to_csv(RESULT_DIR / "local_metrics.csv", index=False)

    # Print summary
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    # Show MLP results
    mlp_df = results_df[results_df["model"] == "mlp_regression"].copy()
    if not mlp_df.empty:
        print("\n  MLP Regression per house:")
        print(f"  {'House':<10} {'Dataset':<10} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
        print("  " + "-" * 48)
        for _, row in mlp_df.iterrows():
            print(
                f"  {row['house']:<10} {row['dataset']:<10} "
                f"{row['mae']:>8.4f} {row['rmse']:>8.4f} {row['r2']:>8.4f}"
            )

        print(f"\n  Average across {len(mlp_df)} houses:")
        print(f"    MAE  = {mlp_df['mae'].mean():.4f}")
        print(f"    RMSE = {mlp_df['rmse'].mean():.4f}")
        print(f"    R2   = {mlp_df['r2'].mean():.4f}")

        print(f"\n  REDD average    (house1-5):")
        redd = mlp_df[mlp_df["dataset"] == "REDD"]
        if not redd.empty:
            print(f"    MAE={redd['mae'].mean():.4f}  RMSE={redd['rmse'].mean():.4f}  R2={redd['r2'].mean():.4f}")

        print(f"  UK-DALE average (house6-10):")
        ukdale = mlp_df[mlp_df["dataset"] == "UK-DALE"]
        if not ukdale.empty:
            print(f"    MAE={ukdale['mae'].mean():.4f}  RMSE={ukdale['rmse'].mean():.4f}  R2={ukdale['r2'].mean():.4f}")

    print(f"\n  Saved metrics -> {RESULT_DIR / 'local_metrics.csv'}")
    print(f"  Saved plots  -> {PLOT_DIR}")
    print()


if __name__ == "__main__":
    main()
