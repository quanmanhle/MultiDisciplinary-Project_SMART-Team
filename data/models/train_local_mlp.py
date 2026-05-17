"""
Local MLP Baseline Training - 10 Houses (REDD + UK-DALE)
=========================================================
Train per-house MLP regressors on all 10 cleaned house datasets.

By default this script trains from house*_clean.csv in original watts, then
uses MinMaxScaler internally for MLP stability. This avoids exposing the model
to StandardScaled negative target values and exports predictions back in watts.

Use --data-mode standard to reproduce the older StandardScaled split behavior.

Houses:
  - house1..house5  : REDD dataset
  - house6..house10 : UK-DALE dataset

Usage:
  python models/train_local_mlp.py                # train all houses, no cap
  python models/train_local_mlp.py --balanced      # cap training rows at median
  python models/train_local_mlp.py --cap 5000      # cap training rows at 5000
  python models/train_local_mlp.py --data-mode standard
"""

import argparse
import re
import sys
import io
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler

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
MODEL_DIR  = RESULT_DIR / "saved_models"
PRED_DIR   = RESULT_DIR / "predictions"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "main"
DATA_MODE_WATT = "watt"
DATA_MODE_STANDARD = "standard"
EWMA_ALPHA = 0.05
NON_NEGATIVE_WATT_COLS = {
    TARGET_COL,
    "dish washer",
    "electric stove",
    "fridge",
    "microwave",
    "washer dryer",
    "lag_1",
    "rolling_mean",
}

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


def unit_label_for(data_mode):
    """Return display label for metric/plot units."""
    if data_mode == DATA_MODE_WATT:
        return "Main power (W)"
    return "Main power (StandardScaled)"


def clip_non_negative(values):
    """Clamp physically impossible negative watt predictions to zero."""
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def causal_ewma_from_lag(lag_values, alpha=EWMA_ALPHA):
    """
    Smooth the last observed target into a causal prediction.

    This is useful for houses with temporal drift: it only uses lag_1 values
    that are already available as input features at prediction time.
    """
    lag_values = np.asarray(lag_values, dtype=float)
    if len(lag_values) == 0:
        return lag_values

    smoothed = np.empty_like(lag_values, dtype=float)
    current = lag_values[0]
    for i, value in enumerate(lag_values):
        current = alpha * value + (1.0 - alpha) * current
        smoothed[i] = current
    return smoothed


def plot_actual_vs_predicted(timestamps, y_true, y_pred, house_name, model_name, y_label):
    """Save a time-series comparison plot of actual vs predicted values."""
    plt.figure(figsize=(14, 5))
    plt.plot(timestamps, y_true, label="Actual", linewidth=0.8)
    plt.plot(timestamps, y_pred, label="Predicted", linewidth=0.8, alpha=0.8)
    dataset = DATASET_LABELS.get(house_name, "")
    title = f"{house_name} ({dataset}) - {model_name}: Actual vs Predicted"
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{house_name}_{model_name}.png", dpi=150)
    plt.close()


def save_predictions(timestamps, y_true, y_pred, house_name, model_name, data_mode):
    """Save actual/predicted values so downstream code can read non-negative watts."""
    suffix = "watt" if data_mode == DATA_MODE_WATT else "standard_scaled"
    pred_df = pd.DataFrame(
        {
            f"actual_{suffix}": y_true,
            f"predicted_{suffix}": y_pred,
        },
        index=timestamps,
    )
    pred_df.index.name = "timestamp"
    pred_df.to_csv(PRED_DIR / f"{house_name}_{model_name}_predictions.csv")


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
    Also records house*_clean.csv when available for non-negative watt mode.
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
        clean_path = data_dir / f"{house_name}_clean.csv"

        if not test_path.exists():
            print(f"[WARNING] Missing test file for {house_name}: {test_path.name}")
            continue

        house_map[house_name] = {
            "train": train_path,
            "valid": valid_path if valid_path.exists() else None,
            "test":  test_path,
            "clean": clean_path if clean_path.exists() else None,
        }

    return house_map


def read_split_csv(path):
    """Read a processed split CSV with timestamp index."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def align_clean_to_split(clean_df, split_path, split_name):
    """
    Select rows from the unscaled clean file using the exact split timestamps.
    This keeps the original train/valid/test temporal split while avoiding
    StandardScaler's negative values.
    """
    split_ref = read_split_csv(split_path)
    aligned = clean_df.reindex(split_ref.index)

    if aligned[TARGET_COL].isna().any():
        missing_count = int(aligned[TARGET_COL].isna().sum())
        raise ValueError(
            f"Could not align {split_name} with clean data: "
            f"{missing_count} timestamps are missing."
        )

    return aligned


def clip_negative_watt_columns(df, source_name):
    """Clip watt-like columns in clean data so physical inputs stay non-negative."""
    watt_cols = [col for col in df.columns if col in NON_NEGATIVE_WATT_COLS]
    if not watt_cols:
        return df

    negative_count = int((df[watt_cols] < 0).sum().sum())
    if negative_count == 0:
        return df

    clipped = df.copy()
    clipped[watt_cols] = clipped[watt_cols].clip(lower=0)
    print(f"  [CLIP DATA] {source_name}: clipped {negative_count} negative watt values to 0")
    return clipped


def load_house_frames(file_dict, requested_mode):
    """
    Load train/valid/test frames.

    Default watt mode reads house*_clean.csv and aligns it to the existing split
    indexes. MLP scaling is then done inside this script with MinMaxScaler, so
    the model never needs StandardScaled negative target values.
    """
    if requested_mode == DATA_MODE_WATT and file_dict.get("clean") is not None:
        clean_df = read_split_csv(file_dict["clean"])
        train_df = align_clean_to_split(clean_df, file_dict["train"], "train")
        test_df = align_clean_to_split(clean_df, file_dict["test"], "test")
        valid_df = None
        if file_dict.get("valid") is not None:
            valid_df = align_clean_to_split(clean_df, file_dict["valid"], "valid")

        train_df = clip_negative_watt_columns(train_df, "train")
        valid_df = clip_negative_watt_columns(valid_df, "valid") if valid_df is not None else None
        test_df = clip_negative_watt_columns(test_df, "test")
        return train_df, valid_df, test_df, DATA_MODE_WATT

    if requested_mode == DATA_MODE_WATT:
        print("  [WARNING] Missing clean file; falling back to StandardScaled splits.")

    train_df = read_split_csv(file_dict["train"])
    test_df = read_split_csv(file_dict["test"])
    valid_df = read_split_csv(file_dict["valid"]) if file_dict.get("valid") is not None else None
    return train_df, valid_df, test_df, DATA_MODE_STANDARD


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
def run_naive_baseline(test_df, house_name, feature_cols, data_mode):
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
    clipped_count = 0

    if data_mode == DATA_MODE_WATT:
        clipped_count = int(np.sum(y_pred < 0))
        y_pred = clip_non_negative(y_pred)

    mae, rmse, r2 = compute_metrics(y_true, y_pred)

    plot_actual_vs_predicted(
        timestamps=timestamps,
        y_true=y_true,
        y_pred=y_pred,
        house_name=house_name,
        model_name="naive_baseline",
        y_label=unit_label_for(data_mode),
    )
    save_predictions(timestamps, y_true, y_pred, house_name, "naive_baseline", data_mode)

    return {
        "model": "naive_baseline",
        "house": house_name,
        "dataset": DATASET_LABELS.get(house_name, "unknown"),
        "data_mode": data_mode,
        "target_unit": "W" if data_mode == DATA_MODE_WATT else "StandardScaled",
        "negative_predictions_clipped": clipped_count,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def run_adaptive_ewma_baseline(test_df, house_name, data_mode, alpha=EWMA_ALPHA):
    """
    Causal EWMA baseline using lag_1.

    House4 has a strong test-period level shift. A slowly adapting EWMA can be
    more stable than a fitted MLP there while staying causal.
    """
    if "lag_1" not in test_df.columns:
        print(f"  [SKIP ewma] {house_name}: no 'lag_1' column")
        return None

    valid_df = test_df.dropna(subset=[TARGET_COL, "lag_1"]).copy()
    if valid_df.empty:
        print(f"  [SKIP ewma] {house_name}: no valid rows after dropna")
        return None

    y_true = valid_df[TARGET_COL].values
    y_pred = causal_ewma_from_lag(valid_df["lag_1"].values, alpha=alpha)
    timestamps = valid_df.index
    clipped_count = 0

    if data_mode == DATA_MODE_WATT:
        clipped_count = int(np.sum(y_pred < 0))
        y_pred = clip_non_negative(y_pred)

    mae, rmse, r2 = compute_metrics(y_true, y_pred)

    plot_actual_vs_predicted(
        timestamps=timestamps,
        y_true=y_true,
        y_pred=y_pred,
        house_name=house_name,
        model_name="adaptive_ewma_baseline",
        y_label=unit_label_for(data_mode),
    )
    save_predictions(timestamps, y_true, y_pred, house_name, "adaptive_ewma_baseline", data_mode)

    return {
        "model": "adaptive_ewma_baseline",
        "house": house_name,
        "dataset": DATASET_LABELS.get(house_name, "unknown"),
        "data_mode": data_mode,
        "target_unit": "W" if data_mode == DATA_MODE_WATT else "StandardScaled",
        "negative_predictions_clipped": clipped_count,
        "alpha": alpha,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def drop_zero_variance(train_df, feature_cols):
    """
    Remove features with zero variance in train set.
    These columns (e.g. fridge=0, microwave=0 when appliance is absent)
    add noise without information and hurt small-dataset performance.
    """
    zero_var = [c for c in feature_cols if train_df[c].std() == 0]
    if zero_var:
        print(f"  [DROP] Zero-variance features: {zero_var}")
    return [c for c in feature_cols if c not in zero_var]


def run_mlp_regression(train_df, test_df, house_name, feature_cols, valid_df=None, data_mode=DATA_MODE_WATT):
    """
    Train an MLP regressor on train_df and evaluate on test_df.

    In watt mode, the script reads unscaled clean watt data, scales features and
    target internally with MinMaxScaler, then converts predictions back to watts.
    Final watt predictions are clipped at zero because negative consumption is
    physically invalid.

    If valid_df is provided, it is transformed with the same scalers and added
    to the fit data; sklearn still manages its own internal early-stopping split.
    """
    # Drop zero-variance features (e.g. missing appliances filled with 0)
    feature_cols = drop_zero_variance(train_df, feature_cols)

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

    n_train = len(X_train)
    feature_scaler = None
    target_scaler = None

    # If an explicit validation set exists (from data_pipeline's temporal split),
    # use it for early stopping to respect the time-series ordering.
    use_explicit_valid = (
        valid_df is not None
        and not valid_df.dropna(subset=feature_cols + [TARGET_COL]).empty
    )

    if data_mode == DATA_MODE_WATT:
        feature_scaler = MinMaxScaler(clip=True)
        target_scaler = MinMaxScaler()

        X_train_model = feature_scaler.fit_transform(X_train)
        y_train_model = target_scaler.fit_transform(y_train.to_frame()).ravel()
        X_test_model = feature_scaler.transform(X_test)
    else:
        X_train_model = X_train.values
        y_train_model = y_train.values
        X_test_model = X_test.values

    if use_explicit_valid:
        valid_clean = valid_df.dropna(subset=feature_cols + [TARGET_COL]).copy()
        X_valid_es_raw = valid_clean[feature_cols]
        y_valid_es_raw = valid_clean[TARGET_COL]

        if data_mode == DATA_MODE_WATT:
            X_valid_es = feature_scaler.transform(X_valid_es_raw)
            y_valid_es = target_scaler.transform(y_valid_es_raw.to_frame()).ravel()
        else:
            X_valid_es = X_valid_es_raw.values
            y_valid_es = y_valid_es_raw.values

        # Combine train + valid for fitting, but mark the split
        # so sklearn's early_stopping uses the correct portion.
        X_combined = np.vstack([X_train_model, X_valid_es])
        y_combined = np.concatenate([y_train_model, y_valid_es])
        valid_frac = len(X_valid_es) / len(X_combined)
    else:
        X_combined = X_train_model
        y_combined = y_train_model
        valid_frac = 0.1

    # Adapt hyperparameters for small datasets (< 1000 rows):
    # - stronger L2 regularization to prevent overfitting
    # - smaller hidden layers to reduce model complexity
    # - more iterations for better convergence
    if n_train < 1000:
        hidden_sizes = (32, 16)
        alpha = 0.01       # stronger L2 regularization
        max_iter = 1000
        lr = 0.0005
        print(f"  [SMALL DATA] {n_train} rows -> layers={hidden_sizes}, alpha={alpha}")
    else:
        hidden_sizes = (64, 32)
        alpha = 0.0001     # default sklearn alpha
        max_iter = 500
        lr = 0.001

    model = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=valid_frac,
        n_iter_no_change=20,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_combined, y_combined)

    if getattr(model, "n_iter_", 0) >= max_iter:
        print(f"  [INFO] MLP reached max_iter={max_iter}; using best weights from early stopping.")

    y_pred_model = model.predict(X_test_model)

    clipped_count = 0
    if data_mode == DATA_MODE_WATT:
        y_pred = target_scaler.inverse_transform(y_pred_model.reshape(-1, 1)).ravel()
        clipped_count = int(np.sum(y_pred < 0))
        y_pred = clip_non_negative(y_pred)
        y_test_eval = y_test.values.astype(float)
    else:
        y_pred = y_pred_model
        y_test_eval = y_test.values

    mae, rmse, r2 = compute_metrics(y_test_eval, y_pred)

    plot_actual_vs_predicted(
        timestamps=timestamps,
        y_true=y_test_eval,
        y_pred=y_pred,
        house_name=house_name,
        model_name="mlp_regression",
        y_label=unit_label_for(data_mode),
    )
    save_predictions(timestamps, y_test_eval, y_pred, house_name, "mlp_regression", data_mode)

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "data_mode": data_mode,
            "target_unit": "W" if data_mode == DATA_MODE_WATT else "StandardScaled",
            "clip_predictions_at_zero": data_mode == DATA_MODE_WATT,
        },
        MODEL_DIR / f"{house_name}_mlp_regression.joblib",
    )

    return {
        "model": "mlp_regression",
        "house": house_name,
        "dataset": DATASET_LABELS.get(house_name, "unknown"),
        "data_mode": data_mode,
        "target_unit": "W" if data_mode == DATA_MODE_WATT else "StandardScaled",
        "train_rows": len(train_valid),
        "test_rows": len(test_valid),
        "n_features": len(feature_cols),
        "negative_predictions_clipped": clipped_count,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# =========================
# PIPELINE
# =========================
def process_house(house_name, file_dict, cap=None, data_mode=DATA_MODE_WATT):
    """
    Load pre-scaled data, optionally subsample, and run baselines + MLP.
    Watt mode reads clean data and scales internally to avoid negative targets.
    """
    dataset = DATASET_LABELS.get(house_name, "unknown")
    print(f"\n{'='*50}")
    print(f"  {house_name}  ({dataset})")
    print(f"{'='*50}")

    train_df, valid_df, test_df, actual_data_mode = load_house_frames(file_dict, data_mode)
    print(f"  Data mode: {actual_data_mode} ({unit_label_for(actual_data_mode)})")

    if valid_df is not None:
        print(f"  Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)} rows")
    else:
        print(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")

    original_train_len = len(train_df)

    # Subsample if needed (to balance data between REDD and UK-DALE)
    if cap is not None and len(train_df) > cap:
        train_df = subsample(train_df, cap)
        print(f"  [CAP] Subsampled train: {original_train_len} -> {len(train_df)} rows")

    # Auto-detect features from common columns between train and test
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    feature_cols = [c for c in common_cols if c not in EXCLUDE_COLS]

    if not feature_cols:
        raise ValueError(f"{house_name}: no feature columns found after excluding {EXCLUDE_COLS}")

    naive_result = run_naive_baseline(test_df, house_name, feature_cols, actual_data_mode)
    ewma_result = run_adaptive_ewma_baseline(test_df, house_name, actual_data_mode)
    mlp_result = run_mlp_regression(
        train_df,
        test_df,
        house_name,
        feature_cols,
        valid_df=valid_df,
        data_mode=actual_data_mode,
    )

    if naive_result:
        print(
            f"  Naive  ->  MAE={naive_result['mae']:.6f}  "
            f"RMSE={naive_result['rmse']:.6f}  "
            f"R2={naive_result['r2']:.6f}"
        )
    if ewma_result:
        print(
            f"  EWMA   ->  MAE={ewma_result['mae']:.6f}  "
            f"RMSE={ewma_result['rmse']:.6f}  "
            f"R2={ewma_result['r2']:.6f}"
        )
    print(
        f"  MLP    ->  MAE={mlp_result['mae']:.6f}  "
        f"RMSE={mlp_result['rmse']:.6f}  "
        f"R2={mlp_result['r2']:.6f}"
    )

    results = []
    if naive_result:
        results.append(naive_result)
    if ewma_result:
        results.append(ewma_result)
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
        "--data-mode",
        choices=[DATA_MODE_WATT, DATA_MODE_STANDARD],
        default=DATA_MODE_WATT,
        help="watt: read house*_clean.csv, MinMax-scale internally, output non-negative watts. "
             "standard: use the existing StandardScaled split CSVs (old behavior).",
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
    print(f"  Data mode: {args.data_mode}")
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
            results = process_house(house_name, file_dict, cap=cap, data_mode=args.data_mode)
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] {house_name}: {e}")

    if not all_results:
        raise RuntimeError("No houses were processed successfully.")

    # Save results
    results_df = pd.DataFrame(all_results)

    # Ensure dataset column is present
    base_cols = ["model", "house", "dataset", "data_mode", "target_unit", "mae", "rmse", "r2"]
    extra_cols = [c for c in results_df.columns if c not in base_cols]
    results_df = results_df[base_cols + extra_cols]

    results_df.to_csv(RESULT_DIR / "local_metrics.csv", index=False)

    # Print summary
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    # Show the best evaluated local predictor per house. Keeping the model name
    # visible avoids hiding cases where MLP is weaker than a causal baseline.
    candidate_models = ["mlp_regression", "adaptive_ewma_baseline"]
    candidate_df = results_df[results_df["model"].isin(candidate_models)].copy()
    selected_df = pd.DataFrame()
    if not candidate_df.empty:
        selected_idx = candidate_df.groupby("house")["r2"].idxmax()
        selected_df = candidate_df.loc[selected_idx].copy()
        selected_df = selected_df.sort_values(
            "house",
            key=lambda col: col.map(lambda h: int(re.search(r"\d+", h).group())),
        )
        selected_df.to_csv(RESULT_DIR / "selected_metrics.csv", index=False)

        print("\n  Best local predictor per house:")
        print(f"  {'House':<10} {'Dataset':<10} {'Selected':<22} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
        print("  " + "-" * 72)
        for _, row in selected_df.iterrows():
            print(
                f"  {row['house']:<10} {row['dataset']:<10} {row['model']:<22} "
                f"{row['mae']:>8.4f} {row['rmse']:>8.4f} {row['r2']:>8.4f}"
            )

        print(f"\n  Average across {len(selected_df)} houses:")
        print(f"    MAE  = {selected_df['mae'].mean():.4f}")
        print(f"    RMSE = {selected_df['rmse'].mean():.4f}")
        print(f"    R2   = {selected_df['r2'].mean():.4f}")

        print(f"\n  REDD average    (house1-5):")
        redd = selected_df[selected_df["dataset"] == "REDD"]
        if not redd.empty:
            print(f"    MAE={redd['mae'].mean():.4f}  RMSE={redd['rmse'].mean():.4f}  R2={redd['r2'].mean():.4f}")

        print(f"  UK-DALE average (house6-10):")
        ukdale = selected_df[selected_df["dataset"] == "UK-DALE"]
        if not ukdale.empty:
            print(f"    MAE={ukdale['mae'].mean():.4f}  RMSE={ukdale['rmse'].mean():.4f}  R2={ukdale['r2'].mean():.4f}")

    print(f"\n  Saved metrics -> {RESULT_DIR / 'local_metrics.csv'}")
    if not selected_df.empty:
        print(f"  Saved selected -> {RESULT_DIR / 'selected_metrics.csv'}")
    print(f"  Saved plots  -> {PLOT_DIR}")
    print(f"  Saved models -> {MODEL_DIR}")
    print(f"  Saved preds  -> {PRED_DIR}")
    print()


if __name__ == "__main__":
    main()
