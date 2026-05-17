import argparse
import csv
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


TARGET_COL = "main"
SPLIT_COL = "split"
HOUSE_ORDER = ["house1", "house2", "house3", "house4", "house5",
               "house6", "house7", "house8", "house9", "house10"]

# Columns that are physically non-negative watt values
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

# Data modes (aligned with models/train_local_mlp.py)
DATA_MODE_WATT = "watt"           # read clean.csv, MinMaxScale internally, output non-neg watts
DATA_MODE_STANDARD = "standard"   # read pre-scaled train/test split CSVs as-is

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ============================================================
# DATA LOADING
# ============================================================
def split_time_series(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def split_from_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_values = df[SPLIT_COL].astype(str).str.strip().str.lower()
    train_mask = split_values.isin({"train", "tr", "0", "false"})
    test_mask = split_values.isin({"test", "te", "1", "true", "val", "validation"})

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"{SPLIT_COL} column exists but cannot determine train/test groups. "
            f"train={int(train_mask.sum())}, test={int(test_mask.sum())}"
        )

    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def find_house_file(project_root: Path, house_name: str) -> Path:
    candidates = [
        project_root / "data" / "data" / "processed" / f"{house_name}_clean.csv",
        project_root / "data" / "processed" / f"{house_name}_clean.csv",
        project_root / "data" / "data" / "processed" / f"{house_name}_hourly_clean.csv",
        project_root / "data" / "processed" / f"{house_name}_hourly_clean.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find clean data file for {house_name}. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def find_house_split_files(project_root: Path, house_name: str) -> Tuple[Path, Path]:
    """Prefer explicit train/test split files if present."""
    train_candidates = [
        project_root / "data" / "data" / "processed" / f"{house_name}_train.csv",
        project_root / "data" / "processed" / f"{house_name}_train.csv",
    ]
    test_candidates = [
        project_root / "data" / "data" / "processed" / f"{house_name}_test.csv",
        project_root / "data" / "processed" / f"{house_name}_test.csv",
    ]

    train_path = next((p for p in train_candidates if p.exists()), None)
    test_path = next((p for p in test_candidates if p.exists()), None)

    if train_path is not None and test_path is not None:
        return train_path, test_path

    missing = []
    if train_path is None:
        missing.append("train")
    if test_path is None:
        missing.append("test")

    raise FileNotFoundError(
        f"Missing {', '.join(missing)} split file(s) for {house_name}. "
        f"Checked train: {', '.join(str(p) for p in train_candidates)}; "
        f"test: {', '.join(str(p) for p in test_candidates)}"
    )


def find_clean_file(project_root: Path, house_name: str) -> Optional[Path]:
    """Locate the unscaled clean CSV (for watt mode)."""
    candidates = [
        project_root / "data" / "data" / "processed" / f"{house_name}_clean.csv",
        project_root / "data" / "processed" / f"{house_name}_clean.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def clip_non_negative(values: np.ndarray) -> np.ndarray:
    """Clamp physically impossible negative watt predictions to zero."""
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def clip_negative_watt_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clip watt-like columns so physical inputs stay non-negative."""
    watt_cols = [col for col in df.columns if col in NON_NEGATIVE_WATT_COLS]
    if not watt_cols:
        return df
    clipped = df.copy()
    clipped[watt_cols] = clipped[watt_cols].clip(lower=0)
    return clipped


def subsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Evenly-spaced subsample to preserve time-series distribution."""
    if max_rows is None or len(df) <= max_rows:
        return df
    indices = np.linspace(0, len(df) - 1, max_rows, dtype=int)
    return df.iloc[indices].copy()


def drop_zero_variance(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """Remove features with zero variance (e.g. missing appliances filled with 0)."""
    zero_var = [c for c in feature_cols if df[c].std() == 0]
    if zero_var:
        print(f"[CLIENT] Dropping zero-variance features: {zero_var}")
    return [c for c in feature_cols if c not in zero_var]


def load_house_xy(
    project_root: Path,
    house_name: str,
    data_mode: str = DATA_MODE_WATT,
    cap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[MinMaxScaler], Optional[MinMaxScaler]]:
    """
    Load (x_train, y_train, x_test, y_test, feature_cols, feature_scaler, target_scaler).

    data_mode='watt':
        Reads house*_clean.csv (original watts), aligns to the existing train/test
        split timestamps, then fits MinMaxScaler *only on train* — no negative targets.

    data_mode='standard':
        Reads the pre-scaled house*_train.csv / house*_test.csv as-is (StandardScaled).
        Returns (None, None) for the two scaler slots.

    Both modes return raw numpy arrays; scaling for watt mode is handled here so the
    MLPRegressor always sees [0,1]-normalised inputs and targets.
    """
    feature_scaler: Optional[MinMaxScaler] = None
    target_scaler: Optional[MinMaxScaler] = None

    # ------------------------------------------------------------------
    # WATT MODE: read clean CSV, align to split timestamps, MinMaxScale
    # ------------------------------------------------------------------
    if data_mode == DATA_MODE_WATT:
        clean_path = find_clean_file(project_root, house_name)

        if clean_path is not None:
            # Load the unscaled clean dataframe
            clean_df = pd.read_csv(clean_path, index_col=0, parse_dates=True)
            clean_df = clip_negative_watt_columns(clean_df)

            # Try to get split timestamp references from pre-split CSVs
            try:
                train_ref_path, test_ref_path = find_house_split_files(project_root, house_name)
                train_ref = pd.read_csv(train_ref_path, index_col=0, parse_dates=True)
                test_ref = pd.read_csv(test_ref_path, index_col=0, parse_dates=True)
                train_df = clean_df.reindex(train_ref.index)
                test_df = clean_df.reindex(test_ref.index)
                source_name = f"{clean_path.name} aligned to {train_ref_path.name}"
            except FileNotFoundError:
                # Fallback: time-based 80/20 split on clean data
                train_df, test_df = split_time_series(clean_df, train_ratio=0.8)
                source_name = f"{clean_path.name} (80/20 split)"

            if TARGET_COL not in train_df.columns or TARGET_COL not in test_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COL}' missing in aligned data for {house_name}"
                )

        else:
            # No clean file found → fall back to standard mode silently
            print(
                f"[CLIENT] {house_name}: No clean.csv found, falling back to standard mode."
            )
            data_mode = DATA_MODE_STANDARD

    # ------------------------------------------------------------------
    # STANDARD MODE: read pre-scaled train/test CSVs directly
    # ------------------------------------------------------------------
    if data_mode == DATA_MODE_STANDARD:
        try:
            train_path, test_path = find_house_split_files(project_root, house_name)
            train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
            test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
            source_name = f"{train_path.name} + {test_path.name}"
        except FileNotFoundError:
            csv_path = find_house_file(project_root, house_name)
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            source_name = csv_path.name

            if TARGET_COL not in df.columns:
                raise ValueError(
                    f"{csv_path.name} does not contain target column '{TARGET_COL}'"
                )

            if SPLIT_COL in df.columns:
                train_df, test_df = split_from_column(df)
            else:
                train_df, test_df = split_time_series(df, train_ratio=0.8)

    # ------------------------------------------------------------------
    # COMMON: validate, align columns, dropna
    # ------------------------------------------------------------------
    if TARGET_COL not in train_df.columns or TARGET_COL not in test_df.columns:
        raise ValueError(
            f"Split data for {house_name} is missing target column '{TARGET_COL}' "
            f"(source={source_name})"
        )

    common_cols = [c for c in train_df.columns if c in set(test_df.columns)]
    if TARGET_COL not in common_cols:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in common columns (source={source_name})"
        )

    feature_cols = [c for c in common_cols if c not in {TARGET_COL, SPLIT_COL}]
    if not feature_cols:
        raise ValueError(
            f"No feature columns found for {house_name} (source={source_name})"
        )

    train_df = train_df[feature_cols + [TARGET_COL]].copy()
    test_df = test_df[feature_cols + [TARGET_COL]].copy()

    train_valid = train_df.dropna(subset=feature_cols + [TARGET_COL]).copy()
    test_valid = test_df.dropna(subset=feature_cols + [TARGET_COL]).copy()

    if len(train_valid) == 0 or len(test_valid) == 0:
        raise ValueError(
            f"Not enough valid train/test rows after dropna for {house_name}. "
            f"train={len(train_valid)}, test={len(test_valid)}"
        )

    # Drop zero-variance features before scaling
    feature_cols = drop_zero_variance(train_valid, feature_cols)

    # Optionally cap training rows (reduces REDD/UK-DALE imbalance).
    # cap=None or cap=0 -> no limit; cap>0 -> subsample to that many rows.
    effective_cap = cap if (cap is not None and cap > 0) else None
    if effective_cap is not None and len(train_valid) > effective_cap:
        original_len = len(train_valid)
        train_valid = subsample(train_valid, effective_cap)
        print(
            f"[CLIENT] {house_name}: training rows capped {original_len} -> {len(train_valid)}"
        )

    # ------------------------------------------------------------------
    # WATT MODE: fit MinMaxScaler on train, transform both splits
    # ------------------------------------------------------------------
    if data_mode == DATA_MODE_WATT:
        feature_scaler = MinMaxScaler(clip=True)
        target_scaler = MinMaxScaler()

        x_train_raw = train_valid[feature_cols]
        y_train_raw = train_valid[[TARGET_COL]]
        x_test_raw = test_valid[feature_cols]
        y_test_raw = test_valid[[TARGET_COL]]

        x_train = feature_scaler.fit_transform(x_train_raw).astype(np.float64)
        y_train = target_scaler.fit_transform(y_train_raw).ravel().astype(np.float64)
        x_test = feature_scaler.transform(x_test_raw).astype(np.float64)
        y_test_scaled = target_scaler.transform(y_test_raw).ravel().astype(np.float64)

        return x_train, y_train, x_test, y_test_scaled, feature_cols, feature_scaler, target_scaler

    # ------------------------------------------------------------------
    # STANDARD MODE: return as-is
    # ------------------------------------------------------------------
    x_train = train_valid[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_valid[TARGET_COL].to_numpy(dtype=np.float64)
    x_test = test_valid[feature_cols].to_numpy(dtype=np.float64)
    y_test = test_valid[TARGET_COL].to_numpy(dtype=np.float64)

    return x_train, y_train, x_test, y_test, feature_cols, None, None


# ============================================================
# FLOWER CLIENT
# ============================================================
class HouseMlpClient(fl.client.NumPyClient):
    """Flower client for one house, using MLPRegressor as the local model.

    Supports both data_mode='watt' (MinMaxScaled, non-negative outputs) and
    data_mode='standard' (pre-scaled StandardScaled data).
    """

    def __init__(
        self,
        house_name: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        feature_cols: List[str],
        data_mode: str = DATA_MODE_WATT,
        target_scaler: Optional[MinMaxScaler] = None,
    ) -> None:
        self.house_name = house_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.feature_cols = feature_cols
        self.data_mode = data_mode
        self.target_scaler = target_scaler

        self.n_features = x_train.shape[1]
        n_train = len(x_train)

        # Adapt architecture based on dataset size.
        # max_iter=100 per FL round is intentional: with warm_start=True the model
        # accumulates ~num_rounds*100 effective iterations across all FL rounds,
        # keeping each local fit fast (per team decision to speed up FL training).
        if n_train < 1000:
            hidden_sizes = (32, 16)
            alpha = 0.01
            max_iter = 100
            lr = 0.0005
        else:
            hidden_sizes = (64, 32)
            alpha = 0.0001
            max_iter = 100
            lr = 0.001

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=lr,
            alpha=alpha,
            max_iter=max_iter,
            warm_start=True,
            shuffle=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        self._initialize_model()

        # ---- Per-house FL round logging ----
        # Track FL round count locally (Flower does not pass round number by default).
        self._fl_round = 0
        log_dir = Path(__file__).resolve().parent.parent / "results" / "fl_run_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._fl_log_path = log_dir / f"{self.house_name}_fl_rounds.csv"
        # Initialise / clear the log file so each FL run starts fresh.
        with open(self._fl_log_path, "w", newline="", encoding="utf-8") as _f:
            _w = csv.writer(_f)
            _w.writerow([
                "round", "loss_rmse", "mae", "rmse", "r2",
                "train_rows", "test_rows", "data_mode", "timestamp",
            ])

    def _initialize_model(self) -> None:
        """Fit once to initialise sklearn's internal weight structure."""
        self.model.fit(self.x_train, self.y_train)

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        n_layers = len(self.model.coefs_)
        expected_param_count = n_layers * 2

        if len(parameters) != expected_param_count:
            raise ValueError(
                f"Parameter count mismatch for {self.house_name}: "
                f"received={len(parameters)}, expected={expected_param_count}"
            )

        new_coefs = [np.array(parameters[i], dtype=np.float64) for i in range(n_layers)]
        new_intercepts = [
            np.array(parameters[n_layers + i], dtype=np.float64) for i in range(n_layers)
        ]

        for i, (old_w, new_w) in enumerate(zip(self.model.coefs_, new_coefs)):
            if old_w.shape != new_w.shape:
                raise ValueError(
                    f"Weight shape mismatch at layer {i} for {self.house_name}: "
                    f"received={new_w.shape}, expected={old_w.shape}"
                )

        for i, (old_b, new_b) in enumerate(zip(self.model.intercepts_, new_intercepts)):
            if old_b.shape != new_b.shape:
                raise ValueError(
                    f"Bias shape mismatch at layer {i} for {self.house_name}: "
                    f"received={new_b.shape}, expected={old_b.shape}"
                )

        self.model.coefs_ = new_coefs
        self.model.intercepts_ = new_intercepts
        self.model.n_features_in_ = self.n_features

        # ---- WARM_START + EARLY_STOPPING FIX ----
        # sklearn preserves _no_improvement_count and best_loss_ across warm_start
        # fit() calls. After receiving new aggregated weights from the server, the
        # stale counter could cause local training to terminate prematurely (e.g.
        # stopping after only 5 iterations if the counter was at 15 last round).
        # Resetting both ensures each FL round runs a full max_iter=100 iterations
        # (subject to the normal patience window starting fresh).
        if hasattr(self.model, "_no_improvement_count"):
            self.model._no_improvement_count = 0
        if hasattr(self.model, "best_loss_"):
            self.model.best_loss_ = np.inf
        if hasattr(self.model, "best_validation_score_"):
            self.model.best_validation_score_ = -np.inf

    def _get_parameters(self) -> List[np.ndarray]:
        weights = [np.array(w, dtype=np.float64) for w in self.model.coefs_]
        biases = [np.array(b, dtype=np.float64) for b in self.model.intercepts_]
        return weights + biases

    def get_parameters(self, config):
        return self._get_parameters()

    def fit(self, parameters, config):
        self._set_parameters(parameters)

        self.model.fit(self.x_train, self.y_train)
        y_pred_train = self.model.predict(self.x_train)

        # Inverse-transform for reporting in original units when in watt mode
        if self.data_mode == DATA_MODE_WATT and self.target_scaler is not None:
            y_true_report = self.target_scaler.inverse_transform(
                self.y_train.reshape(-1, 1)
            ).ravel()
            y_pred_report = clip_non_negative(
                self.target_scaler.inverse_transform(
                    y_pred_train.reshape(-1, 1)
                ).ravel()
            )
        else:
            y_true_report = self.y_train
            y_pred_report = y_pred_train

        train_metrics = compute_metrics(y_true_report, y_pred_report)
        train_metrics.update(
            {
                "house": self.house_name,
                "num_features": float(self.n_features),
                "model": "mlp_regression",
                "data_mode": self.data_mode,
            }
        )

        return self._get_parameters(), len(self.x_train), train_metrics

    def evaluate(self, parameters, config):
        self._set_parameters(parameters)

        y_pred = self.model.predict(self.x_test)

        # Inverse-transform for metrics in original units when in watt mode
        if self.data_mode == DATA_MODE_WATT and self.target_scaler is not None:
            y_true_eval = self.target_scaler.inverse_transform(
                self.y_test.reshape(-1, 1)
            ).ravel()
            y_pred_eval = clip_non_negative(
                self.target_scaler.inverse_transform(
                    y_pred.reshape(-1, 1)
                ).ravel()
            )
        else:
            y_true_eval = self.y_test
            y_pred_eval = y_pred

        metrics = compute_metrics(y_true_eval, y_pred_eval)
        metrics.update(
            {
                "house": self.house_name,
                "num_features": float(self.n_features),
                "model": "mlp_regression",
                "data_mode": self.data_mode,
            }
        )

        # Loss metric for Flower: use RMSE
        loss = metrics["rmse"]

        # ---- Per-house per-round logging ----
        self._fl_round += 1
        self._log_fl_round(self._fl_round, loss, metrics)

        return float(loss), len(self.x_test), metrics

    def _log_fl_round(self, round_num: int, loss: float, metrics: Dict[str, float]) -> None:
        """Append one row to results/fl_run_logs/{house}_fl_rounds.csv.

        Called after every evaluate() — both during actual FL training and dry-run.
        This creates per-house round-by-round progression data for the report and
        the dashboard handoff to Duy.
        """
        with open(self._fl_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                round(loss, 6),
                round(metrics.get("mae", 0.0), 6),
                round(metrics.get("rmse", 0.0), 6),
                round(metrics.get("r2", 0.0), 6),
                len(self.x_train),
                len(self.x_test),
                self.data_mode,
                datetime.now().isoformat(timespec="seconds"),
            ])


# ============================================================
# CLI
# ============================================================
def resolve_house_name(client_index: int) -> str:
    if client_index < 0 or client_index >= len(HOUSE_ORDER):
        raise ValueError(
            f"client_index must be in [0, {len(HOUSE_ORDER)-1}], got {client_index}"
        )
    return HOUSE_ORDER[client_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flower client for one house in the smart-home FL task"
    )
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--house",
        type=str,
        choices=HOUSE_ORDER,
        help="House name to run as client",
    )
    group.add_argument(
        "--client-index",
        type=int,
        choices=range(10),
        help="Client-to-house mapping: 0→house1, 1→house2, ..., 9→house10",
    )

    parser.add_argument(
        "--data-mode",
        choices=[DATA_MODE_WATT, DATA_MODE_STANDARD],
        default=DATA_MODE_WATT,
        help=(
            "watt: read house*_clean.csv, MinMaxScale internally, output non-neg watts. "
            "standard: use the pre-scaled StandardScaled train/test CSVs (old behavior)."
        ),
    )

    parser.add_argument(
        "--cap",
        type=int,
        default=1000,
        help=(
            "Cap training rows per house (default: 1000). "
            "Balances REDD (562-812 rows) vs UK-DALE (up to 105k rows) so FL is fast and fair. "
            "Use the same value as in models/train_local_mlp.py for fair comparison. "
            "Set to 0 to disable capping (use all rows)."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load data and run one local fit/evaluate pass without connecting to server",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    house_name = args.house if args.house is not None else resolve_house_name(args.client_index)

    print(f"[CLIENT] House     = {house_name}")
    print(f"[CLIENT] Data mode = {args.data_mode}")
    cap_effective = args.cap if (args.cap is not None and args.cap > 0) else None
    if cap_effective is not None:
        print(f"[CLIENT] Cap       = {cap_effective} training rows (balancing REDD/UK-DALE)")
    else:
        print(f"[CLIENT] Cap       = disabled (using all rows)")

    x_train, y_train, x_test, y_test, feature_cols, feature_scaler, target_scaler = \
        load_house_xy(project_root, house_name, data_mode=args.data_mode, cap=args.cap)

    print(f"[CLIENT] Feature cols : {feature_cols}")
    print(f"[CLIENT] Train shape  : x={x_train.shape}, y={y_train.shape}")
    print(f"[CLIENT] Test shape   : x={x_test.shape}, y={y_test.shape}")

    client = HouseMlpClient(
        house_name=house_name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_cols=feature_cols,
        data_mode=args.data_mode,
        target_scaler=target_scaler,
    )

    if args.dry_run:
        init_params = client.get_parameters(config={})
        updated_params, n_train, train_metrics = client.fit(init_params, config={})
        loss, n_test, test_metrics = client.evaluate(updated_params, config={})

        print("[DRY-RUN] Train samples =", n_train)
        print("[DRY-RUN] Train metrics =", train_metrics)
        print("[DRY-RUN] Test samples  =", n_test)
        print("[DRY-RUN] Loss (rmse)   =", loss)
        print("[DRY-RUN] Test metrics  =", test_metrics)
        print(f"[DRY-RUN] Per-house log -> {client._fl_log_path}")

        # Save dry-run summary to results/fl_client_test_results.csv.
        # Always overwrite the file header when the format has changed.
        EXPECTED_HEADER = [
            "timestamp", "house", "data_mode", "cap",
            "train_rows", "test_rows",
            "train_mae", "train_rmse", "train_r2",
            "test_loss", "test_mae", "test_rmse", "test_r2",
        ]

        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        out_path = results_dir / "fl_client_test_results.csv"

        # Check if the file has the expected header; if not, start fresh.
        header_ok = False
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as _f:
                first_line = _f.readline().strip()
            header_ok = (first_line == ",".join(EXPECTED_HEADER))

        with open(out_path, "a" if header_ok else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not header_ok:
                writer.writerow(EXPECTED_HEADER)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                house_name,
                args.data_mode,
                cap_effective if cap_effective else 0,
                n_train,
                n_test,
                train_metrics.get("mae", ""),
                train_metrics.get("rmse", ""),
                train_metrics.get("r2", ""),
                loss,
                test_metrics.get("mae", ""),
                test_metrics.get("rmse", ""),
                test_metrics.get("r2", ""),
            ])
        print(f"[DRY-RUN] Summary appended to {out_path}")
        return

    # Use the non-deprecated Flower API: .to_client() wraps NumPyClient.
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
