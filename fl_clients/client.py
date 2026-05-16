import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


TARGET_COL = "main"
SPLIT_COL = "split"
HOUSE_ORDER = ["house1", "house2", "house3", "house4", "house5",
               "house6", "house7", "house8", "house9", "house10"]

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def split_time_series(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


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


def load_house_xy(project_root: Path, house_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    # Ưu tiên dùng dữ liệu đã tách train/test sẵn từ data pipeline:
    # data/processed/<house>_train.csv và data/processed/<house>_test.csv
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
            raise ValueError(f"{csv_path.name} does not contain target column '{TARGET_COL}'")

        if SPLIT_COL in df.columns:
            train_df, test_df = split_from_column(df)
        else:
            train_df, test_df = split_time_series(df, train_ratio=0.8)

    if TARGET_COL not in train_df.columns or TARGET_COL not in test_df.columns:
        raise ValueError(f"Split data for {house_name} is missing target column '{TARGET_COL}' (source={source_name})")

    # Đồng bộ cột feature giữa train/test (an toàn khi thứ tự cột khác nhau)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = [c for c in train_df.columns if c in test_cols]

    if TARGET_COL not in common_cols:
        raise ValueError(f"Target column '{TARGET_COL}' not found in common columns (source={source_name})")

    feature_cols = [c for c in common_cols if c not in {TARGET_COL, SPLIT_COL}]
    if not feature_cols:
        raise ValueError(f"No feature columns found for {house_name} (source={source_name})")

    # Chỉ giữ các cột chung để tránh lệch cột âm thầm
    train_df = train_df[feature_cols + [TARGET_COL]].copy()
    test_df = test_df[feature_cols + [TARGET_COL]].copy()

    train_valid = train_df.dropna(subset=feature_cols + [TARGET_COL]).copy()
    test_valid = test_df.dropna(subset=feature_cols + [TARGET_COL]).copy()

    x_train = train_valid[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_valid[TARGET_COL].to_numpy(dtype=np.float64)

    x_test = test_valid[feature_cols].to_numpy(dtype=np.float64)
    y_test = test_valid[TARGET_COL].to_numpy(dtype=np.float64)

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError(
            f"Not enough valid train/test rows after dropna for {house_name}. "
            f"train={len(x_train)}, test={len(x_test)}"
        )

    return x_train, y_train, x_test, y_test, feature_cols


class HouseMlpClient(fl.client.NumPyClient):
    """Flower client for one house, using MLPRegressor as the local model."""

    def __init__(self, house_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, feature_cols: List[str]) -> None:
        self.house_name = house_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.feature_cols = feature_cols

        self.n_features = x_train.shape[1]
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            warm_start=True,
            shuffle=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        self._initialize_model()

    def _initialize_model(self) -> None:
        # Fit tối thiểu 1 lần để sklearn tạo cấu trúc weights nội bộ
        # Dùng full local train để khởi tạo ổn định hơn giữa các house.
        # Nếu chỉ bootstrap ít mẫu đầu, một số house có thể bị bias mạnh từ
        # đoạn đầu chuỗi thời gian và kéo metric xấu trong các vòng FL đầu.
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

        train_metrics = compute_metrics(self.y_train, y_pred_train)
        train_metrics.update(
            {
                "house": self.house_name,
                "num_features": float(self.n_features),
                "model": "mlp_regression",
            }
        )

        return self._get_parameters(), len(self.x_train), train_metrics

    def evaluate(self, parameters, config):
        self._set_parameters(parameters)

        y_pred = self.model.predict(self.x_test)
        metrics = compute_metrics(self.y_test, y_pred)
        metrics.update(
            {
                "house": self.house_name,
                "num_features": float(self.n_features),
                "model": "mlp_regression",
            }
        )

        # Với regression, dùng RMSE làm loss khi evaluate
        loss = metrics["rmse"]
        return float(loss), len(self.x_test), metrics


def resolve_house_name(client_index: int) -> str:
    if client_index < 0 or client_index >= len(HOUSE_ORDER):
        raise ValueError(
            f"client_index must be in [0, {len(HOUSE_ORDER)-1}], got {client_index}"
        )
    return HOUSE_ORDER[client_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower client for one house in the smart-home task")
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
        help="Client-to-house mapping: 0->house1, 1->house2, ..., 9->house10",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load data and run one local fit/evaluate pass without connecting to server",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    house_name = args.house if args.house is not None else resolve_house_name(args.client_index)

    x_train, y_train, x_test, y_test, feature_cols = load_house_xy(project_root, house_name)

    print(f"[CLIENT] Running house={house_name}")
    print(f"[CLIENT] Data columns={feature_cols + [TARGET_COL]}")
    print(f"[CLIENT] Train shape: x={x_train.shape}, y={y_train.shape}")
    print(f"[CLIENT] Test shape : x={x_test.shape}, y={y_test.shape}")

    client = HouseMlpClient(
        house_name=house_name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_cols=feature_cols,
    )

    if args.dry_run:
        init_params = client.get_parameters(config={})
        updated_params, n_train, train_metrics = client.fit(init_params, config={})
        loss, n_test, test_metrics = client.evaluate(updated_params, config={})

        print("[DRY-RUN] Train samples=", n_train)
        print("[DRY-RUN] Train metrics=", train_metrics)
        print("[DRY-RUN] Test samples=", n_test)
        print("[DRY-RUN] Loss eval (rmse)=", loss)
        print("[DRY-RUN] Test metrics=", test_metrics)
        return

    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
