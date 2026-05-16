# Models — Local MLP Baseline Training

## Mục đích

Thư mục `models/` chứa script huấn luyện và đánh giá **local MLP baseline** cho bài toán dự đoán tiêu thụ điện hộ gia đình.  
Kết quả local này là **baseline** để so sánh với **mô hình toàn cục** được huấn luyện bằng Federated Learning.

## Kiến trúc tổng quan 

```
┌─────────────────────────────────────────────────────────┐
│           Decentralized Federated Learning              │
│                                                         │
│  ┌──────────┐  ┌──────────┐       ┌───────────┐        │
│  │ Client 1 │  │ Client 2 │  ...  │ Client 10 │        │
│  │ (house1) │  │ (house2) │       │ (house10) │        │
│  │  MLP     │  │  MLP     │       │  MLP      │        │
│  └────┬─────┘  └────┬─────┘       └─────┬─────┘        │
│       │              │                   │              │
│       └──────────────┼───────────────────┘              │
│                      ▼                                  │
│              ┌───────────────┐                          │
│              │  FL Server    │                          │
│              │  (FedAvg)     │                          │
│              │  Aggregate    │                          │
│              │  weights      │                          │
│              └───────┬───────┘                          │
│                      │                                  │
│                      ▼                                  │
│              ┌───────────────┐                          │
│              │ Global Model  │  ← Dùng để dự đoán      │
│              │  (converged)  │     khi hội tụ           │
│              └───────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

## Script: `train_local_mlp.py`

### Chức năng

- Load **10 house datasets** đã xử lý từ `data/processed/`
- Chạy **2 baseline** cho mỗi house:
  - **Naive baseline**: dự đoán bằng giá trị giờ trước (`lag_1`)
  - **MLP Regression**: huấn luyện MLP local cho từng house
- Tính metrics: **MAE**, **RMSE**, **R²**
- Xuất biểu đồ actual vs predicted cho mỗi house
- In bảng **data size summary** để thấy rõ imbalance giữa REDD và UK-DALE

### Cách chạy

```bash
# Train tất cả 10 houses (không cap data)
python models/train_local_mlp.py

# Train với data được cân bằng (cap tại median)
python models/train_local_mlp.py --balanced

# Train với cap cụ thể (ví dụ 5000 rows/house)
python models/train_local_mlp.py --cap 5000
```

### Tham số CLI

| Tham số      | Mô tả                                                                 |
|-------------|------------------------------------------------------------------------|
| `--balanced` | Cap training rows mỗi house tại **median** của tất cả houses          |
| `--cap N`   | Cap training rows mỗi house tại đúng **N** rows                       |

### Output

- `models/resultmlp/local_metrics.csv` — bảng metrics đầy đủ
- `models/resultmlp/plots/` — biểu đồ actual vs predicted

### MLP Architecture

```
Input (11 features) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1)
```

- Optimizer: Adam (lr=0.001)
- Early stopping: patience=20, validation_fraction=10%
- Max iterations: 500

---

## Dataset

### 10 Houses

| House    | Dataset  | Nguồn gốc                    |
|----------|----------|-------------------------------|
| house1   | REDD     | REDD house1                   |
| house2   | REDD     | REDD house2                   |
| house3   | REDD     | REDD house3                   |
| house4   | REDD     | REDD house4                   |
| house5   | REDD     | REDD house6 (house5 thiếu data) |
| house6   | UK-DALE  | UK-DALE house_1               |
| house7   | UK-DALE  | UK-DALE house_2               |
| house8   | UK-DALE  | UK-DALE house_3               |
| house9   | UK-DALE  | UK-DALE house_4               |
| house10  | UK-DALE  | UK-DALE house_5               |

### ⚠️ Data Imbalance

UK-DALE có lượng data **lớn hơn rất nhiều** so với REDD:

| House    | Dataset  | Approx Train Rows | Ratio vs Min |
|----------|----------|-------------------|--------------|
| house1   | REDD     | ~1,100            | 1.0x         |
| house2   | REDD     | ~1,050            | ~1x          |
| house3   | REDD     | ~770              | 0.7x         |
| house4   | REDD     | ~840              | 0.8x         |
| house5   | REDD     | ~1,100            | 1.0x         |
| house6   | UK-DALE  | **~135,000**      | **~123x** ⚠️ |
| house7   | UK-DALE  | ~18,000           | ~16x         |
| house8   | UK-DALE  | ~2,900            | ~2.7x        |
| house9   | UK-DALE  | ~13,300           | ~12x         |
| house10  | UK-DALE  | ~12,500           | ~11x         |

**Hệ quả**: Nếu FL server dùng `weighted average` theo `num_examples` (mặc định của FedAvg), model toàn cục sẽ gần như chỉ học từ house6 UK-DALE.

### Target & Features

- **Target**: `main` (tổng tiêu thụ điện hộ gia đình, đã chuẩn hóa)
- **Features** (auto-detect từ data, thường gồm):
  - Appliance: `dish washer`, `electric stove`, `fridge`, `microwave`, `washer dryer`
  - Time: `hour_sin`, `hour_cos`, `day_sin`, `day_cos`
  - Lag: `lag_1`, `rolling_mean`

---

## 🔔 Lưu ý cho các thư mục khác

### 1. `fl_clients/` — Mở rộng lên 10 clients

File `fl_clients/client.py` hiện chỉ hỗ trợ 5 houses:
```python
HOUSE_ORDER = ["house1", "house2", "house3", "house4", "house6"]
```

**Cần sửa thành:**
```python
HOUSE_ORDER = [
    "house1", "house2", "house3", "house4", "house5",
    "house6", "house7", "house8", "house9", "house10",
]
```

Và cập nhật `--client-index` help text tương ứng.

---

### 2. `fl_server/` — Convergence & Data Imbalance

#### 2a. Mở rộng lên 10 clients

```python
# server.py — sửa expected_clients
expected_clients = 1 if TEST_MODE_ONE_CLIENT else 10  # thay vì 5
```

#### 2b. Convergence-based Early Stopping 

**Đề xuất implementation:**

```python
# Thêm vào server.py
CONVERGENCE_THRESHOLD = 0.001   # RMSE thay đổi < 0.001 coi là hội tụ
PATIENCE = 3                    # số round liên tiếp phải thỏa điều kiện
NUM_ROUNDS = 30                 # tăng max rounds để quan sát hội tụ
```

Trong `CsvLoggingFedAvg.aggregate_evaluate()`, thêm logic:

```python
class CsvLoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, log_file, convergence_threshold=0.001, patience=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.prev_rmse = None
        self.stable_count = 0
        self.converged = False
        self.converged_round = None
        # ... khởi tạo CSV ...

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        current_rmse = aggregated_metrics.get("rmse", None) if aggregated_metrics else None

        # Kiểm tra hội tụ
        if current_rmse is not None and self.prev_rmse is not None:
            change = abs(current_rmse - self.prev_rmse)
            if change < self.convergence_threshold:
                self.stable_count += 1
            else:
                self.stable_count = 0

            if self.stable_count >= self.patience and not self.converged:
                self.converged = True
                self.converged_round = server_round
                print(f"\n{'='*50}")
                print(f"  ✅ MODEL CONVERGED at round {server_round}")
                print(f"  Final RMSE: {current_rmse:.6f}")
                print(f"  Threshold: {self.convergence_threshold}")
                print(f"  → Dừng huấn luyện, sử dụng model toàn cục này để dự đoán")
                print(f"{'='*50}\n")

        self.prev_rmse = current_rmse

        # ... ghi CSV ...
        return aggregated_loss, aggregated_metrics
```

> **Lưu ý**: Flower (flwr) không hỗ trợ dừng server giữa chừng qua API. Có 2 cách:
> 1. Khi `converged=True`, ghi flag ra file, các round sau client vẫn chạy nhưng kết quả đã ổn định
> 2. Hoặc set `NUM_ROUNDS` đủ lớn (30+), khi thấy converged thì biết round nào dừng

#### 2c. Xử lý Data Imbalance — Capped Weighted Average

Thay hàm `weighted_average` hiện tại bằng **capped version**:

```python
import numpy as np

def capped_weighted_average(metrics):
    """
    Aggregate client metrics nhưng cap max examples mỗi client
    tại giá trị median → tránh UK-DALE (data lớn) áp đảo REDD.
    """
    if not metrics:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

    all_examples = [num for num, _ in metrics]
    cap = int(np.median(all_examples))

    capped = [(min(num, cap), m) for num, m in metrics]
    total = sum(n for n, _ in capped)

    if total == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

    return {
        "mae":  sum(n * float(m.get("mae", 0))  for n, m in capped) / total,
        "rmse": sum(n * float(m.get("rmse", 0)) for n, m in capped) / total,
        "r2":   sum(n * float(m.get("r2", 0))   for n, m in capped) / total,
    }
```

**Tại sao cap tại median?**
- Median cân bằng giữa REDD (~1000 rows) và UK-DALE (~13000 rows)
- Các house UK-DALE nhỏ (house8 ~2900) vẫn giữ nguyên trọng số
- House6 (~135000) bị giới hạn xuống, tránh dominance

#### 2d. FedAvg Aggregation Weights

Ngoài metrics aggregation, **FedAvg cũng aggregate model weights** theo `num_examples`.  
Để cân bằng, cần override `aggregate_fit()`:

```python
class BalancedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Cap num_examples trước khi aggregate
        all_examples = [fit_res.num_examples for _, fit_res in results]
        cap = int(np.median(all_examples))

        capped_results = []
        for client_proxy, fit_res in results:
            # Tạo bản copy với num_examples đã cap
            capped_res = fl.common.FitRes(
                status=fit_res.status,
                parameters=fit_res.parameters,
                num_examples=min(fit_res.num_examples, cap),
                metrics=fit_res.metrics,
            )
            capped_results.append((client_proxy, capped_res))

        return super().aggregate_fit(server_round, capped_results, failures)
```

---

### 3. Sử dụng Global Model sau khi hội tụ

Sau khi FL training converge, model toàn cục có thể dùng để dự đoán:

```python
# Ví dụ: load global model weights sau FL
from sklearn.neural_network import MLPRegressor
import numpy as np

# Khởi tạo model cùng kiến trúc
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu")

# Fit dummy để tạo cấu trúc weights
model.fit(X_train[:10], y_train[:10])

# Load weights từ FL server (sau khi converge)
# global_weights = [coef_0, coef_1, intercept_0, intercept_1]
model.coefs_ = [global_weights[0], global_weights[1]]
model.intercepts_ = [global_weights[2], global_weights[3]]

# Dự đoán
y_pred = model.predict(X_new)
```
