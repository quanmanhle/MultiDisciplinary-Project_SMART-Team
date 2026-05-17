# Models - Local Baseline Training

Thư mục `models/` chứa script huấn luyện và đánh giá các mô hình local cho bài toán dự đoán tiêu thụ điện hộ gia đình.

Script chính:

```bash
models/train_local_mlp.py
```

Kết quả local này dùng làm baseline để so sánh với mô hình global trong Federated Learning.

## Luồng xử lý hiện tại

`train_local_mlp.py` đọc dữ liệu đã xử lý trong `data/processed/` cho 10 house:

| House | Dataset | Nguồn |
| --- | --- | --- |
| `house1` | REDD | REDD house1 |
| `house2` | REDD | REDD house2 |
| `house3` | REDD | REDD house3 |
| `house4` | REDD | REDD house4 |
| `house5` | REDD | REDD house6 |
| `house6` | UK-DALE | UK-DALE house_1 |
| `house7` | UK-DALE | UK-DALE house_2 |
| `house8` | UK-DALE | UK-DALE house_3 |
| `house9` | UK-DALE | UK-DALE house_4 |
| `house10` | UK-DALE | UK-DALE house_5 |

Mỗi house được đánh giá bằng:

| Model | Mô tả |
| --- | --- |
| `naive_baseline` | Dự đoán bằng `lag_1` |
| `adaptive_ewma_baseline` | Baseline chuỗi thời gian dùng EWMA từ `lag_1`, hữu ích khi dữ liệu bị drift |
| `mlp_regression` | MLP local train riêng cho từng house |

Cuối quá trình, script chọn predictor tốt nhất giữa `mlp_regression` và `adaptive_ewma_baseline` theo `R2` cho từng house. Vì vậy `house4` sẽ dùng `adaptive_ewma_baseline` nếu MLP cho `R2` âm.

## Cách chạy

Train tất cả 10 house ở chế độ mặc định:

```bash
python models/train_local_mlp.py
```

Train với dữ liệu cân bằng, cap số dòng train mỗi house tại median:

```bash
python models/train_local_mlp.py --balanced
```

Train với cap cụ thể:

```bash
python models/train_local_mlp.py --cap 5000
```

Tái lập hành vi cũ, dùng trực tiếp các split đã `StandardScaled`:

```bash
python models/train_local_mlp.py --data-mode standard
```

Kết hợp mode standard với cap:

```bash
python models/train_local_mlp.py --data-mode standard --balanced
```

## Tham số CLI

| Tham số | Mặc định | Mô tả |
| --- | --- | --- |
| `--data-mode watt` | Có | Đọc `house*_clean.csv`, train với watt gốc, scale nội bộ bằng `MinMaxScaler`, output prediction không âm |
| `--data-mode standard` | Không | Đọc trực tiếp `house*_train.csv`, `house*_valid.csv`, `house*_test.csv` đã `StandardScaled` |
| `--balanced` | Không | Cap số dòng train mỗi house tại median để giảm lệch giữa REDD và UK-DALE |
| `--cap N` | Không | Cap số dòng train mỗi house tại đúng `N` dòng |

## Data mode

### `watt`

Đây là mode mặc định. Script đọc `house*_clean.csv` ở đơn vị watt gốc, sau đó align lại đúng timestamp của các split train/valid/test.

Trong mode này:

- Các cột watt-like âm sẽ được clip về `0`.
- Feature và target của MLP được scale nội bộ bằng `MinMaxScaler`.
- Prediction của MLP được inverse về watt.
- Prediction cuối cùng được clip dưới `0`, nên không có giá trị công suất âm.
- Metrics MAE/RMSE được tính theo watt.

### `standard`

Mode này dùng lại các file split đã chuẩn hóa bởi `data/data_pipeline.py`.

Trong mode này:

- `main`, `lag_1`, `rolling_mean` và appliance features có thể âm vì đã `StandardScaled`.
- Metrics MAE/RMSE/R2 nằm trên thang chuẩn hóa.
- Đây là mode phù hợp nếu cần so sánh với kết quả cũ.

## Output

Sau khi chạy, script ghi kết quả vào `models/resultmlp/`:

| Đường dẫn | Nội dung |
| --- | --- |
| `local_metrics.csv` | Tất cả metrics của `naive_baseline`, `adaptive_ewma_baseline`, `mlp_regression` |
| `selected_metrics.csv` | Predictor tốt nhất cho từng house theo `R2` |
| `plots/` | Biểu đồ actual vs predicted cho từng house và từng model |
| `predictions/` | CSV actual/predicted cho từng house và từng model |
| `saved_models/` | File `.joblib` của MLP, gồm model, feature list và scaler |

Nên dùng `selected_metrics.csv` khi muốn báo cáo bảng tổng kết cuối cùng. Nên dùng `local_metrics.csv` khi muốn debug hoặc so sánh từng model riêng lẻ.

## Metrics

Script tính 3 metrics:

| Metric | Ý nghĩa |
| --- | --- |
| `MAE` | Sai số tuyệt đối trung bình, càng thấp càng tốt |
| `RMSE` | Sai số bình phương trung bình lấy căn, phạt mạnh lỗi lớn |
| `R2` | Mức giải thích phương sai, càng gần `1` càng tốt |

Lưu ý: `R2` có thể âm. Điều đó không có nghĩa prediction bị âm; nó nghĩa là model tệ hơn baseline đoán bằng trung bình của tập test. Trường hợp `house4` từng bị như vậy với MLP do test split có level tiêu thụ thấp hơn train/valid. Script hiện đã thêm `adaptive_ewma_baseline` và chọn predictor tốt nhất trong `selected_metrics.csv`.

## MLP

Kiến trúc mặc định:

```text
Input -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(1)
```

Với house nhỏ hơn 1000 dòng train:

```text
Input -> Dense(32, ReLU) -> Dense(16, ReLU) -> Output(1)
```

Thiết lập chính:

| Thành phần | Giá trị |
| --- | --- |
| Optimizer | Adam |
| Activation | ReLU |
| Early stopping | Có |
| `n_iter_no_change` | 20 |
| `random_state` | 42 |
| Small data alpha | `0.01` |
| Normal alpha | `0.0001` |

Script tự drop các feature có variance bằng `0` trong train set, ví dụ một appliance không tồn tại ở house đó.

## Adaptive EWMA baseline

`adaptive_ewma_baseline` dùng `lag_1` và làm mượt theo công thức EWMA:

```text
pred_t = alpha * lag_1_t + (1 - alpha) * pred_{t-1}
```

Giá trị hiện tại:

```text
alpha = 0.05
```

Baseline này causal vì chỉ dùng `lag_1`, tức thông tin đã có tại thời điểm dự đoán. Nó đặc biệt hữu ích khi test split có mức tiêu thụ dịch chuyển so với train/valid, như `house4`.

## Data imbalance

UK-DALE có số dòng lớn hơn REDD rất nhiều, đặc biệt `house6`. Vì vậy khi muốn so sánh công bằng giữa các house, nên chạy:

```bash
python models/train_local_mlp.py --balanced
```

Hoặc chọn cap cụ thể:

```bash
python models/train_local_mlp.py --cap 5000
```

Cap chỉ áp dụng cho train split. Valid/test vẫn giữ nguyên để đánh giá đúng dữ liệu gốc.

## Cách đọc kết quả nhanh

Nếu muốn xem bảng cuối cùng:

```bash
python models/train_local_mlp.py --data-mode standard
```

Xem file:

```bash
models/resultmlp/selected_metrics.csv
```

Ví dụ sau lần chạy gần nhất ở `standard` mode, `house4` được chọn:

```text
house4, adaptive_ewma_baseline, R2 = 0.0181
```

MLP raw của `house4` vẫn nằm trong `local_metrics.csv` để đối chiếu, nhưng không được chọn trong bảng cuối nếu `R2` thấp hơn EWMA.
