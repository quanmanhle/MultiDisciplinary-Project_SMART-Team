# FL Clients

Main file: `fl_clients/client.py`

This module runs one Flower client per house for the smart-home FL pipeline.

## What it does
- Maps 10 houses to 10 Flower clients (decentralised FL setup)
- Loads one house dataset per client
- Implements required Flower methods: `get_parameters()`, `fit()`, `evaluate()`
- Returns regression metrics: `mae`, `rmse`, `r2`
- Supports **watt mode** (default): reads unscaled watts, MinMaxScales internally — no negative predictions
- Supports **standard mode**: reads pre-scaled StandardScaled CSVs (for legacy runs)
- Supports **`--cap`**: subsamples training rows to balance REDD vs UK-DALE data size

## Client mapping

| Index | House   | Dataset |
|-------|---------|---------|
| 0     | house1  | REDD    |
| 1     | house2  | REDD    |
| 2     | house3  | REDD    |
| 3     | house4  | REDD    |
| 4     | house5  | REDD    |
| 5     | house6  | UK-DALE |
| 6     | house7  | UK-DALE |
| 7     | house8  | UK-DALE |
| 8     | house9  | UK-DALE |
| 9     | house10 | UK-DALE |

## Data flow (watt mode, default)

```
data/data/processed/<house>_clean.csv   ← original watts (unscaled)
         │
         ├─ aligned to train/test timestamps from <house>_train/test.csv
         ├─ clipped: watt cols < 0 → 0
         ├─ [optional] subsampled to --cap rows
         ├─ zero-variance features dropped
         ├─ MinMaxScaler fitted on train → transform train + test
         └─ [evaluate] inverse_transform predictions → report in watts
```

This mirrors `models/train_local_mlp.py --data-mode watt` so FL and local baseline metrics are directly comparable.

## Data rules
Client searches files in this priority order:
1. `data/data/processed/<house>_clean.csv` (watt mode default)
2. `data/processed/<house>_clean.csv`
3. Falls back to standard mode if no clean file found

Split timestamp reference:
1. `data/data/processed/<house>_train.csv` + `<house>_test.csv`
2. `data/processed/<house>_train.csv` + `<house>_test.csv`
3. Time-based 80/20 split on clean data (last resort)

## Model
- Architecture: `MLPRegressor` (sklearn)
  - Large datasets (≥1000 rows): layers=(64,32), lr=0.001, α=1e-4, max_iter=500
  - Small datasets (<1000 rows): layers=(32,16), lr=0.0005, α=0.01, max_iter=1000
- Shared parameters: all MLP layer weights + biases
- Evaluate loss: `rmse` (in original watts for watt mode)
- Returned metrics: `mae`, `rmse`, `r2`, `house`, `num_features`, `data_mode`

## Commands

### Dry run (no server, single client)
```bash
# Watt mode (default, recommended)
python fl_clients/client.py --client-index 0 --dry-run

# With cap (match local baseline --cap 1000)
python fl_clients/client.py --client-index 0 --dry-run --cap 1000

# Standard mode (legacy)
python fl_clients/client.py --client-index 0 --dry-run --data-mode standard
```

### Dry run all 10 clients (no server needed)
```bash
python fl_clients/run_clients.py --dry-run

# With cap to balance REDD vs UK-DALE
python fl_clients/run_clients.py --dry-run --cap 1000
```

### Full FL training run (recommended)
```bash
# 1. Start server (in a separate terminal):
python fl_server/server.py

# 2. Launch all 10 clients (in another terminal):
python fl_clients/run_clients.py

# With cap for balanced training:
python fl_clients/run_clients.py --cap 1000
```

### `run_clients.py` options

| Flag               | Default          | Description                                              |
|--------------------|------------------|----------------------------------------------------------|
| `--num-clients`    | 10               | Number of clients to launch (1–10)                       |
| `--server-address` | 127.0.0.1:8080   | Flower server address                                    |
| `--data-mode`      | watt             | `watt` (MinMaxScaled, recommended) or `standard`         |
| `--cap`            | none             | Cap training rows per house (e.g. 1000 to balance sets)  |
| `--dry-run`        | off              | Offline test, no server needed                           |

### `client.py` options

| Flag               | Default          | Description                                              |
|--------------------|------------------|----------------------------------------------------------|
| `--house`          | —                | House name (`house1`..`house10`), mutually exclusive with `--client-index` |
| `--client-index`   | —                | 0–9 index mapped to house1–house10                       |
| `--server-address` | 127.0.0.1:8080   | Flower server address                                    |
| `--data-mode`      | watt             | `watt` or `standard`                                     |
| `--cap`            | none             | Max training rows for this client                        |
| `--dry-run`        | off              | Offline test, results saved to `results/fl_client_test_results.csv` |

### Examples
```bash
# Launch only the first 5 clients
python fl_clients/run_clients.py --num-clients 5

# Connect to a remote server
python fl_clients/run_clients.py --server-address 192.168.1.100:8080

# Balanced run matching local baseline
python fl_clients/run_clients.py --cap 1000 --data-mode watt
```

Press **Ctrl+C** at any time to gracefully terminate all running client processes.

## Quick sanity check

```bash
# Should print data shapes and metrics without errors
python fl_clients/client.py --client-index 0 --dry-run
python fl_clients/client.py --house house4 --dry-run --cap 1000
```

Expected output:
```
[CLIENT] House     = house1
[CLIENT] Data mode = watt
[CLIENT] Feature cols : [...]
[CLIENT] Train shape  : x=(812, 11), y=(812,)
[CLIENT] Test shape   : x=(233, 11), y=(233,)
[DRY-RUN] ...
```

## Alignment with local baseline (Thành's `train_local_mlp.py`)

| Setting              | `train_local_mlp.py`        | `client.py` (FL)              |
|----------------------|-----------------------------|-------------------------------|
| Data source          | `house*_clean.csv`          | `house*_clean.csv` (watt mode)|
| Internal scaling     | `MinMaxScaler` on train     | `MinMaxScaler` on train       |
| Architecture         | (64,32) or (32,16)          | (64,32) or (32,16) same rule  |
| Training cap         | `--cap N` or `--balanced`   | `--cap N`                     |
| Metrics unit         | watts (W)                   | watts (W) via inverse_transform|
| house4 handling      | EWMA wins → used in summary | same MLP; EWMA on server side |
