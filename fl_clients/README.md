# FL Clients

Main file: `fl_clients/client.py`

## Purpose

This module runs one Flower client per house for the federated learning pipeline.

Each client:
- loads its house dataset (REDD or UK-DALE)
- trains a local `MLPRegressor` model
- participates in FL rounds: `get_parameters()`, `fit()`, `evaluate()`
- reports metrics in original watts: `mae`, `rmse`, `r2`

## Current Setup

- **Framework:** Flower (`flwr`)
- **Model:** MLPRegressor — layers=(64,32), fixed across all clients for FedAvg compatibility
- **Data mode:** watt (default) — reads `house*_clean.csv`, scales with MinMaxScaler internally
- **Training cap:** 1000 rows per house (balances REDD vs UK-DALE dataset sizes)
- **Clients:** 10 (house1–house10)
- **Server address:** `127.0.0.1:8080`

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

## File Structure

- `client.py` — Flower client for one house (data loading, model, FL methods)
- `run_clients.py` — launches all 10 clients in parallel via subprocess

## How to Run

From the project root:

```bash
# 1. Start the server first (separate terminal):
python fl_server/server.py

# 2. Launch all 10 clients:
python fl_clients/run_clients.py
```

With training cap (recommended, matches local baseline):
```bash
python fl_clients/run_clients.py --cap 1000
```

## Quick sanity check (no server needed)

```bash
python fl_clients/client.py --client-index 0 --dry-run
python fl_clients/client.py --house house4 --dry-run --cap 1000
```

Output:
```
[CLIENT] House     = house1
[CLIENT] Data mode = watt
[CLIENT] Feature cols : [...]
[CLIENT] Train shape  : x=(812, 11), y=(812,)
[CLIENT] Test shape   : x=(233, 11), y=(233,)
[DRY-RUN] ...
```

Dry-run results are saved to `results/fl_client_test_results.csv`.
