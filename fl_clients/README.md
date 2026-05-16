# FL Clients

Main file: `fl_clients/client.py`

This module runs one Flower client per house for the smart-home FL pipeline.

## What it does
- maps 10 houses to 10 clients (decentralized FL setup)
- loads one house dataset per client
- implements required Flower methods:
  - `get_parameters()`
  - `fit()`
  - `evaluate()`
- returns regression metrics: `mae`, `rmse`, `r2`

## Client mapping

| Index | House    |
|-------|----------|
| 0     | house1   |
| 1     | house2   |
| 2     | house3   |
| 3     | house4   |
| 4     | house5   |
| 5     | house6   |
| 6     | house7   |
| 7     | house8   |
| 8     | house9   |
| 9     | house10  |

Run mode options:
- `--client-index <0..9>`
- `--house <house1|house2|...|house10>`

## Data rules
Client searches files in this priority order:
1. `data/data/processed/<house>_train.csv` + `data/data/processed/<house>_test.csv`
2. `data/processed/<house>_train.csv` + `data/processed/<house>_test.csv`
3. `data/data/processed/<house>_clean.csv` (fallback)
4. `data/processed/<house>_hourly_clean.csv` (legacy fallback)

Assumptions:
- target = `main`
- features = all columns except `main` and `split`
- schema safety = uses only common feature columns between train/test
- split = uses explicit train/test files when available; otherwise time-based 80/20

## Model interface
- local model: `MLPRegressor` (sklearn)
- shared parameters: all MLP layer weights + biases
- evaluate loss: `rmse`
- returned metrics: `mae`, `rmse`, `r2`, `house`, `num_features`

## Commands

### Dry run (no server, single client)
```bash
python fl_clients/client.py --client-index 0 --dry-run
```

### Launch all 10 clients at once (recommended)
```bash
# Start server first (do bạn Server phụ trách):
python fl_server/server.py

# Then launch all 10 clients in one command:
python fl_clients/run_clients.py
```

`run_clients.py` options:
| Flag               | Default          | Description                          |
|--------------------|------------------|--------------------------------------|
| `--num-clients`    | 10               | Number of clients to launch (1-10)   |
| `--server-address` | 127.0.0.1:8080   | Flower server address                |
| `--dry-run`        | off              | Offline test, no server needed       |

Examples:
```bash
# Launch only the first 5 clients
python fl_clients/run_clients.py --num-clients 5

# Dry-run all 10 clients (no server needed)
python fl_clients/run_clients.py --dry-run

# Connect to a remote server
python fl_clients/run_clients.py --server-address 192.168.1.100:8080
```

Press **Ctrl+C** at any time to gracefully terminate all running client processes.

### Manual launch (individual terminals)
```bash
python fl_clients/client.py --client-index 0 --server-address 127.0.0.1:8080
python fl_clients/client.py --client-index 1 --server-address 127.0.0.1:8080
# ... repeat for indices 2-9
python fl_clients/client.py --client-index 9 --server-address 127.0.0.1:8080
```

## Quick check
- client starts without errors
- correct house is loaded
- train/test shapes are printed
- server receives 10 clients
