# FL Clients

Main file: `fl_clients/client.py`

This module runs one Flower client per house for the smart-home FL pipeline.

## What it does
- maps 5 houses to 5 clients
- loads one house dataset per client
- implements required Flower methods:
  - `get_parameters()`
  - `fit()`
  - `evaluate()`
- returns regression metrics: `mae`, `rmse`, `r2`

## Client mapping
- `0 -> house1`
- `1 -> house2`
- `2 -> house3`
- `3 -> house4`
- `4 -> house6`

Run mode options:
- `--client-index <0..4>`
- `--house <house1|house2|house3|house4|house6>`

## Data rules
Client searches files in this order:
1. `data/processed/<house>_hourly_clean.csv`
2. `models/processed/<house>_hourly_clean.csv` (fallback)

Assumptions:
- target = `main`
- features = all columns except `main`
- split = time-based 80/20 (train/test)

## Model interface
- local model: `LinearRegression`
- shared parameters: `[coef, intercept]`
- evaluate loss: `rmse`
- returned metrics: `mae`, `rmse`, `r2`, `house`, `num_features`

## Commands

### Dry run (no server)
```bash
python fl_clients/client.py --client-index 0 --dry-run
```

### Full run
Start server:
```bash
python fl_server/server.py
```

Start all clients (5 terminals):
```bash
python fl_clients/client.py --client-index 0 --server-address 127.0.0.1:8080
python fl_clients/client.py --client-index 1 --server-address 127.0.0.1:8080
python fl_clients/client.py --client-index 2 --server-address 127.0.0.1:8080
python fl_clients/client.py --client-index 3 --server-address 127.0.0.1:8080
python fl_clients/client.py --client-index 4 --server-address 127.0.0.1:8080
```

## Quick check
- client starts without errors
- correct house is loaded
- train/test shapes are printed
- server receives 5 clients
