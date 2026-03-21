# Federated Learning for Smart Home Energy Optimization

## Goal
Build a distributed smart energy management demo system where multiple homes collaboratively train a model to predict next-hour energy consumption without sharing raw data.

## Project Scope
- 5 simulated homes first
- Next-hour household energy consumption prediction
- Naive baseline + local model + federated model
- Flower for federated learning
- Rule-based optimization for peak-hour control
- Streamlit dashboard

## Folder Structure
- `data/`: raw and processed datasets
- `models/`: local model scripts
- `fl_server/`: federated server code
- `fl_clients/`: client code
- `dashboard/`: demo UI
- `results/`: metrics, plots, logs
- `report/`: report files
