# FL Server Module

## Purpose
This module contains the Flower federated learning server used in the project **Federated Learning for Smart Home Energy Optimization**.

The server is responsible for:
- coordinating federated training
- receiving updated model weights from clients
- aggregating client updates using **FedAvg**
- producing the global model across training rounds

## Current Setup
- **Framework:** Flower (`flwr`)
- **Strategy:** FedAvg
- **Number of clients:** 5
- **Training rounds:** 10
- **Server address:** `0.0.0.0:8080`

## File Structure
- `server.py` — starts the Flower server and defines the FL strategy

## How It Works
1. The server starts and waits for federated clients to connect
2. Each client represents one house
3. Clients train their local model using their own house data
4. Clients send updated model weights back to the server
5. The server aggregates these weights using FedAvg
6. The updated global model is sent back to clients in the next round

## Current Status
At the current stage of the project, the FL server skeleton is completed and can run locally.

The next steps will be:
- connect the Flower clients
- run full federated training
- log training metrics for comparison and dashboard use

## How to Run
From the project root directory:

```bash
python fl_server/server.py
