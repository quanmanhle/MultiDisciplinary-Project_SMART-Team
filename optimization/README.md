# Device Optimization Module

## Overview

This module implements a rule-based smart-home device optimization engine for the Smart Home Energy Optimization system.

After the Federated Learning model predicts the next household energy consumption, this module checks whether the predicted load exceeds a predefined peak threshold. If a peak is detected, it recommends control actions for household appliances based on device priority.

The optimization module is designed for the final system demo and dashboard integration.

---

## Files

| File | Description |
| --- | --- |
| `optimization.py` | Main rule-based optimization script |
| `results/optimization_result.json` | Output JSON containing optimization actions and estimated load reduction |

---

## Optimization Logic

The system divides devices into three priority groups:

| Priority | Devices | Action During Peak |
| --- | --- | --- |
| High | `fridge` | Keep running |
| Medium | `electric stove` | Defer |
| Low | `dish washer`, `washer dryer`, `microwave` | Turn off |

The optimization process follows this logic:

```text
If predicted_main <= threshold:
    No peak is detected
    No control action is needed

If predicted_main > threshold:
    Peak is detected
    Turn off low-priority devices
    Defer medium-priority devices
    Keep high-priority devices running
    Estimate the reduced load
    Check whether the peak is resolved