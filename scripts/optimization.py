from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List


HIGH_PRIORITY = {"fridge"}
MEDIUM_PRIORITY = {"electric stove"}
LOW_PRIORITY = {"dish washer", "washer dryer", "microwave"}

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULT_DIR / "optimization_result.json"


def optimize_devices(
    predicted_main: float,
    threshold: float,
    current_devices: Dict[str, float],
) -> Dict[str, Any]:
    """
    Rule-based optimization for peak-load control.

    Args:
        predicted_main: predicted next-hour household consumption
        threshold: peak threshold
        current_devices: current appliance values/power usage

    Returns:
        Dictionary containing optimization results and recommended actions
    """
    actions: List[Dict[str, str]] = []
    estimated_reduced_load = 0.0

    peak_detected = predicted_main > threshold

    if not peak_detected:
        return {
            "actions": actions,
            "peak_detected": False,
            "predicted_main": predicted_main,
            "threshold": threshold,
            "estimated_reduced_load": 0.0,
            "estimated_post_optimization_load": predicted_main,
            "devices_turned_off": 0,
            "peak_resolved": False,
        }

    for device in LOW_PRIORITY:
        current_value = current_devices.get(device, 0.0)
        if current_value > 0:
            actions.append(
                {
                    "device": device,
                    "action": "off",
                    "reason": "peak predicted",
                }
            )
            estimated_reduced_load += current_value

    for device in MEDIUM_PRIORITY:
        current_value = current_devices.get(device, 0.0)
        if current_value > 0:
            actions.append(
                {
                    "device": device,
                    "action": "defer",
                    "reason": "medium priority during peak",
                }
            )
            estimated_reduced_load += current_value

    estimated_post_optimization_load = max(
        predicted_main - estimated_reduced_load, 0.0
    )

    peak_resolved = estimated_post_optimization_load <= threshold

    return {
        "actions": actions,
        "peak_detected": True,
        "predicted_main": predicted_main,
        "threshold": threshold,
        "estimated_reduced_load": estimated_reduced_load,
        "estimated_post_optimization_load": estimated_post_optimization_load,
        "devices_turned_off": len([a for a in actions if a["action"] == "off"]),
        "peak_resolved": peak_resolved,
    }


def save_result(result: Dict[str, Any], output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    predicted_main = 0.18
    threshold = 0.12
    current_devices = {
        "fridge": 0.08,
        "dish washer": 0.04,
        "electric stove": 0.03,
        "microwave": 0.02,
        "washer dryer": 0.05,
    }

    result = optimize_devices(predicted_main, threshold, current_devices)
    save_result(result, OUTPUT_FILE)

    print("Peak detected:", result["peak_detected"])
    print("Predicted main:", result["predicted_main"])
    print("Threshold:", result["threshold"])
    print("Estimated reduced load:", result["estimated_reduced_load"])
    print("Estimated post-optimization load:", result["estimated_post_optimization_load"])
    print("Devices turned off:", result["devices_turned_off"])
    print("Peak resolved:", result["peak_resolved"])
    print("Actions:")
    for action in result["actions"]:
        print(action)

    print(f"\nSaved optimization result to: {OUTPUT_FILE}")