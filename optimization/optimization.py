from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


HIGH_PRIORITY = {"fridge"}
MEDIUM_PRIORITY = {"electric stove"}
LOW_PRIORITY = {"dish washer", "washer dryer", "microwave"}

DEFAULT_DEVICE_WATTS = {
    "fridge": 80.0,
    "dish washer": 120.0,
    "washer dryer": 250.0,
    "microwave": 100.0,
    "electric stove": 300.0,
}


def find_project_root() -> Path:
    """
    Find the project root robustly whether this file is placed at:
    - project_root/optimization.py
    - project_root/optimization/optimization.py
    - project_root/src/optimization.py
    """
    current = Path(__file__).resolve().parent
    candidates = [current, *current.parents]

    for candidate in candidates:
        if (
            (candidate / "data").exists()
            or (candidate / "fl_server").exists()
            or (candidate / "fl_clients").exists()
            or (candidate / "results").exists()
        ):
            return candidate

    # Safe fallback: current working directory when launched from project root.
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
RESULT_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULT_DIR / "optimization_result.json"


def clean_non_negative(value: float) -> float:
    """Power values cannot be negative; clamp to zero for safety."""
    return max(float(value), 0.0)


def optimize_devices(
    predicted_main: float,
    threshold: float,
    current_devices: Dict[str, float],
) -> Dict[str, Any]:
    """
    Rule-based optimization for peak-load control in watt mode.

    Args:
        predicted_main: predicted household aggregate power in watts.
        threshold: peak threshold in watts.
        current_devices: current appliance power values in watts.

    Returns:
        Dictionary containing optimization results and recommended actions.
    """
    predicted_main = clean_non_negative(predicted_main)
    threshold = clean_non_negative(threshold)
    current_devices = {
        device: clean_non_negative(power)
        for device, power in current_devices.items()
    }

    actions: List[Dict[str, str]] = []
    estimated_reduced_load = 0.0
    peak_detected = predicted_main > threshold

    if not peak_detected:
        return {
            "mode": "watt",
            "unit": "W",
            "actions": actions,
            "peak_detected": False,
            "predicted_main": predicted_main,
            "threshold": threshold,
            "current_devices": current_devices,
            "estimated_reduced_load": 0.0,
            "estimated_post_optimization_load": predicted_main,
            "devices_turned_off": 0,
            "devices_deferred": 0,
            "peak_resolved": False,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    for device in sorted(LOW_PRIORITY):
        current_value = current_devices.get(device, 0.0)
        if current_value > 0:
            actions.append(
                {
                    "device": device,
                    "action": "off",
                    "reason": "low priority device during predicted peak",
                }
            )
            estimated_reduced_load += current_value

    for device in sorted(MEDIUM_PRIORITY):
        current_value = current_devices.get(device, 0.0)
        if current_value > 0:
            actions.append(
                {
                    "device": device,
                    "action": "defer",
                    "reason": "medium priority device during predicted peak",
                }
            )
            estimated_reduced_load += current_value

    estimated_post_optimization_load = max(
        predicted_main - estimated_reduced_load,
        0.0,
    )

    peak_resolved = estimated_post_optimization_load <= threshold

    return {
        "mode": "watt",
        "unit": "W",
        "actions": actions,
        "peak_detected": True,
        "predicted_main": predicted_main,
        "threshold": threshold,
        "current_devices": current_devices,
        "estimated_reduced_load": estimated_reduced_load,
        "estimated_post_optimization_load": estimated_post_optimization_load,
        "devices_turned_off": sum(1 for action in actions if action["action"] == "off"),
        "devices_deferred": sum(1 for action in actions if action["action"] == "defer"),
        "peak_resolved": peak_resolved,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def save_result(result: Dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rule-based smart-home device optimization in watt mode."
    )
    parser.add_argument(
        "--predicted-main",
        type=float,
        default=850.0,
        help="Predicted aggregate household load in watts. Default: 850 W.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=600.0,
        help="Peak threshold in watts. Default: 600 W.",
    )
    parser.add_argument("--fridge", type=float, default=DEFAULT_DEVICE_WATTS["fridge"])
    parser.add_argument("--dish-washer", type=float, default=DEFAULT_DEVICE_WATTS["dish washer"])
    parser.add_argument("--washer-dryer", type=float, default=DEFAULT_DEVICE_WATTS["washer dryer"])
    parser.add_argument("--microwave", type=float, default=DEFAULT_DEVICE_WATTS["microwave"])
    parser.add_argument("--electric-stove", type=float, default=DEFAULT_DEVICE_WATTS["electric stove"])
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help="Output JSON path. Default: results/optimization_result.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    current_devices = {
        "fridge": args.fridge,
        "dish washer": args.dish_washer,
        "washer dryer": args.washer_dryer,
        "microwave": args.microwave,
        "electric stove": args.electric_stove,
    }

    result = optimize_devices(
        predicted_main=args.predicted_main,
        threshold=args.threshold,
        current_devices=current_devices,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    save_result(result, output_path)

    print("Optimization mode:", result["mode"])
    print("Unit:", result["unit"])
    print("Peak detected:", result["peak_detected"])
    print("Predicted main:", result["predicted_main"])
    print("Threshold:", result["threshold"])
    print("Estimated reduced load:", result["estimated_reduced_load"])
    print("Estimated post-optimization load:", result["estimated_post_optimization_load"])
    print("Devices turned off:", result["devices_turned_off"])
    print("Devices deferred:", result["devices_deferred"])
    print("Peak resolved:", result["peak_resolved"])
    print("Actions:")
    for action in result["actions"]:
        print(" -", action)

    print(f"\nSaved optimization result to: {output_path}")


if __name__ == "__main__":
    main()
