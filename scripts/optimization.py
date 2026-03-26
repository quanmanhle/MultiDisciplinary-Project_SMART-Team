from __future__ import annotations

from typing import Dict, List


HIGH_PRIORITY = {"fridge"}
MEDIUM_PRIORITY = {"electric stove"}
LOW_PRIORITY = {"dish washer", "washer dryer", "microwave"}


def optimize_devices(
    predicted_main: float,
    threshold: float,
    current_devices: Dict[str, float],
) -> List[Dict[str, str]]:
    """
    Rule-based optimization for peak-load control.

    Args:
        predicted_main: predicted next-hour household consumption
        threshold: peak threshold
        current_devices: current appliance values/power usage

    Returns:
        List of recommended control actions
    """
    actions: List[Dict[str, str]] = []

    if predicted_main <= threshold:
        return actions

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

    return actions


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
    for item in result:
        print(item)