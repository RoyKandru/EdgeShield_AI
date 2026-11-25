"""Minimal promotion gate. Promotion is explicit and requires supplied evaluation metrics."""
from pathlib import Path
import json

REQUIRED_METRICS = {"precision", "recall", "f1", "false_positive_rate"}

def evaluate_candidate(metrics: dict, minimum_f1: float = 0.80,
                       maximum_fpr: float = 0.10) -> dict:
    missing = sorted(REQUIRED_METRICS - set(metrics))
    if missing:
        return {"eligible": False, "reasons": [f"Missing metrics: {', '.join(missing)}"]}
    reasons = []
    for key in REQUIRED_METRICS:
        value = metrics[key]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            reasons.append(f"{key} must be between 0 and 1")
    if reasons:
        return {"eligible": False, "reasons": reasons}
    if metrics["f1"] < minimum_f1:
        reasons.append(f"F1 below minimum {minimum_f1}")
    if metrics["false_positive_rate"] > maximum_fpr:
        reasons.append(f"False-positive rate above maximum {maximum_fpr}")
    return {"eligible": not reasons, "reasons": reasons,
            "note": "This gate checks supplied metrics only; metrics must come from a held-out, representative test set."}
