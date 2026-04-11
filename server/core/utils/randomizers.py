from __future__ import annotations

from typing import Dict


def sample_data_anomaly(rng, can_inconsistency: bool) -> Dict[str, bool]:
    return {
        "stale": rng.random() < 0.26,
        "missing_field": rng.random() < 0.20,
        "tool_failure": rng.random() < 0.10,
        "inconsistent": can_inconsistency and rng.random() < 0.22,
    }

def sample_data_probabilities(rng, can_inconsistency: bool) -> Dict[str, float]:
    # We return the CHANCE of failure, not the failure itself
    return {
        "stale_prob": 0.26 if rng.random() < 0.5 else 0.0,
        "missing_field_prob": 0.20 if rng.random() < 0.4 else 0.0,
        "tool_failure_prob": 0.10 if rng.random() < 0.3 else 0.0,
        "inconsistent_prob": 0.22 if (can_inconsistency and rng.random() < 0.5) else 0.0,
    }