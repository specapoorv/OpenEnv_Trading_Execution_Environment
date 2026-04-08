from __future__ import annotations

from typing import Dict


def sample_data_anomaly(rng, can_inconsistency: bool) -> Dict[str, bool]:
    return {
        "stale": rng.random() < 0.26,
        "missing_field": rng.random() < 0.20,
        "tool_failure": rng.random() < 0.10,
        "inconsistent": can_inconsistency and rng.random() < 0.22,
    }
