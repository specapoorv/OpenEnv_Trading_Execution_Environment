from __future__ import annotations

from typing import Dict

from server.core.env.execution_desk_env import ExecutionDeskEnv, heuristic_policy


def _bounded_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def grade_data_validation(seed: int = 7) -> float:
    env = ExecutionDeskEnv(seed=seed)
    observation, info = env.reset(seed=seed)
    for _ in range(30):
        action = heuristic_policy(observation, info)
        observation, _, terminated, truncated, info = env.step(action)
        if info["completed_flags"]["data_ready"]:
            return 1.0
        if terminated or truncated:
            break
    score = 1.0 - min(1.0, len(info["data_validation"]["issues"]) / 8.0)
    return _bounded_score(score)


def grade_system_readiness(seed: int = 7) -> float:
    env = ExecutionDeskEnv(seed=seed)
    observation, info = env.reset(seed=seed)
    while not info["completed_flags"]["data_ready"]:
        observation, _, terminated, truncated, info = env.step(heuristic_policy(observation, info))
        if terminated or truncated:
            return 0.0
    for _ in range(20):
        observation, _, terminated, truncated, info = env.step(heuristic_policy(observation, info))
        if info["completed_flags"]["systems_ready"] or info["event"].get("correct_escalation"):
            return 1.0
        if terminated or truncated:
            break
    score = 1.0 - min(1.0, len(info["system_readiness"]["issues"]) / 4.0)
    return _bounded_score(score)


def grade_execution(seed: int = 7) -> float:
    env = ExecutionDeskEnv(seed=seed)
    observation, info = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _, terminated, truncated, info = env.step(heuristic_policy(observation, info))
        if info["completed_flags"]["execution_complete"]:
            return 1.0
    tracking_error = info["execution_status"]["tracking_error"]
    score = 1.0 - min(1.0, tracking_error / 100.0)
    return _bounded_score(score)


def run_all_graders(seed: int = 7) -> Dict[str, float]:
    return {
        "task1_data_validation": grade_data_validation(seed=seed),
        "task2_system_readiness": grade_system_readiness(seed=seed),
        "task3_execution": grade_execution(seed=seed),
    }
