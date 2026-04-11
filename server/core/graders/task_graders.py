from __future__ import annotations

from typing import Dict

from server.core.env.base_state import ScenarioState
from server.core.tasks.task1_data_verification import *
from server.core.tasks.task2_system_monitoring import *
from server.core.tasks.task3_execution_assistance import *


def _bounded_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


# def grade_data_validation(seed: int = 7) -> float:
#     env = ExecutionDeskEnv(seed=seed)
#     observation, info = env.reset(seed=seed)
#     for _ in range(30):
#         action = heuristic_policy(observation, info)
#         observation, _, terminated, truncated, info = env.step(action)
#         if info["completed_flags"]["data_ready"]:
#             return 1.0
#         if terminated or truncated:
#             break
#     score = 1.0 - min(1.0, len(info["data_validation"]["issues"]) / 8.0)
#     return _bounded_score(score)

def grade_data_validation(state: ScenarioState) -> float:
    """
    Grader for Task 1: Data Validation.
    Uses ScenarioState to compare hidden ground truth vs. agent performance.
    """
    # 1. Completion Multiplier (The Gate)
    # If the agent never successfully declared 'data_ready', they fail the process.
    if not state.completed_flags.get("data_ready", False):
        # We give a small partial credit (max 0.3) if they at least called 
        # the right tools, but they didn't finish the task.
        return min(0.3, len(state.tool_call_counts) * 0.05)

    # 2. Identification Accuracy (The Truth Check)
    # Ground Truth: Which anomalies were actually 'is_issue' at start?
    # Note: In your ToolSimulator, simulate_data_tool determines the 'detected' 
    # value based on the base anomalies in the state.
    
    total_anomalies = state.data_anomalies
    critical_keys = [k for k, v in total_anomalies.items() if v.get("is_issue")]
    
    if not critical_keys:
        # If the seed was 'clean', and they declared ready, full marks for accuracy.
        accuracy_score = 1.0
    else:
        # We check how many tools they called to investigate these specific keys.
        # An agent that doesn't call 'bloomberg_pull' can't possibly have 
        # validated a 'bloomberg_pull' anomaly.
        investigated = sum(1 for k in critical_keys if state.tool_call_counts.get(k, 0) > 0)
        accuracy_score = investigated / len(critical_keys)

    # 3. Efficiency & Noise (The Penalty)
    # Penalize for:
    # - Redundant tool calls (calling the same tool when data hasn't changed)
    # - Hallucinated issues (adding notes to issue_log that aren't in truth)
    
    # We allow 1 call per tool in DATA_TOOLS + 2 're-checks' as a grace period.
    expected_calls = len(DATA_TOOLS) + 2
    actual_calls = sum(state.tool_call_counts.values())
    
    # Penalty: 0.05 per excessive tool call, capped at 0.3
    efficiency_penalty = min(0.3, max(0, actual_calls - expected_calls) * 0.05)
    
    # 4. Final Aggregation
    # Formula: Accuracy (70%) + Efficiency (30%)
    final_score = (accuracy_score * 0.7) + ((1.0 - efficiency_penalty) * 0.3)
    
    return round(max(0.0, min(1.0, final_score)), 4)

# def grade_system_readiness(seed: int = 7) -> float:
#     env = ExecutionDeskEnv(seed=seed)
#     observation, info = env.reset(seed=seed)
#     while not info["completed_flags"]["data_ready"]:
#         observation, _, terminated, truncated, info = env.step(heuristic_policy(observation, info))
#         if terminated or truncated:
#             return 0.0
#     for _ in range(20):
#         observation, _, terminated, truncated, info = env.step(heuristic_policy(observation, info))
#         if info["completed_flags"]["systems_ready"] or info["event"].get("correct_escalation"):
#             return 1.0
#         if terminated or truncated:
#             break
#     score = 1.0 - min(1.0, len(info["system_readiness"]["issues"]) / 4.0)
#     return _bounded_score(score)

def grade_system_readiness(state: ScenarioState) -> float:
    """
    Grader for Task 2: System Monitoring & Recovery.
    Evaluates if the agent reached the EXECUTION stage with a healthy system
    or correctly escalated unrecoverable issues.
    """
    system = state.system_truth
    
    # 1. Goal Achievement (The Gate)
    # Success in Task 2 is defined as EITHER:
    # A) systems_ready flag is True (and truth is actually healthy)
    # B) escalated is True (and truth is actually unrecoverable)
    
    declared_ready = state.completed_flags.get("systems_ready", False)
    escalated = state.escalated
    
    # Check Ground Truth
    unresolved = system_unresolved_issues(state)
    is_actually_healthy = (len(unresolved) == 0)
    is_unrecoverable = ("strategy_unrecoverable" in unresolved or 
                        not system["oms_recoverable"])

    # Core logic: Did the agent take the right final action?
    if declared_ready and is_actually_healthy:
        base_score = 1.0
    elif escalated and is_unrecoverable:
        base_score = 1.0  # Correct escalation is a "win"
    elif declared_ready and not is_actually_healthy:
        base_score = 0.2  # Penalty: "False Start" (dangerous in trading)
    elif escalated and not is_unrecoverable:
        base_score = 0.4  # Penalty: "Lazy Escalation" (escalating fixable issues)
    else:
        base_score = 0.0

    # 2. Discovery Quality (The "Due Diligence" Check)
    # Did the agent call all three health check tools at least once?
    required_tools = ["ping_oms_connection", "strategy_health_check", "compliance_recheck"]
    discovery_count = sum(1 for t in required_tools if state.tool_call_counts.get(t, 0) > 0)
    discovery_score = discovery_count / len(required_tools)

    # 3. Efficiency Penalty (The "Looping" Check)
    # If the agent calls 'restart_strategy' but it fails, and they keep calling it 
    # instead of escalating, they are inefficient.
    restart_calls = state.tool_call_counts.get("restart_strategy", 0)
    # Penalty if they tried to restart an unrecoverable strategy more than twice
    efficiency_penalty = 0.0
    if not system["strategy_recoverable"] and restart_calls > 2:
        efficiency_penalty = 0.3

    # 4. Final Aggregation
    # Formula: Outcome (60%) + Discovery (30%) + Efficiency (10%)
    final_score = (base_score * 0.6) + (discovery_score * 0.3) + ((1.0 - efficiency_penalty) * 0.1)
    
    return round(max(0.0, min(1.0, final_score)), 4)


# def grade_execution(seed: int = 7) -> float:
#     env = ExecutionDeskEnv(seed=seed)
#     observation, info = env.reset(seed=seed)
#     terminated = truncated = False
#     while not (terminated or truncated):
#         observation, _, terminated, truncated, info = env.step(heuristic_policy(observation, info))
#         if info["completed_flags"]["execution_complete"]:
#             return 1.0
#     tracking_error = info["execution_status"]["tracking_error"]
#     score = 1.0 - min(1.0, tracking_error / 100.0)
#     return _bounded_score(score)


def grade_execution(state: ScenarioState) -> float:
    """
    Grader for Task 3: Execution Assistance.
    Wraps the quality metrics to provide a final score for the execution phase.
    """
    # 1. Check for Terminal Success
    # Even if the quality is high, if the agent never called declare("execution_complete")
    # while within tolerance, they haven't "finished" the job.
    declared_done = state.completed_flags.get("execution_complete", False)
    
    # 2. Extract Quality Metrics
    # Your existing function provides the heavy lifting here.
    quality_report = grade_execution_quality(state)
    
    # 3. Penalize for "Unfinished Business"
    # If the simulation ended (max steps) but the agent didn't declare complete,
    # we apply a multiplier to the tracking_score because an open-ended trade is a risk.
    final_score = quality_report["final"]
    
    if not declared_done:
        # If tracking error is still high, the quality_report already penalized them.
        # But we add a 20% penalty for not reaching the 'DONE' stage.
        final_score *= 0.8
        
    return round(max(0.0, min(1.0, final_score)), 4)

def run_all_graders(seed: int = 7) -> Dict[str, float]:
    return {
        "task1_data_validation": grade_data_validation(seed=seed),
        "task2_system_readiness": grade_system_readiness(seed=seed),
        "task3_execution": grade_execution(seed=seed),
    }
