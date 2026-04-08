from __future__ import annotations

from typing import Any, Dict, List

from server.core.env.base_state import ScenarioState


def system_unresolved_issues(state: ScenarioState) -> List[str]:
    system = state.system_truth
    issues = []
    if not system["oms_connected"]:
        issues.append("oms_disconnected")
    if system["strategy_status"] != "running":
        issues.append("strategy_unrecoverable" if not system["strategy_recoverable"] else "strategy_not_running")
    if not system["compliance_ok"]:
        issues.append("compliance_violation")
    return issues


def evaluate_system_readiness(state: ScenarioState) -> Dict[str, Any]:
    outputs = state.tool_outputs
    issues = []
    for tool_name in ["ping_oms_connection", "strategy_health_check", "compliance_recheck"]:
        if tool_name not in outputs:
            issues.append(f"missing_tool:{tool_name}")
    if not state.system_truth["oms_connected"]:
        issues.append("oms_disconnected")
    if state.system_truth["strategy_status"] != "running":
        issues.append("strategy_not_running")
    if not state.system_truth["compliance_ok"]:
        issues.append("compliance_violation")
    return {"ready": not issues, "issues": issues}
