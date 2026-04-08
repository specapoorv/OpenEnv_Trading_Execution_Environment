from __future__ import annotations

from typing import Any, Dict, Tuple

from server.core.env.base_state import ScenarioState
from server.core.tasks.task1_data_verification import evaluate_data_readiness
from server.core.tasks.task2_system_monitoring import system_unresolved_issues
from server.core.tasks.task3_execution_assistance import evaluate_execution_complete
from server.core.utils.constants import ACTION_TYPES, ALL_TOOLS, BROKERS, DECLARE_FLAGS, URGENCY_LEVELS


def scalar(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def normalize_action(action: Dict[str, Any], state: ScenarioState) -> Dict[str, Any]:
    from server.core.utils.constants import ActionType

    action_type = action.get("action_type", ActionType.CALL_TOOL)
    if not isinstance(action_type, ActionType):
        if isinstance(action_type, int):
            action_type = ACTION_TYPES[action_type]
        else:
            action_type = ActionType(str(action_type))

    normalized = {
        "action_type": action_type,
        "tool_name": action.get("tool_name"),
        "params": action.get("params", {}),
        "declare_flag": action.get("declare_flag"),
        "size": int(scalar(action.get("size", 0))),
        "side": action.get("side", "buy"),
        "broker": action.get("broker", state.current_broker),
        "urgency": action.get("urgency", "normal"),
        "order_id": int(scalar(action.get("order_id", 0))),
        "max_clip": int(max(1, scalar(action.get("max_clip", 50)))),
    }

    if isinstance(normalized["tool_name"], int):
        index = normalized["tool_name"] - 1
        normalized["tool_name"] = ALL_TOOLS[index] if 0 <= index < len(ALL_TOOLS) else None
    if isinstance(normalized["declare_flag"], int):
        index = normalized["declare_flag"] - 1
        normalized["declare_flag"] = DECLARE_FLAGS[index] if 0 <= index < len(DECLARE_FLAGS) else None
    if isinstance(normalized["side"], int):
        normalized["side"] = "buy" if normalized["side"] == 0 else "sell"
    if isinstance(normalized["broker"], int):
        normalized["broker"] = BROKERS[normalized["broker"] % len(BROKERS)]
    if isinstance(normalized["urgency"], int):
        normalized["urgency"] = URGENCY_LEVELS[normalized["urgency"] % len(URGENCY_LEVELS)]
    return normalized


def check_terminal_conditions(state: ScenarioState, recency_limit_minutes: int) -> Tuple[bool, bool, Dict[str, Any]]:
    event: Dict[str, Any] = {}
    from server.core.utils.constants import Stage

    if state.stage == Stage.DONE:
        return True, False, {"success": True}
    if state.escalated and state.stage == Stage.SYSTEM_HEALTH:
        unresolved = system_unresolved_issues(state)
        return True, False, {"correct_escalation": bool(unresolved), "unresolved_issues": len(unresolved)}
    if state.step_count >= state.max_steps:
        unresolved = 0
        if state.stage == Stage.DATA_VALIDATION:
            unresolved = len(evaluate_data_readiness(state, recency_limit_minutes)["issues"])
        elif state.stage == Stage.SYSTEM_HEALTH:
            unresolved = len(system_unresolved_issues(state))
        else:
            unresolved = 0 if evaluate_execution_complete(state)["ready"] else 1
        event["unresolved_issues"] = unresolved
        return False, True, event
    return False, False, event
