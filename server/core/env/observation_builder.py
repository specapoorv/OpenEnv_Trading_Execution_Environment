from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from server.core.env.base_state import ScenarioState
from server.core.tasks.task1_data_verification import evaluate_data_readiness
from server.core.tasks.task2_system_monitoring import evaluate_system_readiness
from server.core.tasks.task3_execution_assistance import (
    evaluate_execution_complete,
    grade_execution_quality,
    _cumulative_slippage_bps,
)


def build_observation(state: ScenarioState) -> Dict[str, Any]:
    """
    Converts the full internal environment state into a structured observation dictionary 
    suitable for downstream use 
    The observation includes:
    1. task_stage: 
       - Current workflow stage (data validation, system health, execution, done).

    2. known_data: 
       - Deep copy of all tool outputs collected so far. 
       - Represents the agent's knowledge about data, system checks, and market signals.

    3. system_status: 
       - Key system indicators and environment status:
         * oms_connected: Whether OMS connection is active
         * strategy_status: Health of the trading strategy
         * escalated: Whether an issue has been escalated
         * current_broker: Broker currently in use

    4. compliance_flags:
       - Compliance and restriction information:
         * compliance_ok: Whether compliance checks passed
         * restricted: Whether there are restrictions on trading
         * escalation_reason: Reason for any escalations

    5. position_state:
       - Current market position and tracking metrics:
         * current_position: Current holdings
         * target_position: Desired target position
         * tolerance: Acceptable deviation from target
         * tracking_error: Absolute difference between target and current
         * recent_slippage_bps: Recent slippage in basis points

    6. order_state:
       - Information about outstanding orders:
         * outstanding_orders: List of all order snapshots
         * open_order_count: Number of active/working orders
         * recent_fills: Last few executed fills

    7. timestamps:
       - Temporal information about the simulation:
         * sim_minute: Current simulated minute
         * step_count: Step index in the episode

    Notes:
    - This function does **not modify the internal state**; it only produces a copy for observation.
    - Designed to provide a structured snapshot of the environment for monitoring, analysis, or RL agent observation.
    """

    execution = state.execution_truth
    return {
        "task_stage": state.stage.value,
        "known_data": copy.deepcopy(state.tool_outputs),
        "system_status": {
            "oms_connected": state.system_truth["oms_connected"],
            "strategy_status": state.system_truth["strategy_status"],
            "escalated": state.escalated,
            "current_broker": state.current_broker,
        },
        "compliance_flags": {
            "compliance_ok": state.system_truth["compliance_ok"],
            "restricted": state.data_truth["restricted"],
            "escalation_reason": state.escalation_reason,
        },
        "position_state": {
            "current_position": execution["current_position"],
            "target_position": execution["target_position"],
            "tolerance": execution["tolerance"],
            "tracking_error": abs(execution["target_position"] - execution["current_position"]),
            "recent_slippage_bps": execution["recent_slippage_bps"],
            # Hard Task 3 constraints visible to the agent
            "slippage_budget_bps": execution.get("slippage_budget_bps"),
            "cumulative_slippage_bps": round(_cumulative_slippage_bps(state), 4),
            "exec_step_budget": execution.get("exec_step_budget"),
            "exec_steps_used": (
                state.step_count - execution["exec_stage_start_step"]
                if execution.get("exec_stage_start_step") is not None
                else None
            ),
        },
        "order_state": {
            "outstanding_orders": [order.snapshot() for order in state.outstanding_orders.values()],
            "open_order_count": sum(1 for order in state.outstanding_orders.values() if order.status == "working"),
            "recent_fills": execution["fills"][-5:],
        },
        "timestamps": {
            "sim_minute": state.now_minute,
            "step_count": state.step_count,
        },
    }


def build_info(state: ScenarioState, recency_limit_minutes: int, event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Constructs a diagnostic information dictionary summarizing the environment's current state.

    This function provides additional contextual and meta-information that complements the
    agent's observation. It is typically used for logging, monitoring, or reward computation.

    The returned dictionary includes:

    1. completed_flags:
       - A deep copy of workflow completion flags:
         * data_ready: Whether all data has been validated
         * systems_ready: Whether all system checks passed
         * execution_complete: Whether trade execution tasks are complete

    2. issue_log:
       - List of issues recorded in the environment so far. Useful for debugging and tracking errors.

    3. data_validation:
       - Result of evaluating data readiness using `evaluate_data_readiness(state, recency_limit_minutes)`.
       - Indicates whether required data is fresh and complete within the given recency limit.

    4. system_readiness:
       - Result of evaluating system readiness via `evaluate_system_readiness(state)`.
       - Represents whether all critical systems (OMS, strategy, compliance) are operational.

    5. execution_status:
       - Result of evaluating execution completion using `evaluate_execution_complete(state)`.
       - Provides insight into order fulfillment and trade execution progress.

    6. event:
       - Optional dictionary capturing the most recent external or internal event.
       - Defaults to an empty dictionary if no event is provided.

    Notes:
    - This function does not modify the internal environment state.
    - Primarily intended for diagnostics, logging, or feeding auxiliary information.
    """
    event = event or {}
    exec_status = evaluate_execution_complete(state)
    exec_quality = grade_execution_quality(state)

    return {
        "completed_flags": copy.deepcopy(state.completed_flags),
        "issue_log": list(state.issue_log),
        "data_validation": evaluate_data_readiness(state, recency_limit_minutes),
        "system_readiness": evaluate_system_readiness(state),
        "execution_status": {
            **exec_status,
            # Hard Task 3 diagnostics exposed in info
            "slippage_over_budget": exec_quality["avg_slippage_bps"] > exec_quality["slippage_budget_bps"],
            "risk_limit_breached": state.risk_limit_breached,
            "exec_steps_remaining": (
                max(0, state.execution_truth.get("exec_step_budget", 25) - (
                    state.step_count - state.execution_truth["exec_stage_start_step"]
                ))
                if state.execution_truth.get("exec_stage_start_step") is not None
                else None
            ),
            "quality": exec_quality,
        },
        "event": copy.deepcopy(event),
    }
