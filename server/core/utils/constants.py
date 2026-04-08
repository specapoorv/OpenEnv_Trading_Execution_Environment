from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    DATA_VALIDATION = "data_validation"
    SYSTEM_HEALTH = "system_health"
    EXECUTION = "execution"
    DONE = "done"


class ActionType(str, Enum):
    CALL_TOOL = "CALL_TOOL"
    DECLARE = "DECLARE"
    RESTART_STRATEGY = "RESTART_STRATEGY"
    ESCALATE = "ESCALATE"
    SUBMIT_ORDER = "SUBMIT_ORDER"
    SPLIT_ORDER = "SPLIT_ORDER"
    CANCEL_ORDER = "CANCEL_ORDER"
    CHANGE_BROKER = "CHANGE_BROKER"


STAGE_TO_ID = {
    Stage.DATA_VALIDATION: 0,
    Stage.SYSTEM_HEALTH: 1,
    Stage.EXECUTION: 2,
    Stage.DONE: 3,
}

ACTION_TYPES = list(ActionType)
DECLARE_FLAGS = ["data_ready", "systems_ready", "execution_complete"]
BROKERS = ["broker_alpha", "broker_beta", "broker_delta"]
URGENCY_LEVELS = ["low", "normal", "high"]

DATA_TOOLS = [
    "bloomberg_pull",
    "oms_position_check",
    "risk_system_check",
    "compliance_verify",
    "internal_report_fetch",
    "market_status_check",
]

SYSTEM_TOOLS = [
    "ping_oms_connection",
    "strategy_health_check",
    "compliance_recheck",
    "restart_strategy",
    "escalate_issue",
]

EXECUTION_TOOLS = [
    "submit_order",
    "split_order",
    "cancel_order",
    "change_broker",
    "get_current_position",
]

ALL_TOOLS = DATA_TOOLS + SYSTEM_TOOLS + EXECUTION_TOOLS
