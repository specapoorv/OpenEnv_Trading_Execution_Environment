from server.core.tools.bloomberg_pull import execute as bloomberg_pull
from server.core.tools.cancel_order import execute as cancel_order
from server.core.tools.change_broker import execute as change_broker
from server.core.tools.compliance_recheck import execute as compliance_recheck
from server.core.tools.compliance_verify import execute as compliance_verify
from server.core.tools.escalate_issue import execute as escalate_issue
from server.core.tools.get_current_position import execute as get_current_position
from server.core.tools.internal_report_fetch import execute as internal_report_fetch
from server.core.tools.market_status_check import execute as market_status_check
from server.core.tools.oms_position_check import execute as oms_position_check
from server.core.tools.ping_oms_connection import execute as ping_oms_connection
from server.core.tools.restart_strategy import execute as restart_strategy
from server.core.tools.risk_system_check import execute as risk_system_check
from server.core.tools.split_order import execute as split_order
from server.core.tools.strategy_health_check import execute as strategy_health_check
from server.core.tools.submit_order import execute as submit_order

TOOL_REGISTRY = {
    "bloomberg_pull": bloomberg_pull,
    "oms_position_check": oms_position_check,
    "risk_system_check": risk_system_check,
    "compliance_verify": compliance_verify,
    "internal_report_fetch": internal_report_fetch,
    "market_status_check": market_status_check,
    "ping_oms_connection": ping_oms_connection,
    "strategy_health_check": strategy_health_check,
    "compliance_recheck": compliance_recheck,
    "restart_strategy": restart_strategy,
    "escalate_issue": escalate_issue,
    "submit_order": submit_order,
    "split_order": split_order,
    "cancel_order": cancel_order,
    "change_broker": change_broker,
    "get_current_position": get_current_position,
}

__all__ = ["TOOL_REGISTRY"]
