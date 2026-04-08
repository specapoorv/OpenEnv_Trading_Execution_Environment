from __future__ import annotations

from typing import Any, Dict, List

from server.core.env.base_state import ScenarioState
from server.core.utils.constants import DATA_TOOLS
from server.core.utils.consistency_checks import price_consistent, position_consistent
from server.core.utils.validators import required_field_issues, staleness_issues


def evaluate_data_readiness(state: ScenarioState, recency_limit_minutes: int) -> Dict[str, Any]:
    outputs = state.tool_outputs
    issues: List[str] = []
    consistent = True

    for tool_name in DATA_TOOLS:
        data = outputs.get(tool_name)
        if data is None:
            issues.append(f"missing_tool:{tool_name}")
            continue
        if not data.get("ok", False):
            issues.append(f"tool_failure:{tool_name}")

    issues.extend(staleness_issues(state.now_minute, outputs, DATA_TOOLS, recency_limit_minutes))
    issues.extend(
        required_field_issues(
            outputs,
            [
                ("bloomberg_pull", "mid_price"),
                ("bloomberg_pull", "volume"),
                ("oms_position_check", "position"),
                ("risk_system_check", "risk_limit"),
                ("compliance_verify", "restricted"),
                ("market_status_check", "market_open"),
            ],
        )
    )

    if not price_consistent(outputs.get("bloomberg_pull", {}), outputs.get("internal_report_fetch", {})):
        consistent = False
        issues.append("price_mismatch")
    if not position_consistent(outputs.get("oms_position_check", {}), outputs.get("internal_report_fetch", {})):
        consistent = False
        issues.append("position_mismatch")

    return {"ready": not issues and consistent, "issues": issues, "consistent": consistent}
