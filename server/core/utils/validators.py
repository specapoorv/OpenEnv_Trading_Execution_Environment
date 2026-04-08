from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def staleness_issues(now_minute: int, outputs: Dict[str, Dict], tools: Iterable[str], recency_limit_minutes: int) -> List[str]:
    issues = []
    for tool_name in tools:
        data = outputs.get(tool_name)
        if data and data.get("ok", False):
            if now_minute - int(data.get("timestamp", -999)) > recency_limit_minutes:
                issues.append(f"stale:{tool_name}")
    return issues


def required_field_issues(outputs: Dict[str, Dict], requirements: Iterable[Tuple[str, str]]) -> List[str]:
    issues = []
    for tool_name, field_name in requirements:
        if field_name not in outputs.get(tool_name, {}):
            issues.append(f"missing_field:{tool_name}:{field_name}")
    return issues
