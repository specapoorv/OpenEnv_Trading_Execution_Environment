from __future__ import annotations

from typing import Any, Dict

from server.core.env.base_state import ScenarioState


class RewardManager:
    def compute(
        self,
        prev_state: ScenarioState,
        new_state: ScenarioState,
        event: Dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> float:
        reward = -0.05
        if event.get("useful_tool"):
            reward += 0.45
        if event.get("redundant_tool"):
            reward -= 0.20
        if event.get("tool_failure"):
            reward -= 0.35
        if event.get("invalid_action"):
            reward -= 0.60
        if event.get("found_inconsistency"):
            reward += 0.20
        if event.get("missed_inconsistency"):
            reward -= 0.25
        if event.get("stage_advanced"):
            reward += 4.0
        if event.get("fixed_issue"):
            reward += 1.0
        if event.get("correct_escalation"):
            reward += 2.5
        if event.get("bad_escalation"):
            reward -= 2.0
        if event.get("premature_declare"):
            reward -= 5.5
        if event.get("order_rejected"):
            reward -= 0.80
        if event.get("cancelled_working_order"):
            reward -= 0.15
        if event.get("executed_fill_size", 0) > 0:
            reward += min(1.25, event["executed_fill_size"] / 80.0)

        prev_error = self._tracking_error(prev_state)
        new_error = self._tracking_error(new_state)
        reward += max(-1.0, min(1.0, (prev_error - new_error) / 40.0))
        reward -= min(2.0, new_error / 500.0)
        reward -= new_state.step_count / max(new_state.max_steps, 1) * 0.03

        if terminated:
            if event.get("success"):
                reward += 12.0
            elif event.get("correct_escalation"):
                reward += 3.0
            else:
                reward -= 8.0 + event.get("unresolved_issues", 0) * 1.5
        if truncated:
            reward -= 7.0 + min(4.0, new_error / 60.0)
        return round(reward, 4)

    def _tracking_error(self, state: ScenarioState) -> float:
        return abs(state.execution_truth["target_position"] - state.execution_truth["current_position"])
