from __future__ import annotations

from typing import Any, Dict

from server.core.env.base_state import ScenarioState


def evaluate_execution_complete(state: ScenarioState) -> Dict[str, Any]:
    """Gate check: is tracking error within tolerance and no open orders?
    Used by the environment itself to allow DECLARE execution_complete."""
    execution = state.execution_truth
    error = abs(execution["target_position"] - execution["current_position"])
    working_orders = [order for order in state.outstanding_orders.values() if order.status == "working"]
    return {"ready": error <= execution["tolerance"] and not working_orders, "tracking_error": error}


def _cumulative_slippage_bps(state: ScenarioState) -> float:
    """Compute average fill slippage across all fills in the episode."""
    fills = state.execution_truth.get("fills", [])
    if not fills:
        return 0.0
    return sum(f["slippage_bps"] for f in fills) / len(fills)


def grade_execution_quality(state: ScenarioState) -> Dict[str, float]:
    """
    Hard Task 3 grader: scores 0.0–1.0 across four weighted sub-goals.

    Sub-scores
    ----------
    tracking_score  (40%):
        Full credit (1.0) if tracking_error <= tolerance.
        Linearly decays to 0.0 at 3× tolerance (impossible to rescue large misses).

    slippage_score  (30%):
        Full credit if avg slippage <= slippage_budget_bps.
        Linearly decays to 0.0 at 3× the budget.
        This punishes blindly using high-urgency orders.

    risk_score      (20%):
        Full credit (1.0) if the agent never let |position| > risk_limit.
        Hard 0.0 if the risk limit was ever breached.

    step_score      (10%):
        Full credit if execution completed within exec_step_budget steps.
        Linearly decays to 0.5 at 2× the budget (soft penalty, not disqualifying).
        0.0 if execution stage never even started (model never got past stage 2).

    Weights: 0.4, 0.3, 0.2, 0.1 → final in [0.0, 1.0].
    """
    execution = state.execution_truth

    # --- Tracking score ---
    error = abs(execution["target_position"] - execution["current_position"])
    tolerance = max(1, execution["tolerance"])
    tracking_score = max(0.0, 1.0 - error / (3.0 * tolerance))

    # --- Slippage score ---
    avg_slippage = _cumulative_slippage_bps(state)
    budget = execution.get("slippage_budget_bps", 25.0)
    if avg_slippage <= budget:
        slippage_score = 1.0
    else:
        # Decays linearly: 1.0 at budget, 0.0 at 3× budget
        slippage_score = max(0.0, 1.0 - (avg_slippage - budget) / (2.0 * budget))

    # --- Risk score ---
    risk_score = 0.0 if state.risk_limit_breached else 1.0

    # --- Step score ---
    start_step = execution.get("exec_stage_start_step")
    exec_budget = execution.get("exec_step_budget", 25)
    if start_step is None:
        # Execution stage was never entered
        step_score = 0.0
    else:
        steps_used = state.step_count - start_step
        if steps_used <= exec_budget:
            step_score = 1.0
        else:
            # Decays from 1.0 → 0.5 between budget and 2× budget, then stays at 0.5
            step_score = max(0.5, 1.0 - 0.5 * (steps_used - exec_budget) / max(1, exec_budget))

    final = (
        0.40 * tracking_score
        + 0.30 * slippage_score
        + 0.20 * risk_score
        + 0.10 * step_score
    )

    return {
        "tracking_score": round(tracking_score, 4),
        "slippage_score": round(slippage_score, 4),
        "risk_score": round(risk_score, 4),
        "step_score": round(step_score, 4),
        "final": round(min(max(final, 0.0), 1.0), 4),
        # Diagnostics for transparency
        "tracking_error": error,
        "avg_slippage_bps": round(avg_slippage, 4),
        "slippage_budget_bps": budget,
        "risk_limit_breached": state.risk_limit_breached,
        "exec_steps_used": (state.step_count - start_step) if start_step is not None else None,
        "exec_step_budget": exec_budget,
    }
