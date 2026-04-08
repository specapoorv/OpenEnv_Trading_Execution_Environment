# Hard Task 3: Constrained Execution Under Budget & Risk Limits

## Background

Currently Task 3 only checks `tracking_error <= tolerance` and no open orders. A capable frontier model like Qwen2.5-72B solves this nearly perfectly every run. We need to make the task **genuinely difficult** by adding multi-objective constraints that conflict with each other.

## What Makes a Task Hard for an LLM Agent?

1. **Multi-objective tension** – goals that trade off against each other (e.g. urgency fills fast but burns slippage budget)
2. **Partial observability** – agent must actively probe to learn constraints it was not told upfront
3. **Dynamic risk limits** – constraints tighten mid-episode, so a greedy policy fails

## Proposed Hard Constraints for Task 3

Three new constraints are introduced **on top of** the existing tracking-error requirement:

| Constraint | Detail |
|---|---|
| **Slippage Budget** | Total cumulative slippage across all fills must remain ≤ `slippage_budget_bps` (randomly set 15–35 bps per unit). Exceeding it makes the grader heavily penalise the score. |
| **Risk Limit Cap** | At no point can `|current_position|` exceed `risk_limit`. The agent must call `risk_system_check` mid-execution to monitor this. Breaching it reduces the grader score. |
| **Step Budget** | Execution must complete within `exec_step_budget` steps (randomly 20–30 steps out of max 60). Finishing with excess steps burns a proportional score penalty. |

These three sub-scores combine into the final Task 3 grade as a weighted average (0.0–1.0).

## Proposed Changes

---

### Environment State ([base_state.py](file:///home/aditya/openenv/server/core/env/base_state.py))

#### [MODIFY] [base_state.py](file:///home/aditya/openenv/server/core/env/base_state.py)

Add three fields to `execution_truth` (populated at scenario init):
- `slippage_budget_bps: float` – max cumulative avg slippage allowed (random 15–35)
- `exec_step_budget: int` – max steps permitted for the entire execution stage (random 20–30)
- `exec_stage_start_step: int` – step counter when execution stage began (set when stage advances)

---

### Scenario Init ([execution_desk_env.py](file:///home/aditya/openenv/server/core/env/execution_desk_env.py))

#### [MODIFY] [execution_desk_env.py](file:///home/aditya/openenv/server/core/env/execution_desk_env.py)

In `ToolSimulator.initialize_scenario`: add `slippage_budget_bps` and `exec_step_budget` to `execution_truth`. Record `exec_stage_start_step` when the stage transitions to `EXECUTION` in [_handle_declare](file:///home/aditya/openenv/server/core/env/execution_desk_env.py#418-451).

---

### Task 3 Grader ([task3_execution_assistance.py](file:///home/aditya/openenv/server/core/tasks/task3_execution_assistance.py))

#### [MODIFY] [task3_execution_assistance.py](file:///home/aditya/openenv/server/core/tasks/task3_execution_assistance.py)

Rewrite [evaluate_execution_complete](file:///home/aditya/openenv/server/core/tasks/task3_execution_assistance.py#8-13) to:
1. Check tracking error (unchanged — needed to allow `DECLARE execution_complete`)
2. Expose the three sub-scores for the external grader to consume

Add a new standalone function `grade_execution_quality(state) -> Dict` that computes:
- `tracking_score` = `max(0, 1 - tracking_error / (2 * tolerance))`
- `slippage_score` = `1.0` if cumulative slippage ≤ budget, else decays linearly to 0 at 3× budget
- `risk_score` = `1.0` if risk limit never exceeded, else 0.5 (binary breach flag stored on state)
- `step_score` = `1.0` if completed within step budget, else linearly decays to 0.5 at 2× budget
- Weighted final: `0.4 * tracking + 0.3 * slippage + 0.2 * risk + 0.1 * step`

---

### State ([base_state.py](file:///home/aditya/openenv/server/core/env/base_state.py))

#### [MODIFY] [base_state.py](file:///home/aditya/openenv/server/core/env/base_state.py)

Add `risk_limit_breached: bool = False` to [ScenarioState](file:///home/aditya/openenv/server/core/env/base_state.py#54-80) so the grader can see if the agent ever violated the risk limit.

---

### Execution Desk Env — Risk Limit Monitoring

#### [MODIFY] [execution_desk_env.py](file:///home/aditya/openenv/server/core/env/execution_desk_env.py)

In [_apply_fill](file:///home/aditya/openenv/server/core/env/execution_desk_env.py#284-311), after updating `current_position`, check if `|current_position| > risk_limit`. If so, set `state.risk_limit_breached = True`.

---

### Observation Builder ([observation_builder.py](file:///home/aditya/openenv/server/core/env/observation_builder.py))

#### [MODIFY] [observation_builder.py](file:///home/aditya/openenv/server/core/env/observation_builder.py)

Expose the new constraints in the observation so the agent CAN see them (but must interpret them):
- `position_state.slippage_budget_bps` 
- `position_state.cumulative_slippage_bps` (computed from fills)
- `position_state.exec_step_budget`
- `position_state.exec_steps_used`

Expose in [build_info](file:///home/aditya/openenv/server/core/env/observation_builder.py#95-142) → `execution_status`:
- `slippage_over_budget: bool`
- `risk_limit_breached: bool`
- `exec_steps_remaining: int`

---

### Inference Script Grader ([inference.py](file:///home/aditya/openenv/server/core/inference.py))

#### [MODIFY] [inference.py](file:///home/aditya/openenv/server/core/inference.py)

Replace the binary `execution_score` with a call to `grade_execution_quality(env.scenario)` after the episode ends, to get the weighted 0.0–1.0 score.

---

## Verification Plan

### Automated Smoke Test
Run the heuristic policy (which is greedy and will violate slippage) and confirm it no longer scores 1.0 on Task 3:
```bash
cd /home/aditya/openenv
uv run python3 -c "
from trading_env.server.core.env.execution_desk_env import ExecutionDeskEnv, heuristic_policy
from trading_env.server.core.tasks.task3_execution_assistance import grade_execution_quality

env = ExecutionDeskEnv(seed=7)
obs, info = env.reset(seed=7)
terminated = truncated = False
while not (terminated or truncated):
    action = heuristic_policy(obs, info)
    obs, _, terminated, truncated, info = env.step(action)
scores = grade_execution_quality(env.scenario)
print('Task 3 grade:', scores)
assert scores['final'] < 1.0, 'Heuristic should not score 1.0 on hard constraints'
print('PASS')
"
```

### Manual Inference Run
```bash
uv run python3 server/core/inference.py
```
Confirm the `[END]` line shows `task3_execution` < 1.0 for at least some seeds.
