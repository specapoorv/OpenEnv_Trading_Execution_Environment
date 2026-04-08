---
title: Execution Desk Assistant Environment Server
emoji: 🖱️
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Execution Desk Assistant Environment

This environment simulates a realistic execution desk and is exposed through OpenEnv server/client contracts.

Workflow stages:
1. Data validation
2. System readiness
3. Order execution

---

## Local setup

```bash
cd /home/aditya/openenv/trading_env
uv sync
```

## Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Default URL: `http://127.0.0.1:7860`

Gradio UI: `http://127.0.0.1:7860/ui/`

Useful routes:
- `GET /health`
- `GET /docs`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `WS /ws`

---

## Action space

Action model: `ExecutionDeskAction`

Fields:
- `action_type` (required)
- `tool_name` (optional)
- `params` (optional)
- `declare_flag` (optional)
- `size` (optional)
- `side` (optional)
- `broker` (optional)
- `urgency` (optional)
- `order_id` (optional)
- `max_clip` (optional)

`action_type` values:
- `CALL_TOOL`
- `DECLARE`
- `RESTART_STRATEGY`
- `ESCALATE`
- `SUBMIT_ORDER`
- `SPLIT_ORDER`
- `CANCEL_ORDER`
- `CHANGE_BROKER`

`declare_flag` values:
- `data_ready`
- `systems_ready`
- `execution_complete`

Broker values:
- `broker_alpha`
- `broker_beta`
- `broker_delta`

Urgency values:
- `low`
- `normal`
- `high`

### Tool names (`CALL_TOOL`)

Data tools:
- `bloomberg_pull`
- `oms_position_check`
- `risk_system_check`
- `compliance_verify`
- `internal_report_fetch`
- `market_status_check`

System tools:
- `ping_oms_connection`
- `strategy_health_check`
- `compliance_recheck`
- `restart_strategy`
- `escalate_issue`

Execution tools:
- `submit_order`
- `split_order`
- `cancel_order`
- `change_broker`
- `get_current_position`

---

## Observation and state

Observation model: `ExecutionDeskObservation`

Top-level fields:
- `observation`
- `info`
- `reward`
- `done`
- `metadata`

`observation` payload contains:
- `task_stage` (`data_validation | system_health | execution | done`)
- `known_data` (tool outputs seen so far)
- `system_status` (`oms_connected`, `strategy_status`, `escalated`, `current_broker`)
- `compliance_flags` (`compliance_ok`, `restricted`, `escalation_reason`)
- `position_state` (`current_position`, `target_position`, `tolerance`, `tracking_error`, `recent_slippage_bps`)
- `order_state` (`outstanding_orders`, `open_order_count`, `recent_fills`)
- `timestamps` (`sim_minute`, `step_count`)

`info` payload contains:
- `completed_flags` (`data_ready`, `systems_ready`, `execution_complete`)
- `issue_log`
- `data_validation` (issues and readiness)
- `system_readiness` (issues and readiness)
- `execution_status` (`ready`, `tracking_error`)
- `event` (last transition event)

OpenEnv state endpoint returns session state (`episode_id`, `step_count`).

---

## Reward design

Reward is dense and event-driven:

- Base step cost (small negative).
- Positive:
  - useful tool usage
  - inconsistency detection
  - stage advancement
  - fixing recoverable issues
  - correct escalation
  - meaningful fills
  - lower tracking error
- Negative:
  - redundant tools
  - tool/action failures
  - invalid actions
  - premature declares
  - rejected orders
  - timeout/truncation with unresolved error

Terminal bonus/penalty depends on success vs unresolved issues.

---

## Termination conditions

- `terminated=True` when:
  - stage reaches `done` via successful `DECLARE execution_complete`, or
  - escalation ends the system-health stage.
- `truncated=True` when max steps is reached.

---

## Sample local tests

### 1) Health check

```bash
curl -s http://127.0.0.1:7860/health
```

### 2) Async client smoke test

```bash
uv run --active python - <<'PY'
import asyncio
from trading_env import TradingEnv, ExecutionDeskAction

async def main():
    env = TradingEnv(base_url="http://127.0.0.1:7860")
    try:
        r = await env.reset()
        print("reset stage:", r.observation.observation.get("task_stage"))

        r = await env.step(
            ExecutionDeskAction(action_type="CALL_TOOL", tool_name="market_status_check")
        )
        print("reward:", r.reward)
        print("stage:", r.observation.observation.get("task_stage"))
    finally:
        await env.close()

asyncio.run(main())
PY
```

### 3) Typical stage-1 request sequence

```python
ExecutionDeskAction(action_type="CALL_TOOL", tool_name="bloomberg_pull")
ExecutionDeskAction(action_type="CALL_TOOL", tool_name="oms_position_check")
ExecutionDeskAction(action_type="CALL_TOOL", tool_name="risk_system_check")
ExecutionDeskAction(action_type="CALL_TOOL", tool_name="compliance_verify")
ExecutionDeskAction(action_type="CALL_TOOL", tool_name="internal_report_fetch")
ExecutionDeskAction(action_type="CALL_TOOL", tool_name="market_status_check")
ExecutionDeskAction(action_type="DECLARE", declare_flag="data_ready")
```

Then continue with system checks, declare `systems_ready`, place/split/cancel orders as needed, and finally declare `execution_complete`.

---

## Docker build

```bash
docker build -t trading_env-env:latest -f Dockerfile .
```

---

## Project structure

```text
trading_env/
├── __init__.py
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── README.md
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── requirements.txt
│   └── trading_env_environment.py
└── openenv_quant/
    ├── __init__.py
    ├── app.py
    ├── models.py
    ├── inference.py
    ├── validate_submission.py
    ├── data/
    ├── env/
    ├── tasks/
    ├── tools/
    └── utils/
```
