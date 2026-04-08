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

This repository contains an OpenEnv-compatible execution-desk simulator exposed through a FastAPI API and a Gradio UI. It is designed to work as a Hugging Face Docker Space and as a local OpenEnv environment.

The environment models a realistic three-stage execution workflow:

1. Data validation
2. System readiness
3. Order execution

## What You Get

- A mounted UI at `/ui/` for interactive inspection and manual stepping
- OpenEnv HTTP endpoints for programmatic interaction
- A root `inference.py` entrypoint for submission and evaluation workflows
- A root `demo.py` entrypoint for a simple scripted run
- Pluggable tool calls inside the simulator, so the mocked desk tools can later be replaced with real integrations

## Running Locally

Install dependencies:

```bash
uv sync
```

Start the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Local URLs:

- API root: `http://127.0.0.1:7860`
- Gradio UI: `http://127.0.0.1:7860/ui/`
- FastAPI docs: `http://127.0.0.1:7860/docs`

## HTTP Endpoints

The environment exposes the standard OpenEnv-style routes:

- `GET /health`
- `GET /docs`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `WS /ws`

Example health check:

```bash
curl -s http://127.0.0.1:7860/health
```

Example reset:

```bash
curl -X POST http://127.0.0.1:7860/reset
```

Example step:

```bash
curl -X POST http://127.0.0.1:7860/step \
  -H 'Content-Type: application/json' \
  -d '{"action":{"action_type":"CALL_TOOL","tool_name":"market_status_check"}}'
```

## Using The UI

The Gradio UI is mounted at `/ui/`.

It is intended for fast inspection of a live episode without writing client code.

The UI shows:

- Current task metadata
- Latest observation payload
- Latest info payload
- The last submitted action
- Episode reward and termination status
- A step-by-step history table
- A file-upload tab for replaying saved episode logs

In the live tab, you can:

- Reset the environment
- Paste an action as JSON
- Step the simulator manually

The default action example in the UI is:

```json
{"action_type":"CALL_TOOL","tool_name":"market_status_check"}
```

## Action Space

Action model: `ExecutionDeskAction`

Fields:

- `action_type` required
- `tool_name` optional
- `params` optional
- `declare_flag` optional
- `size` optional
- `side` optional
- `broker` optional
- `urgency` optional
- `order_id` optional
- `max_clip` optional

Supported `action_type` values:

- `CALL_TOOL`
- `DECLARE`
- `RESTART_STRATEGY`
- `ESCALATE`
- `SUBMIT_ORDER`
- `SPLIT_ORDER`
- `CANCEL_ORDER`
- `CHANGE_BROKER`

Supported `declare_flag` values:

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

## Desk Tools

The simulator includes task-specific desk tools such as:

- `bloomberg_pull`
- `oms_position_check`
- `risk_system_check`
- `compliance_verify`
- `internal_report_fetch`
- `market_status_check`
- `ping_oms_connection`
- `strategy_health_check`
- `compliance_recheck`
- `submit_order`
- `split_order`
- `cancel_order`
- `change_broker`
- `get_current_position`

These are currently implemented as environment-side tool handlers, but the design is intentionally pluggable. The same action schema can be wired to real internal services, APIs, or broker adapters later without changing the outer API contract.

## Observation Shape

Observation model: `ExecutionDeskObservation`

Top-level fields:

- `observation`
- `info`
- `reward`
- `done`
- `metadata`

Important observation sections include:

- `task_stage`
- `known_data`
- `system_status`
- `compliance_flags`
- `position_state`
- `order_state`
- `timestamps`

The `/state` endpoint returns OpenEnv session state, including:

- `episode_id`
- `step_count`

## Inference And Demo

Submission-oriented inference entrypoint:

```bash
python inference.py
```

Demo entrypoint:

```bash
python demo.py
```

`inference.py`:

- uses the OpenAI Python client
- requires `HF_TOKEN` or `API_KEY`
- has defaults for `API_BASE_URL` and `MODEL_NAME`
- emits logs in the required `[START]`, `[STEP]`, `[END]` format

## Environment Variables

Common variables used by inference:

- `HF_TOKEN`
- `API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`
- `SEED`
- `MAX_STEPS`
- `TEMPERATURE`
- `MAX_TOKENS`

## Validation

Local OpenEnv validation:

```bash
openenv validate
```

Submission pre-check script:

```bash
./validate-submission.sh https://specapoorv-trading-execution-environment.hf.space
```

## Docker Build

```bash
docker build -t trading_env-env:latest -f Dockerfile .
```

## Project Layout

```text
OpenEnv-Trading-env/
├── README.md
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── pyproject.toml
├── inference.py
├── demo.py
├── client.py
├── models.py
├── server/
│   ├── app.py
│   ├── env_adapter.py
│   └── core/
│       ├── env/
│       ├── tasks/
│       ├── tools/
│       ├── graders/
│       └── utils/
└── validate-submission.sh
```
