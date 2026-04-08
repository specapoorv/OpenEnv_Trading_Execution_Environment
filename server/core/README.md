# OpenEnv Execution Desk

OpenEnv-compatible environment for a three-stage desk-assistant workflow:

1. Data validation
2. System readiness
3. Order execution

The package is intentionally modular:
- `env/` owns environment orchestration
- `tasks/` owns stage-specific readiness logic
- `tools/` owns individual tool handlers
- `utils/` owns constants, validation, and random helpers
- `data/` stores scenario configuration examples

## Quick Start

```bash
python openenv_quant/demo.py
```

## Required Environment Variables

The submission tooling expects these variables to be defined:

```bash
export API_BASE_URL="http://your-openai-compatible-endpoint/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-api-token"
```

## Submission Commands

Run inference:

```bash
python openenv_quant/inference.py
```

Run local validation:

```bash
python openenv_quant/validate_submission.py
```

## Programmatic Use

```python
from openenv_quant import ExecutionDeskEnv

env = ExecutionDeskEnv(seed=7, max_steps=60)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(
    {"action_type": "CALL_TOOL", "tool_name": "bloomberg_pull"}
)
```

## Notes

- Works with `openenv`, `gymnasium`, or a small built-in fallback base/space layer.
- Uses deterministic seeding through Python's `random.Random`.
- Includes a simple heuristic agent in `demo.py` for smoke testing.

## Submission Notes

- Real-world utility: models a realistic execution-desk workflow across data validation, system readiness, and order execution rather than a toy single-step task.
- Tasks and graders: includes 3 graders with deterministic seeded evaluation and scores clamped to `[0.0, 1.0]`.
- Reward design: uses dense shaping for useful progress, penalties for waste/failures, and terminal bonuses/penalties.
- Validation: `validate_submission.py` checks env vars, OpenEnv spec file, API surface, inference script framing, and grader determinism/range.
- External checks still needed on the final host: actual Hugging Face Space deployment, actual Docker build/run on a machine with Docker socket access, and any official `openenv validate` run if your target platform requires the installed CLI.
