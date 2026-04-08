from __future__ import annotations

import json
import os
import random
import textwrap
import argparse
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from server.core.env.execution_desk_env import ExecutionDeskEnv, heuristic_policy
from server.core.tasks.task3_execution_assistance import grade_execution_quality
from server.core.utils.constants import ActionType

# =========================
# ENV SETUP
# =========================
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is required (set in environment or HF Spaces secrets).")

SEED = int(os.getenv("SEED", str(random.randint(1, 10000))))
TASK_NAME = os.getenv("TASK_NAME", "execution-desk-assistant")
BENCHMARK = os.getenv("BENCHMARK", "openenv_execution_desk")
MAX_STEPS = int(os.getenv("MAX_STEPS", "60"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "250"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.8"))

# =========================
# ARGPARSE (OPTIONAL REMOTE ENV)
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-url",
        default=os.getenv("OPENENV_SERVER_URL") or os.getenv("ENV_BASE_URL"),
        help="Optional remote environment URL",
    )
    return parser.parse_args()


# =========================
# PROMPT
# =========================
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a three-stage execution desk environment.
    Choose exactly one next action.
    Return valid JSON with this schema:
    {
      "action_type": "CALL_TOOL|DECLARE|RESTART_STRATEGY|ESCALATE|SUBMIT_ORDER|SPLIT_ORDER|CANCEL_ORDER|CHANGE_BROKER",
      "tool_name": "<optional tool>",
      "declare_flag": "<optional flag>",
      "size": <optional integer>,
      "side": "<optional buy|sell>",
      "broker": "<optional broker_alpha|broker_beta|broker_delta>",
      "urgency": "<optional low|normal|high>",
      "order_id": <optional integer>,
      "max_clip": <optional integer>
    }
    Prefer safe, high-value progress. If uncertain, choose a single useful tool call.
    """
).strip()


# =========================
# CLIENT
# =========================
def build_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,   # ✅ FIXED
        timeout=10.0,
        max_retries=2,
    )


# =========================
# LOGGING
# =========================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, grader_scores: Dict[str, float], rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    score_str = ", ".join(f"{k}={v:.3f}" for k, v in grader_scores.items())
    avg_score = sum(grader_scores.values()) / max(1, len(grader_scores))
    print(
        f"[END] success={str(success).lower()} steps={steps} mean_score={avg_score:.3f} | Tasks: {score_str} | rewards={rewards_str}",
        flush=True,
    )


# =========================
# UTILITIES
# =========================
def action_to_string(action: Dict[str, Any]) -> str:
    action_type = str(action.get("action_type"))

    if action_type.endswith(ActionType.CALL_TOOL.value):
        return f"call_tool('{action.get('tool_name', '')}')"

    if action_type.endswith(ActionType.DECLARE.value):
        return f"declare('{action.get('declare_flag', '')}')"

    if action_type.endswith(ActionType.RESTART_STRATEGY.value):
        return "restart_strategy()"

    if action_type.endswith(ActionType.ESCALATE.value):
        return "escalate()"

    if action_type.endswith(ActionType.SUBMIT_ORDER.value):
        return (
            f"submit_order(size={int(action.get('size', 0))},"
            f"side='{action.get('side', 'buy')}',"
            f"broker='{action.get('broker', 'broker_alpha')}',"
            f"urgency='{action.get('urgency', 'normal')}')"
        )

    if action_type.endswith(ActionType.SPLIT_ORDER.value):
        return (
            f"split_order(size={int(action.get('size', 0))},"
            f"side='{action.get('side', 'buy')}',"
            f"broker='{action.get('broker', 'broker_alpha')}',"
            f"urgency='{action.get('urgency', 'normal')}',"
            f"max_clip={int(action.get('max_clip', 0))})"
        )

    if action_type.endswith(ActionType.CANCEL_ORDER.value):
        return f"cancel_order(order_id={int(action.get('order_id', 0))})"

    if action_type.endswith(ActionType.CHANGE_BROKER.value):
        return f"change_broker('{action.get('broker', 'broker_alpha')}')"

    return json.dumps(action, sort_keys=True)


def summarize_for_model(observation: Dict[str, Any], info: Dict[str, Any], step: int) -> str:
    return json.dumps(
        {
            "step": step,
            "task_stage": observation["task_stage"],
            "position_state": observation["position_state"],
            "system_status": observation["system_status"],
            "order_state": {
                "open_order_count": observation["order_state"]["open_order_count"],
                "recent_fills": observation["order_state"]["recent_fills"][-2:],
            },
            "data_validation": info["data_validation"],
            "system_readiness": info["system_readiness"],
            "execution_status": info["execution_status"],
        },
        sort_keys=True,
    )


def parse_model_action(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


# =========================
# MODEL CALL
# =========================
def get_model_action(client: OpenAI, observation, info, step):
    fallback = heuristic_policy(observation, info)

    try:
        obs_summary = summarize_for_model(observation, info, step)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_summary},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        content = (completion.choices[0].message.content or "").strip()
        parsed = parse_model_action(content)

        return parsed if parsed else fallback

    except Exception:
        return fallback


# =========================
# MAIN
# =========================
def main():
    args = parse_args()
    client = build_client()

    # NOTE: env-url not used unless you implement remote env
    env = ExecutionDeskEnv(seed=SEED, max_steps=MAX_STEPS)

    rewards = []
    steps_taken = 0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    obs, info = env.reset(seed=SEED)

    for step in range(1, MAX_STEPS + 1):
        action = get_model_action(client, obs, info, step)

        obs, reward, done, truncated, info = env.step(action)

        rewards.append(reward)
        steps_taken = step

        log_step(step, action_to_string(action), reward, done or truncated, None)

        if done or truncated:
            break

    t3 = grade_execution_quality(env.scenario)

    log_end(
        success=True,
        steps=steps_taken,
        grader_scores=t3,
        rewards=rewards,
    )

    env.close()


if __name__ == "__main__":
    main()
