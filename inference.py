"""
Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL
    MODEL_NAME
    HF_TOKEN
    LOCAL_IMAGE_NAME

- Defaults are set only for API_BASE_URL and MODEL_NAME.
- The inference script must be named `inference.py` and placed in the root directory.
- OpenAI Client must be used for all LLM calls.

STDOUT FORMAT
- The script emits only:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import httpx
from openai import OpenAI

from server.core.env.execution_desk_env import heuristic_policy
from server.core.utils.constants import ActionType

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
DEFAULT_ENV_URL = "http://127.0.0.1:7860"
SEED = int(os.getenv("SEED", str(random.randint(1, 10000))))
TASK_NAME = os.getenv("TASK_NAME", "execution-desk-assistant")
BENCHMARK = os.getenv("BENCHMARK", "openenv_execution_desk")
MAX_STEPS = int(os.getenv("MAX_STEPS", "60"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "250"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.8"))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-url",
        default=os.getenv("OPENENV_SERVER_URL") or os.getenv("ENV_BASE_URL", DEFAULT_ENV_URL),
    )
    return parser.parse_known_args()[0]


def build_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=2.0, max_retries=0)


class RemoteExecutionDeskEnv:
    def __init__(self, base_url: str) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0, follow_redirects=True)

    @staticmethod
    def _parse_env_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        observation_wrapper = payload.get("observation", {})
        observation = observation_wrapper.get("observation", {})
        info = observation_wrapper.get("info", {})
        reward = float(payload.get("reward", observation_wrapper.get("reward", 0.0)) or 0.0)
        done = bool(payload.get("done", observation_wrapper.get("done", False)))
        return observation, reward, done, info

    def reset(self, seed: int, max_steps: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        response = self._client.post("/reset", json={"seed": seed, "max_steps": max_steps})
        if response.status_code >= 400:
            response = self._client.post("/reset")
        response.raise_for_status()
        observation, _, _, info = self._parse_env_payload(response.json())
        return observation, info

    def step(self, action: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        response = self._client.post("/step", json={"action": action})
        response.raise_for_status()
        return self._parse_env_payload(response.json())

    def close(self) -> None:
        self._client.close()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


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
            f"submit_order(size={int(action.get('size', 0))},side='{action.get('side', 'buy')}',"
            f"broker='{action.get('broker', 'broker_alpha')}',urgency='{action.get('urgency', 'normal')}')"
        )
    if action_type.endswith(ActionType.SPLIT_ORDER.value):
        return (
            f"split_order(size={int(action.get('size', 0))},side='{action.get('side', 'buy')}',"
            f"broker='{action.get('broker', 'broker_alpha')}',urgency='{action.get('urgency', 'normal')}',"
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
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def get_model_action(client: OpenAI, observation: Dict[str, Any], info: Dict[str, Any], step: int) -> Dict[str, Any]:
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
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = parse_model_action(content)
        return {
            "action": parsed if parsed else fallback,
            "model_output": content,
            "model_input": obs_summary,
        }
    except Exception:
        return {
            "action": fallback,
            "model_output": "ERROR",
            "model_input": "",
        }


def extract_error(info: Dict[str, Any]) -> Optional[str]:
    event = info.get("event", {})
    for key in ["last_tool_result", "last_order_result"]:
        payload = event.get(key)
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
    if event.get("premature_declare"):
        return "premature_declare"
    if event.get("invalid_action"):
        return "invalid_action"
    if event.get("bad_escalation"):
        return "bad_escalation"
    return None


def bounded_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def compute_remote_scores(info: Dict[str, Any]) -> Dict[str, float]:
    completed_flags = info.get("completed_flags", {})
    data_validation = info.get("data_validation", {})
    system_readiness = info.get("system_readiness", {})
    execution_status = info.get("execution_status", {})
    event = info.get("event", {})

    data_ready_score = 1.0 if completed_flags.get("data_ready") else 1.0 - min(
        1.0, len(data_validation.get("issues", [])) / 8.0
    )
    systems_ready_score = 1.0 if (
        completed_flags.get("systems_ready") or event.get("correct_escalation")
    ) else 1.0 - min(1.0, len(system_readiness.get("issues", [])) / 4.0)

    if completed_flags.get("execution_complete"):
        execution_score = 1.0
    else:
        tracking_error = float(execution_status.get("tracking_error", 100.0) or 100.0)
        execution_score = 1.0 - min(1.0, tracking_error / 100.0)

    return {
        "task1_data_validation": bounded_score(data_ready_score),
        "task2_system_readiness": bounded_score(systems_ready_score),
        "task3_execution": bounded_score(execution_score),
    }


episode_log = {
    "meta": {
        "task_name": TASK_NAME,
        "benchmark": BENCHMARK,
        "model": MODEL_NAME,
    },
    "steps": [],
}


def main() -> None:
    args = parse_args()
    client = build_client()
    env = RemoteExecutionDeskEnv(args.env_url)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    grader_scores: Dict[str, float] = {}
    success = False
    info: Dict[str, Any] = {}

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    episode_log["meta"]["env_url"] = args.env_url

    try:
        observation, info = env.reset(seed=SEED, max_steps=MAX_STEPS)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            model_result = get_model_action(client, observation, info, step)
            action = model_result["action"]
            model_output = model_result["model_output"]
            model_input = model_result["model_input"]

            observation, reward, done, info = env.step(action)

            step_data = {
                "step": step,
                "observation": {
                    "task_stage": observation.get("task_stage"),
                    "position_state": observation["position_state"],
                    "system_status": observation["system_status"],
                    "order_state": observation["order_state"],
                    "data_validation": info["data_validation"],
                    "execution_status": info["execution_status"],
                },
                "available_actions": [action_type.value for action_type in ActionType],
                "action": action,
                "action_str": action_to_string(action),
                "hidden_state": info,
                "reward": reward,
                "done": done,
                "error": extract_error(info),
                "prompts": {
                    "system": SYSTEM_PROMPT,
                    "user": model_input,
                    "model_output": model_output,
                },
                "grader": {},
            }

            episode_log["steps"].append(step_data)
            with open("episode_log.jsonl", "a") as f:
                f.write(json.dumps(step_data) + "\n")

            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=action_to_string(action),
                reward=reward,
                done=done,
                error=extract_error(info),
            )

            if done:
                break

        grader_scores = compute_remote_scores(info)
        score = sum(grader_scores.values()) / max(1, len(grader_scores))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD and info.get("completed_flags", {}).get("execution_complete", False)
        for step_data in episode_log["steps"]:
            step_data["grader"] = {
                "score": score,
                "passed": success,
                "threshold": SUCCESS_SCORE_THRESHOLD,
            }
    finally:
        try:
            env.close()
        finally:
            episode_log["final"] = {
                "success": success,
                "score": score,
                "task_scores": grader_scores,
                "steps": steps_taken,
                "rewards": rewards,
                "seed_used": SEED,
            }
            with open("episode_log.json", "w") as f:
                json.dump(episode_log, f, indent=2)

            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
