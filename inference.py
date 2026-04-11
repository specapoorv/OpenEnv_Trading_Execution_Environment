from __future__ import annotations

import argparse
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
#LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

DEFAULT_ENV_URL = "https://specapoorv-trading-execution-environment.hf.space"

SEED = int(os.getenv("SEED", str(random.randint(1, 10000))))
TASK_NAME = os.getenv("TASK_NAME", "execution-desk-assistant")
BENCHMARK = os.getenv("BENCHMARK", "openenv_execution_desk")

MAX_STEPS = int(os.getenv("MAX_STEPS", "60"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))

SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.8"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a trading execution environment.

    Choose EXACTLY ONE action.

    Output STRICTLY one action string in this format:
    - call_tool(tool_name)
    - declare(flag)
    - restart_strategy()
    - escalate()
    - submit_order(size,side,broker,urgency)
    - split_order(size,side,broker,urgency,max_clip)
    - cancel_order(order_id)
    - change_broker(broker)

    No JSON. No explanations. Only the action string.
    """
).strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-url",
        default=os.getenv("OPENENV_SERVER_URL")
        or os.getenv("ENV_BASE_URL", DEFAULT_ENV_URL),
    )
    return parser.parse_known_args()[0]


def build_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("Missing HF_TOKEN / API_KEY")
    #if not LOCAL_IMAGE_NAME:
        #raise RuntimeError("Missing LOCAL_IMAGE_NAME")

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=10.0,
        max_retries=2,
    )


class RemoteExecutionDeskEnv:
    def __init__(self, base_url: str):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=30.0,
            follow_redirects=True,
        )

    def _parse(self, payload):
        obs_wrap = payload.get("observation", {})
        obs = obs_wrap.get("observation", {})
        info = obs_wrap.get("info", {})
        reward = float(payload.get("reward", 0.0) or 0.0)
        done = bool(payload.get("done", False))
        return obs, reward, done, info

    def reset(self, seed, max_steps):
        r = self._client.post("/reset", json={"seed": seed, "max_steps": max_steps})
        if r.status_code >= 400:
            r = self._client.post("/reset")
        r.raise_for_status()
        obs, _, _, info = self._parse(r.json())
        return obs, info

    def step(self, action):
        r = self._client.post("/step", json={"action": action})
        r.raise_for_status()
        return self._parse(r.json())

    def close(self):
        self._client.close()


# ---------------- Logging ----------------

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(task, success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------- Action Parsing ----------------

def action_str_to_dict(action_str: str) -> Dict[str, Any]:
    try:
        if action_str.startswith("call_tool"):
            name = action_str[action_str.find("(")+1:-1]
            return {"action_type": ActionType.CALL_TOOL.value, "tool_name": name}

        if action_str.startswith("declare"):
            flag = action_str[action_str.find("(")+1:-1]
            return {"action_type": ActionType.DECLARE.value, "declare_flag": flag}

        if action_str.startswith("restart_strategy"):
            return {"action_type": ActionType.RESTART_STRATEGY.value}

        if action_str.startswith("escalate"):
            return {"action_type": ActionType.ESCALATE.value}

        if action_str.startswith("submit_order"):
            args = action_str[action_str.find("(")+1:-1].split(",")
            return {
                "action_type": ActionType.SUBMIT_ORDER.value,
                "size": int(args[0]),
                "side": args[1],
                "broker": args[2],
                "urgency": args[3],
            }

        if action_str.startswith("split_order"):
            args = action_str[action_str.find("(")+1:-1].split(",")
            return {
                "action_type": ActionType.SPLIT_ORDER.value,
                "size": int(args[0]),
                "side": args[1],
                "broker": args[2],
                "urgency": args[3],
                "max_clip": int(args[4]),
            }

        if action_str.startswith("cancel_order"):
            oid = int(action_str[action_str.find("(")+1:-1])
            return {"action_type": ActionType.CANCEL_ORDER.value, "order_id": oid}

        if action_str.startswith("change_broker"):
            broker = action_str[action_str.find("(")+1:-1]
            return {"action_type": ActionType.CHANGE_BROKER.value, "broker": broker}

    except Exception:
        pass

    return {}


def extract_error(info):
    event = info.get("event", {})
    for k in ["last_tool_result", "last_order_result"]:
        if isinstance(event.get(k), dict) and event[k].get("error"):
            return str(event[k]["error"])
    if event.get("invalid_action"):
        return "invalid_action"
    return None


def summarize(observation, info, step):
    return str(
        {
            "step": step,
            "stage": observation.get("task_stage"),
            "orders": observation["order_state"]["open_order_count"],
            "issues": info.get("data_validation"),
        }
    )


# ---------------- Main ----------------

def main():
    args = parse_args()
    client = build_client()
    env = RemoteExecutionDeskEnv(args.env_url)
    task_tiers = ["easy", "medium", "hard"]
    #each id needs to be there in openenv yaml
    task_map = {
        "easy": "task1_data",
        "medium": "task2_system",
        "hard": "task3_execution"
    }

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    for task_id in task_tiers:
        log_start(task_id, BENCHMARK, MODEL_NAME)
        final_task_score = 0.0  # Reset for each tier

        try:
            observation, info = env.reset(SEED, MAX_STEPS)
            done = False

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                prompt = summarize(observation, info, step)

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    action_str = (completion.choices[0].message.content or "").strip()
                except Exception:
                    action_str = ""

                action = action_str_to_dict(action_str)
                if not action:
                    action = heuristic_policy(observation, info)
                    action_str = "fallback"

                observation, reward, done, info = env.step(action)

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step,
                    action_str,
                    reward,
                    done,
                    extract_error(info),
                )

                if done:
                    break

            #remove this, this was averaging reward not using graders
            # score = min(max(sum(rewards) / max(1, len(rewards)), 0.0), 1.0)
            scores = info.get("grades", {})
            final_task_score = scores.get(task_map[task_id], 0.0)
            success = final_task_score >= SUCCESS_SCORE_THRESHOLD
            
            log_end(task_id, success, steps_taken, final_task_score, rewards)

        finally:
            env.close()
            log_end(task_id, success, steps_taken, final_task_score, rewards)



if __name__ == "__main__":
    main()
