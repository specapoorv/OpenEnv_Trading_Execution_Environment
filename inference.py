from __future__ import annotations
import json 
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
from uuid import uuid4


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
You are controlling a three-stage trading execution desk environment.
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
    Follow strict priority:

    1. If data_validation is not complete:
    - ONLY call tools to fix data
    - Do NOT restart strategy or escalate

    2. If data is complete but system_readiness is not:
    - Fix system issues (restart_strategy, escalate, etc.)

    3. Only when both are complete:
    - Perform execution actions

    Never skip steps.
    """
).strip()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

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


# class RemoteExecutionDeskEnv:
#     def __init__(self, base_url: str):
#         self._client = httpx.Client(
#             base_url=base_url.rstrip("/"),
#             timeout=30.0,
#             follow_redirects=True,
#         )
#         self.session_id = str(uuid4())

#     def _parse(self, payload):
#         obs_wrap = payload.get("observation", {})
#         obs = obs_wrap.get("observation", {})
#         info = obs_wrap.get("info", {})
#         reward = float(payload.get("reward", 0.0) or 0.0)
#         done = bool(payload.get("done", False))
#         return obs, reward, done, info

#     def reset(self, seed, max_steps, task_id: str):
#         r = self._client.post("/reset", json={
#             "seed": seed, 
#             "max_steps": max_steps,
#             "task_id": task_id,
#             "session_id": self.session_id
#             })
#         if r.status_code >= 400:
#             r = self._client.post("/reset")
#         r.raise_for_status()
#         obs, _, _, info = self._parse(r.json())
#         return obs, info

#     def step(self, action):
#         r = self._client.post("/step", json={"action": action, "session_id":self.session_id})
#         r.raise_for_status()
#         return self._parse(r.json())

#     def close(self):
#         self._client.close()

import asyncio
import websockets
import json

class RemoteExecutionDeskEnv:
    def __init__(self, base_url: str):
        # Convert http(s) to ws(s)
        self.ws_url = base_url.replace("http", "ws").rstrip("/") + "/ws"
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url)


    def _parse(self, payload):
        dprint("\n" + "="*60)
        dprint("PARSER RECEIVED RAW DATA:")
        dprint(json.dumps(payload, indent=2))
        dprint("="*60 + "\n")

        # 1. Unpack the outer Pydantic/API wrapper
        data = payload.get("data", payload)
        
        # 2. Extract initial candidates
        # In your JSON, 'info' lives inside 'data.observation'
        obs_container = data.get("observation", {})
        reward = float(data.get("reward", 0.0) or 0.0)
        done = bool(data.get("done", False))
        
        # Initialize info
        info = data.get("info", {})

        # 3. RECURSIVE STRIP
        # We want to find the dict that contains 'task_stage'
        # but we must extract 'info' if we see it along the way
        current = obs_container
        max_depth = 5
        
        for _ in range(max_depth):
            if not isinstance(current, dict):
                break
                
            # If we found the target, stop
            if "task_stage" in current:
                break
                
            # Capture 'info' if it exists at this nesting level
            # In your JSON, info is at the same level as the nested observation
            if isinstance(current.get("info"), dict) and current["info"]:
                info = current["info"]

            # Dive deeper if there's another 'observation' key
            if "observation" in current:
                current = current["observation"]
            else:
                break

        obs = current

        # --- FINAL SAFETY ---
        if isinstance(obs, dict) and "task_stage" not in obs and "task_stage" in data:
            obs = data

        dprint(f"Final Obs Keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
        dprint(f"Info Keys: {list(info.keys()) if isinstance(info, dict) else 'Empty'}")
        return obs, reward, done, info


    async def step(self, action):
        dprint("i am in step doing", action)
        try:
            # 1. dprint the action we are about to send
            dprint(f"\n[CLIENT] Attempting step with action: {action.get('action_type')}")
            
# CORRECT (Flattened)
            payload = {
                "type": "step",
                "data": action  # This puts 'action_type' and 'tool_name' directly under 'data'
            }
                        
            # 2. Send and wait
            await self.ws.send(json.dumps(payload))
            
            dprint("[CLIENT] Waiting for response...")
            resp = await self.ws.recv()
            
            # 3. dprint the raw string immediately
            dprint("\n" + "!"*60)
            dprint("RAW RESPONSE RECEIVED:")
            dprint(resp)
            dprint("!"*60 + "\n")
            
            return self._parse(json.loads(resp))

        except Exception as e:
            dprint(f"\n[CRITICAL ERROR IN STEP]: {e}")
            # dprint the traceback to see exactly which line failed
            import traceback
            traceback.print_exc()
            raise e

    async def reset(self, seed, max_steps, task_id: str):
        await self.connect()
        
        # We wrap the parameters inside 'data'
        payload = {
            "type": "reset",
            "data": {
                "seed": seed,
                "max_steps": max_steps,
                "task_id": task_id
            }
        }
        
        dprint(f"[CLIENT] Sending Reset Payload...")
        await self.ws.send(json.dumps(payload))
        resp = await self.ws.recv()
        
        obs, _, _, info = self._parse(json.loads(resp))
        return obs, info

    async def close(self):
        if self.ws:
            await self.ws.close()

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

def summarize_for_model(observation: Dict[str, Any], info: Dict[str, Any], step: int) -> str:
    # Use .get() to avoid KeyErrors if the environment sends partial data
    order_state = observation.get("order_state", {})
    summary_dict = {
        "step": step,
        "task_stage": observation.get("task_stage", "unknown"),
        "position_state": observation.get("position_state", {}),
        "system_status": observation.get("system_status", {}),
        "order_state": {
            "open_order_count": order_state.get("open_order_count", 0),
            "recent_fills": order_state.get("recent_fills", [])[-2:],
        },
        # Info might be empty if the parser missed it or the API didn't send it
        "data_validation": info.get("data_validation", "No data validation info"),
        "system_readiness": info.get("system_readiness", "No readiness info"),
        "execution_status": info.get("execution_status", {}),
    }
    dprint("-------------------------")

    
    return json.dumps(summary_dict, sort_keys=True, indent=2)

def get_model_action(client: OpenAI, observation: Dict[str, Any], info: Dict[str, Any], step: int) -> Dict[str, Any]:
    fallback = heuristic_policy(observation, info)
    dprint("I GETTIGN THE MODEL ACTION ")
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
        dprint(content)
        dprint(parsed)
        return {
            "action": parsed if parsed else fallback,
            }
    except Exception:
        return {
            "action": fallback,
            "model_output": "ERROR",
            "model_input": ""
        }
# ---------------- Main ----------------

async def main():
    args = parse_args()
    client = build_client()
    env = RemoteExecutionDeskEnv(args.env_url)   

    task_tiers = ["easy", "medium", "hard"]
    task_map = {
        "easy": "task1_data",
        "medium": "task2_system",
        "hard": "task3_execution"
    }

    try:
        for task_id in task_tiers:
            rewards = []          
            steps_taken = 0
            success = False
            final_task_score = 0.0

            log_start(task_id, BENCHMARK, MODEL_NAME)

            try:
                observation, info = await env.reset(SEED, MAX_STEPS, task_id)
                done = False

                for step in range(1, MAX_STEPS + 1):
                    if done:
                        break

                    try:
                        model_result = get_model_action(client, observation, info, step)
                        dprint(model_result)
                    except Exception as e:
                        dprint(f"[CRASH in get_model_action]: {e}")
                        # Let's see what keys WERE in observation to see why it failed
                        dprint(f"Observation keys: {list(observation.keys()) if isinstance(observation, dict) else 'Not a dict'}")
                        break
                    action = model_result["action"]
                    dprint("final action to send is", action)
                    dprint("DEBUG: Right before await")
                    try:
                        # Force a timeout to see if it even attempts to start
                        dprint("trying")
                        dprint("Testing event loop heartbeat...")
                        await asyncio.sleep(0.1)
                        dprint("Heartbeat okay. Now entering step...")

                        observation, reward, done, info = await asyncio.wait_for(env.step(action), timeout=5.0)
                        dprint(observation)
                    except asyncio.TimeoutError:
                        dprint("CRITICAL: await env.step(action) timed out before even dprinting inside the function!")
                    except Exception as e:
                        dprint(f"Caught early error: {e}")
                    rewards.append(reward)
                    steps_taken = step

                    log_step(
                        step,
                        action_to_string(action),   
                        reward,
                        done,
                        extract_error(info),
                    )

                    if done:
                        break

                scores = info.get("grades", {})
                final_task_score = scores.get(task_map[task_id], 0.0)
                success = final_task_score >= SUCCESS_SCORE_THRESHOLD

                log_end(task_id, success, steps_taken, final_task_score, rewards)

            except Exception as e:
                print(f"An error occurred: {e}")
                continue
    finally:
        await env.close() 
    
if __name__ == "__main__":
    asyncio.run(main())
