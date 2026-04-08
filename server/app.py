"""FastAPI app with an integrated Gradio frontend for the execution desk env."""

from __future__ import annotations

import json
import threading
from typing import Any, Dict
from uuid import uuid4

import gradio as gr
from fastapi.responses import RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ExecutionDeskAction, ExecutionDeskObservation
    from .env_adapter import EnvAdapter
except ImportError:
    from models import ExecutionDeskAction, ExecutionDeskObservation
    from server.env_adapter import EnvAdapter


class SessionRegistry:
    """In-memory Gradio session registry keyed by browser session id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, EnvAdapter] = {}

    def get(self, session_id: str) -> EnvAdapter:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = EnvAdapter()
            return self._sessions[session_id]

    def reset(self, session_id: str) -> EnvAdapter:
        with self._lock:
            self._sessions[session_id] = EnvAdapter()
            return self._sessions[session_id]


SESSIONS = SessionRegistry()


def _default_state() -> Dict[str, Any]:
    return {
        "session_id": None,
        "source": "live",
        "steps": [],
        "meta": {
            "task_name": "execution-desk-assistant",
            "benchmark": "openenv_execution_desk",
            "model": "interactive-user",
        },
        "final": {},
    }


def _normalize_live_step(
    obs: ExecutionDeskObservation,
    step_number: int,
    action_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    action_payload = action_payload or {}
    return {
        "step": step_number,
        "observation": obs.observation or {},
        "info": obs.info or {},
        "action": action_payload,
        "action_str": json.dumps(action_payload, indent=2) if action_payload else "reset()",
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
        "metadata": obs.metadata or {},
    }


def _load_episode(file_obj, app_state: Dict[str, Any]):
    state = dict(app_state or _default_state())
    if file_obj is None:
        return _render(state)

    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        with open(path, encoding="utf-8") as handle:
            raw = handle.read().strip()

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        steps = []
        meta = {}
        final = {}

        if len(lines) == 1:
            obj = json.loads(lines[0])
            if "steps" in obj:
                steps = obj["steps"]
                meta = obj.get("meta", {})
                final = obj.get("final", {})
            elif "step" in obj:
                steps = [obj]
            else:
                raise ValueError("Unrecognized JSON structure.")
        else:
            for line in lines:
                obj = json.loads(line)
                if "step" in obj:
                    steps.append(obj)
                elif "meta" in obj and not meta:
                    meta = obj["meta"]
                elif ("success" in obj or "score" in obj) and not final:
                    final = obj

        if not steps:
            raise ValueError("No step data found.")

        rewards = [float(step.get("reward", 0.0) or 0.0) for step in steps]
        state.update(
            {
                "source": "file",
                "steps": steps,
                "meta": meta or state["meta"],
                "final": final
                or {
                    "success": bool(steps[-1].get("done", False)),
                    "score": round(sum(rewards), 4),
                    "steps": len(steps),
                    "rewards": rewards,
                },
            }
        )
        return _render(state)
    except Exception as exc:
        return _render(state, status=f"File load failed: {exc}")


def _ensure_session(state: Dict[str, Any]) -> tuple[Dict[str, Any], EnvAdapter]:
    if not state.get("session_id"):
        state["session_id"] = str(uuid4())
    return state, SESSIONS.get(state["session_id"])


def _reset_live(state: Dict[str, Any]):
    state = dict(state or _default_state())
    if not state.get("session_id"):
        state["session_id"] = str(uuid4())
    env = SESSIONS.reset(state["session_id"])
    obs = env.reset()
    state.update({"source": "live", "steps": [_normalize_live_step(obs, 1)], "final": {}})
    return _render(state, status="Environment reset.")


def _send_action(action_text: str, state: Dict[str, Any]):
    state = dict(state or _default_state())
    state, env = _ensure_session(state)

    try:
        payload = json.loads(action_text)
        action = ExecutionDeskAction(**payload)
    except Exception as exc:
        return _render(state, status=f"Invalid action JSON: {exc}")

    try:
        obs = env.step(action)
        next_step = len(state["steps"]) + 1
        state["source"] = "live"
        state["steps"] = [*state["steps"], _normalize_live_step(obs, next_step, payload)]
        if obs.done:
            rewards = [float(step.get("reward", 0.0) or 0.0) for step in state["steps"]]
            state["final"] = {
                "success": obs.observation.get("task_stage") == "done",
                "score": round(sum(rewards), 4),
                "steps": len(state["steps"]),
                "rewards": rewards,
            }
        return _render(state, status="Action submitted.")
    except Exception as exc:
        return _render(state, status=f"Step failed: {exc}")


def _history_rows(steps: list[Dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for step in steps:
        action_value = step.get("action_str") or json.dumps(step.get("action", {}))
        rows.append(
            [
                step.get("step", "-"),
                action_value,
                step.get("reward", 0.0),
                str(step.get("done", False)),
                step.get("observation", {}).get("task_stage", "-"),
            ]
        )
    return rows


def _render(state: Dict[str, Any], status: str | None = None):
    steps = state.get("steps", [])
    latest = steps[-1] if steps else {}
    obs = latest.get("observation", {})
    info = latest.get("info", {})
    final = state.get("final", {})
    meta = state.get("meta", {})

    meta_md = (
        f"**Source:** `{state.get('source', '-')}`  "
        f"**Task:** `{meta.get('task_name', '-')}`  "
        f"**Benchmark:** `{meta.get('benchmark', '-')}`  "
        f"**Model:** `{meta.get('model', '-')}`  "
        f"**Steps:** `{len(steps)}`"
    )

    obs_md = "No episode loaded."
    if latest:
        obs_md = "### Observation\n```json\n" + json.dumps(obs, indent=2) + "\n```"

    info_md = "### Info\n```json\n" + json.dumps(info, indent=2) + "\n```"
    action_md = "### Last Action\n```json\n" + json.dumps(latest.get("action", {}), indent=2) + "\n```"
    episode_md = (
        "### Episode\n```json\n"
        + json.dumps(
            {
                "reward": latest.get("reward", 0.0),
                "done": latest.get("done", False),
                "metadata": latest.get("metadata", {}),
                "final": final,
            },
            indent=2,
        )
        + "\n```"
    )
    status_md = status or "Ready."

    return (
        state,
        meta_md,
        obs_md,
        info_md,
        action_md,
        episode_md,
        _history_rows(steps),
        status_md,
    )


def build_gradio_ui() -> gr.Blocks:
    with gr.Blocks(title="Execution Desk UI") as demo:
        app_state = gr.State(_default_state())

        gr.Markdown(
            """
            # Execution Desk UI
            Use the mounted Gradio frontend to inspect logs or interact with the same
            execution-desk simulator process served by FastAPI.
            """
        )

        with gr.Tab("Live Session"):
            with gr.Row():
                reset_btn = gr.Button("Reset Environment", variant="primary")
                action_in = gr.Textbox(
                    label="Action JSON",
                    lines=4,
                    value='{"action_type":"CALL_TOOL","tool_name":"market_status_check"}',
                )
                send_btn = gr.Button("Send Action")

        with gr.Tab("Load Episode File"):
            file_in = gr.File(label="Episode JSON / JSONL", type="filepath")

        meta_out = gr.Markdown()
        status_out = gr.Markdown("Ready.")

        with gr.Row():
            obs_out = gr.Markdown()
            info_out = gr.Markdown()

        with gr.Row():
            action_out = gr.Markdown()
            episode_out = gr.Markdown()

        history_out = gr.Dataframe(
            headers=["Step", "Action", "Reward", "Done", "Stage"],
            datatype=["number", "str", "number", "str", "str"],
            interactive=False,
        )

        outputs = [
            app_state,
            meta_out,
            obs_out,
            info_out,
            action_out,
            episode_out,
            history_out,
            status_out,
        ]

        demo.load(fn=lambda: _render(_default_state()), outputs=outputs)
        reset_btn.click(fn=_reset_live, inputs=app_state, outputs=outputs)
        send_btn.click(fn=_send_action, inputs=[action_in, app_state], outputs=outputs)
        file_in.change(fn=_load_episode, inputs=[file_in, app_state], outputs=outputs)

    return demo


base_app = create_app(
    EnvAdapter,
    ExecutionDeskAction,
    ExecutionDeskObservation,
    env_name="execution_desk_assistant",
    max_concurrent_envs=4,
)


@base_app.get("/")
async def root():
    return RedirectResponse(url="/ui/")


demo = build_gradio_ui()
app = gr.mount_gradio_app(base_app, demo, path="/ui")


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000:
        main()
    else:
        main(port=args.port)
