from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import httpx
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]


def check_openenv_spec() -> None:
    spec_path = PROJECT_ROOT / "openenv.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    required_keys = {"spec_version", "name", "type", "runtime", "app", "port"}
    if not required_keys.issubset(spec):
        raise RuntimeError("openenv.yaml is missing required keys")
    if spec["app"] != "server.app:app":
        raise RuntimeError("openenv.yaml app must be server.app:app")
    if spec["port"] != 7860:
        raise RuntimeError("openenv.yaml port must be 7860")


def _check_http_contract(request: Callable[..., httpx.Response]) -> None:
    health_response = request("GET", "/health")
    health_response.raise_for_status()
    health_payload = health_response.json()
    if health_payload.get("status") not in {"ok", "healthy"}:
        raise RuntimeError(f"Unexpected health response: {health_payload}")

    state_response = request("GET", "/state")
    state_response.raise_for_status()
    state_payload = state_response.json()
    if "episode_id" not in state_payload or "step_count" not in state_payload:
        raise RuntimeError(f"Unexpected state payload: {state_payload}")

    reset_response = request("POST", "/reset")
    reset_response.raise_for_status()
    reset_payload = reset_response.json()
    if "observation" not in reset_payload:
        raise RuntimeError("Reset response missing observation")

    step_response = request(
        "POST",
        "/step",
        json={"action": {"action_type": "CALL_TOOL", "tool_name": "market_status_check"}},
    )
    step_response.raise_for_status()
    step_payload = step_response.json()
    if "observation" not in step_payload or "reward" not in step_payload:
        raise RuntimeError(f"Unexpected step payload: {step_payload}")


def check_app_import() -> None:
    from server.app import app

    paths = {route.path for route in app.routes}
    for required_path in ["/health", "/reset", "/step", "/state", "/ui"]:
        if required_path not in paths:
            raise RuntimeError(f"Missing route in app: {required_path}")


def check_http_url(base_url: str) -> None:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=20.0) as client:
        _check_http_contract(client.request)


def check_named_url(env_var: str) -> bool:
    base_url = os.getenv(env_var)
    if not base_url:
        return False
    check_http_url(base_url)
    return True


def check_inference() -> bool:
    if not os.getenv("HF_TOKEN") and not os.getenv("API_KEY"):
        return False

    result = subprocess.run(
        [sys.executable, str(ROOT / "inference.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
        check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("[START]") or not lines[-1].startswith("[END]"):
        raise RuntimeError("Inference logs do not follow [START]/[STEP]/[END] framing")
    if len([line for line in lines if line.startswith("[STEP]")]) == 0:
        raise RuntimeError("Inference did not emit any [STEP] lines")
    return True


def check_graders() -> None:
    from server.core.graders.task_graders import run_all_graders

    scores = run_all_graders(seed=7)
    repeated_scores = run_all_graders(seed=7)
    if len(scores) < 3:
        raise RuntimeError("Expected at least 3 graders")
    if scores != repeated_scores:
        raise RuntimeError("Graders are not deterministic for the same seed")
    for task_name, score in scores.items():
        if not (0.0 <= score <= 1.0):
            raise RuntimeError(f"Out of range grader score for {task_name}: {score}")
    seed_sweep = [run_all_graders(seed=seed) for seed in [1, 2, 3, 7]]
    if len({json.dumps(item, sort_keys=True) for item in seed_sweep}) == 1:
        raise RuntimeError("Graders appear degenerate across seeds")


def check_docker_build() -> bool:
    if os.getenv("RUN_DOCKER_CHECK") != "1":
        return False
    subprocess.run(
        ["docker", "build", "-t", "execution-desk-check", str(PROJECT_ROOT)],
        cwd=PROJECT_ROOT,
        check=True,
        timeout=1200,
    )
    return True


def main() -> None:
    checks: list[str] = []
    skipped: list[str] = []

    check_openenv_spec()
    checks.append("openenv_spec")

    check_app_import()
    checks.append("app_import")

    if check_named_url("LOCAL_BASE_URL"):
        checks.append("local_http")
    else:
        skipped.append("local_http")

    if check_named_url("SPACE_URL"):
        checks.append("space_http")
    else:
        skipped.append("space_http")

    if check_inference():
        checks.append("inference")
    else:
        skipped.append("inference")

    check_graders()
    checks.append("graders")

    if check_docker_build():
        checks.append("docker_build")
    else:
        skipped.append("docker_build")

    print(json.dumps({"status": "ok", "checks": checks, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
