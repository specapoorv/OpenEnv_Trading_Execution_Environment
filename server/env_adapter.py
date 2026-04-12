# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin adapter to help in readability, directly exposes step, reset, state"""

from __future__ import annotations
from typing import Any, Dict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ExecutionDeskAction, ExecutionDeskObservation
except ImportError:
    from models import ExecutionDeskAction, ExecutionDeskObservation

from server.core.env.execution_desk_env import ExecutionDeskEnv


class EnvAdapter(Environment):
    """Thin adapter for ExecutionDeskEnv to match OpenEnv Environment interface."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: int = 7, max_steps: int = 60):
        self._env = ExecutionDeskEnv(seed=seed, max_steps=max_steps)
        print("[ENV INSTANCE ID]", id(self._env))
        self._seed = seed
        self._state = State(episode_id=str(uuid4()), step_count=0)
    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> ExecutionDeskObservation:
        """Reset environment and return initial observation."""
        # Start with the options dict if provided
        options = dict(kwargs.get("options", {}))
        
        # 1. Capture max_steps if passed as a top-level argument
        if "max_steps" in kwargs:
            options["max_steps"] = kwargs["max_steps"]
            
        # 2. Capture task_id if passed as a top-level argument (from inference.py)
        if "task_id" in kwargs:
            options["task_id"] = kwargs["task_id"]

        # 3. Pass the consolidated options to the actual environment
        observation, info = self._env.reset(seed=seed or self._seed, options=options or None)
        
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        return ExecutionDeskObservation(
            observation=observation,
            info=info,
            done=False,
            reward=0.0,
            metadata={"stage": observation.get("task_stage")},
        )
    def step(self, action: ExecutionDeskAction) -> ExecutionDeskObservation:
        """Take an action and return the observation."""
        action_payload = action.model_dump(exclude_none=True)
        obs, reward, terminated, truncated, info = self._env.step(action_payload)
        self._state.step_count += 1
        done = terminated or truncated

        return ExecutionDeskObservation(
            observation=obs,
            info=info,
            done=done,
            reward=float(reward),
            metadata={"terminated": terminated, "truncated": truncated, "stage": obs.get("task_stage")},
        )

    @property
    def state(self) -> State:
        return self._state