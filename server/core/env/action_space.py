from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

from server.core.utils.constants import (
    ACTION_TYPES,
    ALL_TOOLS,
    BROKERS,
    DECLARE_FLAGS,
    STAGE_TO_ID,
    URGENCY_LEVELS,
)


try:
    from openenv import OpenEnvEnv  # type: ignore
except Exception:
    try:
        from gymnasium import Env as OpenEnvEnv  # type: ignore
    except Exception:
        try:
            from gym import Env as OpenEnvEnv  # type: ignore
        except Exception:
            class OpenEnvEnv:  # type: ignore
                metadata: Dict[str, Any] = {}


try:
    from gymnasium import spaces  # type: ignore
except Exception:
    try:
        from gym import spaces  # type: ignore
    except Exception:
        class _BaseSpace:
            def sample(self) -> Any:
                raise NotImplementedError

        class Discrete(_BaseSpace):
            def __init__(self, n: int) -> None:
                self.n = n

            def sample(self) -> int:
                return random.randrange(self.n)

        class Box(_BaseSpace):
            def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype=float) -> None:
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

            def sample(self) -> List[float]:
                return [random.uniform(self.low, self.high) for _ in range(math.prod(self.shape))]

        class DictSpace(_BaseSpace):
            def __init__(self, mapping: Dict[str, _BaseSpace]) -> None:
                self.mapping = mapping

            def sample(self) -> Dict[str, Any]:
                return {key: space.sample() for key, space in self.mapping.items()}

        class spaces:  # type: ignore
            Discrete = Discrete
            Box = Box
            Dict = DictSpace


def build_action_space() -> Any:
    """
    Defines the action space for the RL agent.

    The action space specifies all possible actions the agent can take in the environment:
      - action_type: Type of action (Discrete)
      - tool_name: Tool to use for the action (Discrete)
      - declare_flag: Optional workflow flags (Discrete)
      - size: Order size if placing an order (Box/float)
      - side: Buy or sell (Discrete)
      - broker: Broker to route order to (Discrete)
      - urgency: Urgency level of the order (Discrete)
      - order_id: Identifier for the order (Discrete)
      - max_clip: Maximum clip size (Box/float)

    Notes:
    - Each element is constrained by the ranges defined here.
    - The agent outputs a **single action dict** per step matching this space.
    - Early workflow stages may ignore order-related fields until verification/preparation is complete.
    """

    return spaces.Dict(
        {
            "action_type": spaces.Discrete(len(ACTION_TYPES)),
            "tool_name": spaces.Discrete(len(ALL_TOOLS) + 1),
            "declare_flag": spaces.Discrete(len(DECLARE_FLAGS) + 1),
            "size": spaces.Box(low=0, high=400, shape=(1,), dtype=float),
            "side": spaces.Discrete(2),
            "broker": spaces.Discrete(len(BROKERS)),
            "urgency": spaces.Discrete(len(URGENCY_LEVELS)),
            "order_id": spaces.Discrete(128),
            "max_clip": spaces.Box(low=1, high=200, shape=(1,), dtype=float),
        }
    )


def build_observation_space(max_steps: int) -> Any:
    """
    Defines the observation space for the RL agent.

    The observation space is the **simplified view of the environment state** that the agent sees:
      - task_stage: Current stage of the workflow
      - step_count: Current step in the episode
      - tracking_error: Current tracking error (float)
      - current_position: Current position in the market
      - target_position: Target position to reach
      - open_orders: Number of open/outstanding orders

    Notes:
    - This does NOT expose full internal state (orders, logs, system truths) to the agent.
    - Early stages may have 'next order = NA', indicating the agent should perform verification/preparation.
    - As the workflow progresses, observations reflect actionable info for order placement.
    """

    return spaces.Dict(
        {
            "task_stage": spaces.Discrete(len(STAGE_TO_ID)),
            "step_count": spaces.Discrete(max_steps + 1),
            "tracking_error": spaces.Box(low=0, high=10_000, shape=(1,), dtype=float),
            "current_position": spaces.Box(low=-10_000, high=10_000, shape=(1,), dtype=float),
            "target_position": spaces.Box(low=-10_000, high=10_000, shape=(1,), dtype=float),
            "open_orders": spaces.Discrete(128),
        }
    )

