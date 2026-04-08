# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the execution desk environment."""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ExecutionDeskAction(Action):
    """Action schema used by the execution desk simulator."""

    action_type: str = Field(..., description="Action type enum value")
    tool_name: Optional[str] = Field(default=None, description="Tool name for CALL_TOOL")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Optional tool parameters")
    declare_flag: Optional[str] = Field(default=None, description="Declare flag for DECLARE")
    size: Optional[int] = Field(default=None, description="Order size for order actions")
    side: Optional[str] = Field(default=None, description="Order side: buy or sell")
    broker: Optional[str] = Field(default=None, description="Broker choice")
    urgency: Optional[str] = Field(default=None, description="Order urgency")
    order_id: Optional[int] = Field(default=None, description="Order id for cancel action")
    max_clip: Optional[int] = Field(default=None, description="Max child clip for split orders")


class ExecutionDeskObservation(Observation):
    """Observation payload for the execution desk simulator."""

    observation: Dict[str, Any] = Field(default_factory=dict, description="Environment observation payload")
    info: Dict[str, Any] = Field(default_factory=dict, description="Environment info payload")


# Backward-compatible aliases with the scaffold names.

# [Please see]
# We can safely remove this? and change in init as well
TradingAction = ExecutionDeskAction
TradingObservation = ExecutionDeskObservation
