from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional, Tuple

from server.core.env.action_space import OpenEnvEnv, build_action_space, build_observation_space
from server.core.env.base_state import Order, ScenarioState
from server.core.env.episode_manager import check_terminal_conditions, normalize_action
from server.core.env.observation_builder import build_info, build_observation
from server.core.env.reward_manager import RewardManager
from server.core.tasks.task1_data_verification import evaluate_data_readiness
from server.core.tasks.task2_system_monitoring import evaluate_system_readiness, system_unresolved_issues
from server.core.tasks.task3_execution_assistance import evaluate_execution_complete
from server.core.graders.task_graders import grade_data_validation, grade_system_readiness, grade_execution
from server.core.tools import TOOL_REGISTRY
from server.core.utils.constants import (
    ActionType,
    BROKERS,
    DATA_TOOLS,
    Stage,
    SYSTEM_TOOLS,
    URGENCY_LEVELS,
)
from server.core.utils.randomizers import sample_data_anomaly


class ToolSimulator:
    def __init__(self, rng: random.Random, recency_limit_minutes: int = 8) -> None:
        self.rng = rng
        self.recency_limit_minutes = recency_limit_minutes

    def initialize_scenario(self, max_steps: int, stage: str) -> ScenarioState:
        mid_price = round(self.rng.uniform(95.0, 105.0), 2)
        base_position = self.rng.randint(-200, 200)
        target_delta = self.rng.choice([-1, 1]) * self.rng.randint(120, 320)
        target_position = base_position + target_delta

        if stage == "DATA":
            data_anomalies = {
                "bloomberg_pull": sample_data_anomaly(self.rng, True),
                "oms_position_check": sample_data_anomaly(self.rng, True),
                "risk_system_check": False,
                "compliance_verify": False,
                "internal_report_fetch": sample_data_anomaly(self.rng, True),
                "market_status_check": False,
            }
            system_truth = {
                "oms_connected": True,
                "strategy_status": "running",
                "strategy_recoverable": True,
                "compliance_ok": True,
                "oms_recoverable": True,
            }

        elif stage == "SYSTEM":
            data_anomalies={}
            system_truth=self._system_truth()

        elif stage == "EXECUTION":
            data_anomalies= {}
            system_truth = {
                "oms_connected": True,
                "strategy_status": "running",
                "strategy_recoverable": True,
                "compliance_ok": True,
                "oms_recoverable": True,
            }

        state = ScenarioState(
            stage = stage,
            max_steps=max_steps,
            current_broker=self.rng.choice(BROKERS),
            data_truth={
                "mid_price": mid_price,
                "volume": self.rng.randint(150_000, 600_000),
                "market_open": self.rng.random() > 0.1,
                "position": base_position,
                "risk_limit": self.rng.randint(500, 800),
                "restricted": False,
                "timestamp": 0,
            },
            data_anomalies=data_anomalies,
            system_truth=system_truth,
            execution_truth={
                "current_position": base_position,
                "target_position": target_position,
                "tolerance": self.rng.randint(4, 12),
                "base_liquidity": self.rng.randint(40, 130),
                "market_mid": mid_price,
                "recent_slippage_bps": 0.0,
                "fills": [],
                # Hard Task 3 constraints (randomised per seed)
                "slippage_budget_bps": round(self.rng.uniform(15.0, 35.0), 2),
                "exec_step_budget": self.rng.randint(20, 30),
                "exec_stage_start_step": None,   # set when EXECUTION stage begins
            },
        )
        if not state.system_truth["oms_connected"]:
            state.issue_log.append("OMS connection is down.")
        if state.system_truth["strategy_status"] != "running":
            state.issue_log.append(f"Strategy status is {state.system_truth['strategy_status']}.")
        if not state.system_truth["compliance_ok"]:
            state.issue_log.append("Compliance check failed.")
        return state

    def _system_truth(self) -> Dict[str, Any]:
        oms_connected = self.rng.random() > 0.18
        strategy_status = self.rng.choices(["running", "paused", "crashed"], weights=[0.72, 0.18, 0.10], k=1)[0]
        compliance_ok = self.rng.random() > 0.16
        return {
            "oms_connected": oms_connected,
            "strategy_status": strategy_status,
            "strategy_recoverable": strategy_status in {"paused", "crashed"} and self.rng.random() > 0.35,
            "compliance_ok": compliance_ok,
            "oms_recoverable": oms_connected or self.rng.random() > 0.55,
        }

    def call_tool(self, state: ScenarioState, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        state.tool_call_counts[tool_name] = state.tool_call_counts.get(tool_name, 0) + 1
        if tool_name not in TOOL_REGISTRY:
            output = {"ok": False, "error": f"unknown_tool:{tool_name}", "timestamp": state.now_minute}
        else:
            output = TOOL_REGISTRY[tool_name](self, state, params)
        state.tool_outputs[tool_name] = output
        if not output.get("ok", False):
            state.tool_failures[tool_name] = state.tool_failures.get(tool_name, 0) + 1
        return output

    def simulate_data_tool(self, state: ScenarioState, tool_name: str) -> Dict[str, Any]:
        base = state.data_anomalies.get(tool_name, {})
        call_count = state.tool_call_counts.get(tool_name, 1)
        decay = max(0.15, 1.0 - 0.25 * (call_count - 1))
        anomaly = {
            "tool_failure": base.get("tool_failure", False) and self.rng.random() < 0.60 * decay,
            "missing_field": base.get("missing_field", False) and self.rng.random() < 0.55 * decay,
            "stale": base.get("stale", False) and self.rng.random() < 0.65 * decay,
            "inconsistent": base.get("inconsistent", False) and self.rng.random() < 0.70 * decay,
        }
        if anomaly["tool_failure"]:
            return {"ok": False, "error": "transient_tool_failure", "timestamp": state.now_minute}

        truth = state.data_truth
        payload: Dict[str, Any] = {"ok": True, "timestamp": state.now_minute}
        if tool_name == "bloomberg_pull":
            payload.update({"mid_price": truth["mid_price"], "volume": truth["volume"]})
            if anomaly["inconsistent"]:
                payload["mid_price"] = round(truth["mid_price"] + self.rng.uniform(-1.6, 1.6), 2)
        elif tool_name == "oms_position_check":
            payload.update({"position": state.execution_truth["current_position"], "source": "oms"})
            if anomaly["inconsistent"]:
                payload["position"] = state.execution_truth["current_position"] + self.rng.randint(-18, 18)
        elif tool_name == "risk_system_check":
            payload.update({"risk_limit": truth["risk_limit"], "exposure": abs(state.execution_truth["current_position"])})
        elif tool_name == "compliance_verify":
            payload.update({"restricted": truth["restricted"], "compliance_ok": state.system_truth["compliance_ok"]})
        elif tool_name == "internal_report_fetch":
            payload.update(
                {
                    "mid_price": truth["mid_price"],
                    "position": state.execution_truth["current_position"],
                    "report_id": f"rep-{state.now_minute:04d}",
                }
            )
            if anomaly["inconsistent"]:
                if self.rng.random() < 0.5:
                    payload["mid_price"] = round(truth["mid_price"] + self.rng.uniform(-2.0, 2.0), 2)
                else:
                    payload["position"] = state.execution_truth["current_position"] + self.rng.randint(-15, 15)
        elif tool_name == "market_status_check":
            payload.update({"market_open": truth["market_open"], "session": "regular"})

        if anomaly["missing_field"]:
            removable = [key for key in payload if key not in {"ok", "timestamp"}]
            if removable:
                payload.pop(self.rng.choice(removable), None)
        if anomaly["stale"]:
            payload["timestamp"] = max(0, state.now_minute - self.rng.randint(self.recency_limit_minutes + 1, 12))
        return payload

    def simulate_system_tool(self, state: ScenarioState, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        system = state.system_truth
        if tool_name == "ping_oms_connection":
            if not system["oms_connected"] and system["oms_recoverable"] and self.rng.random() < 0.20:
                system["oms_connected"] = True
                state.issue_log.append("OMS link recovered during ping.")
            return {"ok": True, "oms_connected": system["oms_connected"], "timestamp": state.now_minute}
        if tool_name == "strategy_health_check":
            return {
                "ok": True,
                "strategy_status": system["strategy_status"],
                "recoverable": system["strategy_recoverable"],
                "timestamp": state.now_minute,
            }
        if tool_name == "compliance_recheck":
            if not system["compliance_ok"] and self.rng.random() < 0.18:
                system["compliance_ok"] = True
                state.issue_log.append("Compliance cleared on recheck.")
            return {
                "ok": True,
                "compliance_ok": system["compliance_ok"],
                "restricted": not system["compliance_ok"],
                "timestamp": state.now_minute,
            }
        if tool_name == "restart_strategy":
            previous = system["strategy_status"]
            if previous == "running":
                return {"ok": True, "strategy_status": "running", "message": "already_running", "timestamp": state.now_minute}
            if system["strategy_recoverable"]:
                system["strategy_status"] = "running"
                return {"ok": True, "strategy_status": "running", "message": f"recovered_from_{previous}", "timestamp": state.now_minute}
            return {"ok": False, "strategy_status": previous, "error": "restart_failed_requires_escalation", "timestamp": state.now_minute}
        if tool_name == "escalate_issue":
            state.escalated = True
            state.escalation_reason = str(params.get("reason") or "unspecified_issue")
            return {"ok": True, "escalated": True, "reason": state.escalation_reason, "timestamp": state.now_minute}
        return {"ok": False, "error": f"unsupported_system_tool:{tool_name}", "timestamp": state.now_minute}

    def simulate_execution_tool(self, state: ScenarioState, tool_name: str) -> Dict[str, Any]:
        if tool_name == "get_current_position":
            return {
                "ok": True,
                "current_position": state.execution_truth["current_position"],
                "target_position": state.execution_truth["target_position"],
                "timestamp": state.now_minute,
            }
        return {"ok": False, "error": f"unsupported_execution_tool:{tool_name}", "timestamp": state.now_minute}

    def advance_market(self, state: ScenarioState) -> None:
        '''
        Advances Market with randomness out of our control
        '''

        state.step_count += 1
        state.now_minute += 1
        execution = state.execution_truth
        execution["market_mid"] = round(execution["market_mid"] + self.rng.uniform(-0.12, 0.12), 4)
        if self.rng.random() < 0.10:
            execution["current_position"] += self.rng.choice([-2, -1, 1, 2])
        execution["current_liquidity"] = max(20, execution["base_liquidity"] + self.rng.randint(-25, 35))
        for order in list(state.outstanding_orders.values()):
            if order.status != "working":
                continue
            fill_fraction = {"low": 0.20, "normal": 0.45, "high": 0.70}[order.urgency]
            broker_factor = {"broker_alpha": 1.0, "broker_beta": 0.85, "broker_delta": 1.15}[order.broker]
            max_fill = min(order.remaining_size, max(0, int(execution["current_liquidity"] * fill_fraction * broker_factor)))
            if max_fill <= 0 or self.rng.random() < 0.20:
                continue
            self._apply_fill(state, order, self.rng.randint(max(1, max_fill // 3), max_fill))

    def submit_order(self, state: ScenarioState, size: int, side: str, broker: str, urgency: str) -> Dict[str, Any]:
        size = int(abs(size))
        if size <= 0:
            return {"ok": False, "error": "invalid_size"}
        if broker not in BROKERS:
            return {"ok": False, "error": "unknown_broker"}
        if urgency not in URGENCY_LEVELS:
            return {"ok": False, "error": "unknown_urgency"}
        if side not in {"buy", "sell"}:
            return {"ok": False, "error": "unknown_side"}
        liquidity = state.execution_truth.get("current_liquidity", state.execution_truth["base_liquidity"])
        max_clip = 160 if broker == "broker_beta" else 220
        if size > max_clip:
            return {"ok": False, "error": "size_rejected", "max_size": max_clip}
        if broker == "broker_beta" and urgency == "high" and self.rng.random() < 0.35:
            return {"ok": False, "error": "broker_constraint_rejected"}
        if size > liquidity * 2 and self.rng.random() < 0.50:
            return {"ok": False, "error": "liquidity_rejected", "liquidity": liquidity}

        order = Order(
            order_id=state.next_order_id,
            side=side,
            requested_size=size,
            remaining_size=size,
            broker=broker,
            urgency=urgency,
            created_step=state.step_count,
        )
        state.next_order_id += 1
        state.outstanding_orders[order.order_id] = order
        immediate_fill = min(order.remaining_size, max(0, int(liquidity * {"low": 0.18, "normal": 0.35, "high": 0.60}[urgency])))
        if immediate_fill > 0 and self.rng.random() > 0.15:
            self._apply_fill(state, order, self.rng.randint(max(1, immediate_fill // 2), immediate_fill))
        return {"ok": True, "order": order.snapshot()}

    def split_order(self, state: ScenarioState, total_size: int, side: str, broker: str, urgency: str, max_clip: int) -> Dict[str, Any]:
        children = []
        remaining = int(abs(total_size))
        max_clip = max(1, int(abs(max_clip)))
        while remaining > 0:
            child_size = min(max_clip, remaining)
            result = self.submit_order(state, child_size, side, broker, urgency)
            children.append(result)
            remaining -= child_size
            if not result.get("ok", False):
                break
        return {"ok": all(child.get("ok", False) for child in children), "children": children}

    def cancel_order(self, state: ScenarioState, order_id: int) -> Dict[str, Any]:
        order = state.outstanding_orders.get(order_id)
        if order is None:
            return {"ok": False, "error": "unknown_order"}
        if order.status != "working":
            return {"ok": False, "error": f"order_not_cancellable:{order.status}"}
        order.status = "cancelled"
        return {"ok": True, "order": order.snapshot()}

    def change_broker(self, state: ScenarioState, broker: str) -> Dict[str, Any]:
        if broker not in BROKERS:
            return {"ok": False, "error": "unknown_broker"}
        state.current_broker = broker
        return {"ok": True, "broker": broker, "timestamp": state.now_minute}

    def _apply_fill(self, state: ScenarioState, order: Order, fill_size: int) -> None:
        fill_size = min(fill_size, order.remaining_size)
        if fill_size <= 0:
            return
        sign = 1 if order.side == "buy" else -1
        midpoint = state.execution_truth["market_mid"]
        urgency_penalty = {"low": 1.0, "normal": 3.0, "high": 6.5}[order.urgency]
        broker_penalty = {"broker_alpha": 1.5, "broker_beta": 2.0, "broker_delta": 0.8}[order.broker]
        slippage_bps = urgency_penalty + broker_penalty + self.rng.uniform(-0.7, 1.4)
        fill_price = midpoint * (1 + sign * slippage_bps / 10000.0)
        prior_value = order.average_fill_price * order.filled_size
        order.filled_size += fill_size
        order.remaining_size -= fill_size
        order.average_fill_price = (prior_value + fill_price * fill_size) / max(order.filled_size, 1)
        if order.remaining_size == 0:
            order.status = "filled"
        state.execution_truth["current_position"] += sign * fill_size
        state.execution_truth["recent_slippage_bps"] = round(abs(slippage_bps), 4)
        # Check risk limit breach
        risk_limit = state.data_truth.get("risk_limit", float("inf"))
        if abs(state.execution_truth["current_position"]) > risk_limit:
            state.risk_limit_breached = True
        state.execution_truth["fills"].append(
            {
                "order_id": order.order_id,
                "fill_size": fill_size,
                "fill_price": round(fill_price, 4),
                "slippage_bps": round(slippage_bps, 4),
                "timestamp": state.now_minute,
            }
        )


class ExecutionDeskEnv(OpenEnvEnv):
    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None, max_steps: int = 60) -> None:
        super().__init__()
        self.max_steps = max_steps
        self._seed = seed
        self.rng = random.Random(seed)
        self.tool_sim = ToolSimulator(self.rng)
        self.reward_manager = RewardManager()
        self.scenario = self.tool_sim.initialize_scenario(max_steps=max_steps, stage="DATA")
        self.action_space = build_action_space()
        self.observation_space = build_observation_space(max_steps)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self._seed = seed
        options = options or {}
        
        # Map task_id (from inference.py) to internal stage names
        task_id = options.get("task_id", "easy")
        stage_map = {"easy": "DATA", "medium": "SYSTEM", "hard": "EXECUTION"}
        target_stage = stage_map.get(task_id, "DATA")

        self.rng = random.Random(self._seed)
        self.tool_sim = ToolSimulator(self.rng)
        self.reward_manager = RewardManager()
        
        # Pass the target_stage here!
        self.scenario = self.tool_sim.initialize_scenario(
            max_steps=options.get("max_steps", self.max_steps),
            stage=target_stage 
        )
        return build_observation(self.scenario), build_info(self.scenario, self.tool_sim.recency_limit_minutes)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        prev_state = copy.deepcopy(self.scenario)
        event: Dict[str, Any] = {}
        action = normalize_action(action, self.scenario)
        self._apply_action(action, event)
        if self.scenario.stage != Stage.DONE:
            self.tool_sim.advance_market(self.scenario)

        terminated, truncated, terminal_event = check_terminal_conditions(self.scenario, self.tool_sim.recency_limit_minutes)
        event.update(terminal_event)

        # THE SAFETY NET GRADER HERE, grades if timeout happened 
        if terminated or truncated:
            if not hasattr(self.scenario, 'grades'):
                self.scenario.grades = {}
            
            # Fill in any missing scores (for tasks the agent didn't finish)
            if "task1_data" not in self.scenario.grades:
                self.scenario.grades["task1_data"] = grade_data_validation(self.scenario)
            
            if "task2_system" not in self.scenario.grades:
                self.scenario.grades["task2_system"] = grade_system_readiness(self.scenario)
                
            if "task3_execution" not in self.scenario.grades:
                self.scenario.grades["task3_execution"] = grade_execution(self.scenario)


        reward = self.reward_manager.compute(prev_state, self.scenario, event, terminated, truncated)
        return build_observation(self.scenario), reward, terminated, truncated, build_info(self.scenario, self.tool_sim.recency_limit_minutes, event)

    def state(self) -> Dict[str, Any]:
        return {
            "observation": build_observation(self.scenario),
            "info": build_info(self.scenario, self.tool_sim.recency_limit_minutes),
        }

    def close(self) -> None:
        return None

    def _apply_action(self, action: Dict[str, Any], event: Dict[str, Any]) -> None:
        action_type = action["action_type"]
        if action_type == ActionType.CALL_TOOL:
            tool_name = action.get("tool_name")
            if not tool_name:
                event["invalid_action"] = True
                return
            prior = self.scenario.tool_outputs.get(tool_name)
            result = self.tool_sim.call_tool(self.scenario, tool_name, action.get("params"))
            event["last_tool_result"] = result
            event["tool_failure"] = not result.get("ok", False)
            event["useful_tool"] = prior is None or prior != result
            event["redundant_tool"] = prior is not None and prior == result
            if self.scenario.stage == Stage.DATA_VALIDATION:
                event["found_inconsistency"] = not evaluate_data_readiness(self.scenario, self.tool_sim.recency_limit_minutes)["consistent"]
            return
        if action_type == ActionType.DECLARE:
            self._handle_declare(action.get("declare_flag"), event)
            return
        if action_type == ActionType.RESTART_STRATEGY:
            result = self.tool_sim.call_tool(self.scenario, "restart_strategy")
            event["last_tool_result"] = result
            event["fixed_issue"] = result.get("ok", False) and result.get("strategy_status") == "running"
            event["invalid_action"] = not result.get("ok", False)
            return
        if action_type == ActionType.ESCALATE:
            if self.scenario.stage != Stage.SYSTEM_HEALTH:
                event["bad_escalation"] = True
                event["invalid_action"] = True
                return
            unresolved = system_unresolved_issues(self.scenario)
            self.tool_sim.call_tool(self.scenario, "escalate_issue", {"reason": ",".join(unresolved) or "manual_escalation"})
            event["correct_escalation"] = any(issue in unresolved for issue in ["oms_disconnected", "strategy_unrecoverable", "compliance_violation"])
            event["bad_escalation"] = not event["correct_escalation"]
            return
        if action_type == ActionType.SUBMIT_ORDER:
            result = TOOL_REGISTRY["submit_order"](self.tool_sim, self.scenario, action)
            event["last_order_result"] = result
            event["order_rejected"] = not result.get("ok", False)
            event["executed_fill_size"] = self._latest_fill_size()
            return
        if action_type == ActionType.SPLIT_ORDER:
            result = TOOL_REGISTRY["split_order"](self.tool_sim, self.scenario, action)
            event["last_order_result"] = result
            event["order_rejected"] = not result.get("ok", False)
            event["executed_fill_size"] = self._latest_fill_size()
            return
        if action_type == ActionType.CANCEL_ORDER:
            result = TOOL_REGISTRY["cancel_order"](self.tool_sim, self.scenario, action)
            event["last_order_result"] = result
            event["cancelled_working_order"] = result.get("ok", False)
            event["invalid_action"] = not result.get("ok", False)
            return
        if action_type == ActionType.CHANGE_BROKER:
            result = TOOL_REGISTRY["change_broker"](self.tool_sim, self.scenario, action)
            event["last_tool_result"] = result
            event["useful_tool"] = result.get("ok", False)
            event["invalid_action"] = not result.get("ok", False)
            return
        event["invalid_action"] = True

    def _handle_declare(self, declare_flag: Optional[str], event: Dict[str, Any]) -> None:
        if not hasattr(self.scenario, 'grades'):
            self.scenario.grades = {}
        
        if declare_flag == "data_ready":
            validation = evaluate_data_readiness(self.scenario, self.tool_sim.recency_limit_minutes)
            if self.scenario.stage == Stage.DATA_VALIDATION and validation["ready"]:
                #TASK 1 Completed, Grader can grade now
                self.scenario.completed_flags["data_ready"] = True
                self.scenario.grades["task1_data"] = grade_data_validation(self.scenario)
                self.scenario.stage = Stage.SYSTEM_HEALTH
                event["stage_advanced"] = True
            else:
                event["premature_declare"] = True
                event["unresolved_issues"] = len(validation["issues"])
            return
        if declare_flag == "systems_ready":
            readiness = evaluate_system_readiness(self.scenario)
            if self.scenario.stage == Stage.SYSTEM_HEALTH and readiness["ready"]:
                self.scenario.completed_flags["systems_ready"] = True
                self.scenario.grades["task2_system"] = grade_system_readiness(self.scenario)
                self.scenario.stage = Stage.EXECUTION
                # Record when execution stage started for the step-budget grader
                self.scenario.execution_truth["exec_stage_start_step"] = self.scenario.step_count
                event["stage_advanced"] = True
            else:
                event["premature_declare"] = True
                event["unresolved_issues"] = len(readiness["issues"])
            return
        if declare_flag == "execution_complete":
            complete = evaluate_execution_complete(self.scenario)
            if self.scenario.stage == Stage.EXECUTION and complete["ready"]:
                self.scenario.completed_flags["execution_complete"] = True
                self.scenario.grades["task3_execution"] = grade_execution(self.scenario)
                self.scenario.stage = Stage.DONE
                event["stage_advanced"] = True
                event["success"] = True
            else:
                event["premature_declare"] = True
                event["unresolved_issues"] = 1
            return
        event["invalid_action"] = True

    def _latest_fill_size(self) -> int:
        fills = self.scenario.execution_truth["fills"]
        return fills[-1]["fill_size"] if fills else 0


def heuristic_policy(observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    stage = observation["task_stage"]
    known = observation["known_data"]
    if stage == Stage.DATA_VALIDATION.value:
        validation = info["data_validation"]
        if validation["ready"]:
            return {"action_type": ActionType.DECLARE, "declare_flag": "data_ready"}
        for tool_name in DATA_TOOLS:
            tool_output = known.get(tool_name)
            if tool_output is None or not tool_output.get("ok", False):
                return {"action_type": ActionType.CALL_TOOL, "tool_name": tool_name}
            if observation["timestamps"]["sim_minute"] - tool_output.get("timestamp", -999) > 5:
                return {"action_type": ActionType.CALL_TOOL, "tool_name": tool_name}
        if "price_mismatch" in validation["issues"]:
            return {"action_type": ActionType.CALL_TOOL, "tool_name": "bloomberg_pull"}
        if "position_mismatch" in validation["issues"]:
            return {"action_type": ActionType.CALL_TOOL, "tool_name": "oms_position_check"}
        missing = next((item.split(":")[1] for item in validation["issues"] if item.startswith("missing_tool:")), None)
        return {"action_type": ActionType.CALL_TOOL, "tool_name": missing or "internal_report_fetch"}
    if stage == Stage.SYSTEM_HEALTH.value:
        readiness = info["system_readiness"]
        if readiness["ready"]:
            return {"action_type": ActionType.DECLARE, "declare_flag": "systems_ready"}
        for tool_name in ["ping_oms_connection", "strategy_health_check", "compliance_recheck"]:
            if tool_name not in known:
                return {"action_type": ActionType.CALL_TOOL, "tool_name": tool_name}
        if "strategy_not_running" in readiness["issues"] and known.get("strategy_health_check", {}).get("recoverable", False):
            return {"action_type": ActionType.RESTART_STRATEGY}
        if any(issue in readiness["issues"] for issue in ["oms_disconnected", "compliance_violation", "strategy_unrecoverable"]):
            if "oms_disconnected" in readiness["issues"]:
                return {"action_type": ActionType.CALL_TOOL, "tool_name": "ping_oms_connection"}
            if "compliance_violation" in readiness["issues"]:
                return {"action_type": ActionType.CALL_TOOL, "tool_name": "compliance_recheck"}
            return {"action_type": ActionType.ESCALATE}
        return {"action_type": ActionType.CALL_TOOL, "tool_name": "strategy_health_check"}
    if stage == Stage.EXECUTION.value:
        position = observation["position_state"]["current_position"]
        target = observation["position_state"]["target_position"]
        delta = target - position
        working_orders = [order for order in observation["order_state"]["outstanding_orders"] if order["status"] == "working"]
        if abs(delta) <= observation["position_state"]["tolerance"] and not working_orders:
            return {"action_type": ActionType.DECLARE, "declare_flag": "execution_complete"}
        if len(working_orders) > 4:
            return {"action_type": ActionType.CANCEL_ORDER, "order_id": working_orders[0]["order_id"]}
        side = "buy" if delta > 0 else "sell"
        size = abs(delta)
        if size > 170:
            return {"action_type": ActionType.SPLIT_ORDER, "size": min(size, 240), "side": side, "broker": "broker_delta", "urgency": "normal", "max_clip": 110}
        return {"action_type": ActionType.SUBMIT_ORDER, "size": min(size, 140), "side": side, "broker": "broker_delta" if size > 90 else "broker_alpha", "urgency": "high" if size > 50 else "normal"}
    return {"action_type": ActionType.DECLARE, "declare_flag": "execution_complete"}


def run_demo(seed: int = 7, max_steps: int = 60, policy=heuristic_policy) -> None:
    env = ExecutionDeskEnv(seed=seed, max_steps=max_steps)
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    print(f"Starting demo episode with seed={seed}")
    while not (terminated or truncated):
        action = policy(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"step={observation['timestamps']['step_count']:02d} "
            f"stage={observation['task_stage']:<16} "
            f"action={action['action_type']} "
            f"reward={reward:>6.2f} "
            f"tracking_error={observation['position_state']['tracking_error']}"
        )
    print(f"episode_done terminated={terminated} truncated={truncated} total_reward={total_reward:.2f}")
    print(f"completed_flags={info['completed_flags']}")
    print(f"execution_status={info['execution_status']}")
