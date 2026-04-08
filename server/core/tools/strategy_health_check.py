def execute(simulator, state, params=None):
    return simulator.simulate_system_tool(state, "strategy_health_check", params or {})
