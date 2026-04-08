def execute(simulator, state, params=None):
    return simulator.simulate_system_tool(state, "restart_strategy", params or {})
