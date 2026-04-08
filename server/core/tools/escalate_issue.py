def execute(simulator, state, params=None):
    return simulator.simulate_system_tool(state, "escalate_issue", params or {})
