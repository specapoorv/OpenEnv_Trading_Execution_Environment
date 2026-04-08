def execute(simulator, state, params=None):
    return simulator.simulate_system_tool(state, "compliance_recheck", params or {})
