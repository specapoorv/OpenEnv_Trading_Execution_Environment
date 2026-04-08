def execute(simulator, state, params=None):
    return simulator.simulate_system_tool(state, "ping_oms_connection", params or {})
