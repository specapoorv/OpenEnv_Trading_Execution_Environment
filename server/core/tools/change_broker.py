def execute(simulator, state, params=None):
    params = params or {}
    return simulator.change_broker(state, params.get("broker", state.current_broker))
