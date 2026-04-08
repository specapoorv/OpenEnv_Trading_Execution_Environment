def execute(simulator, state, params=None):
    params = params or {}
    return simulator.submit_order(
        state,
        size=params.get("size", 0),
        side=params.get("side", "buy"),
        broker=params.get("broker", state.current_broker),
        urgency=params.get("urgency", "normal"),
    )
