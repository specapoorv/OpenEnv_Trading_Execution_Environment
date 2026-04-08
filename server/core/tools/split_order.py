def execute(simulator, state, params=None):
    params = params or {}
    return simulator.split_order(
        state,
        total_size=params.get("size", 0),
        side=params.get("side", "buy"),
        broker=params.get("broker", state.current_broker),
        urgency=params.get("urgency", "normal"),
        max_clip=params.get("max_clip", 50),
    )
