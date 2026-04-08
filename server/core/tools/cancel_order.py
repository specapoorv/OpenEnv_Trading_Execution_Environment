def execute(simulator, state, params=None):
    params = params or {}
    return simulator.cancel_order(state, params.get("order_id", 0))
