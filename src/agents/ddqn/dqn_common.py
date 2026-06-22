ACTIONS = [
    None,
    {"action": "move", "direction": "WEST"},
    {"action": "move", "direction": "EAST"},
]


NUM_BRICKS = 16
STATE_DIM = 6

def encode_state(state, prev_state=None):
    w = state["width"]
    h = state["height"]

    paddle_cx = (state["paddle_x"] + state["paddle_width"] / 2.0) / w
    ball_x = state["ball_x"] / w
    ball_y = state["ball_y"] / h

    if prev_state is not None:
        dx = (state["ball_x"] - prev_state["ball_x"]) / w
        dy = (state["ball_y"] - prev_state["ball_y"]) / h
    else:
        dx = 0.0
        dy = 0.0

    rel_x = ball_x - paddle_cx

    return [paddle_cx, ball_x, ball_y, dx, dy, rel_x]
