__author__ = "Mário Antunes"
__version__ = "1.0.0"
__status__ = "Development"

import asyncio
from typing import Any, Dict, Optional

import torch

from agents.base_agent import BaseAgent
from agents.DDQN.dqn_common import ACTIONS, STATE_DIM, encode_state
from agents.DDQN.dqn_model import QNetwork

MODEL_PATH = "dqn_breakout.pt"


class DQNAgent(BaseAgent):
    def __init__(self, server_uri: str = "ws://localhost:8765/ws") -> None:
        super().__init__(server_uri)
        self.net = QNetwork(STATE_DIM, len(ACTIONS))
        self.net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.net.eval()
        self.prev_state: Optional[Dict[str, Any]] = None

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        state = self.current_state
        if not state or state.get("game_over"):
            return None

        vec = encode_state(state, self.prev_state)
        self.prev_state = state

        with torch.no_grad():
            t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
            action_idx = int(self.net(t).argmax(dim=1).item())

        return ACTIONS[action_idx]


if __name__ == "__main__":
    agent = DQNAgent()
    asyncio.run(agent.run())
