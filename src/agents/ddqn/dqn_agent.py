import asyncio
import os
from typing import Any, Dict, Optional

import torch
from huggingface_hub import hf_hub_download

from src.agents.base_agent import BaseAgent
from src.agents.ddqn.dqn_common import ACTIONS, STATE_DIM, encode_state
from src.agents.ddqn.dqn_model import QNetwork

SERVER_URI = "ws://localhost:8765/ws"
HF_REPO = "Marinheiro2004/breakout_agent"
MODEL_NAME = "dqn_breakout.pt"
OUTPUT_MODELS_DIR = os.path.join("outputs", "models")


class DQNAgent(BaseAgent):
    def __init__(self, model_path: str, server_uri: str = SERVER_URI) -> None:
        super().__init__(server_uri)
        self.net = QNetwork(STATE_DIM, len(ACTIONS))
        self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
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


def main():
    # Automatically ensure outputs/models exists
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
    local_model_path = os.path.join(OUTPUT_MODELS_DIR, MODEL_NAME)

    print(f"Fetching weights directly from Hugging Face Hub ({HF_REPO})...")
    try:
        hf_hub_download(
            repo_id=HF_REPO,
            filename=MODEL_NAME,
            local_dir=OUTPUT_MODELS_DIR
        )
        print(f"Model pulled successfully! Paths set to: {local_model_path}")
    except Exception as e:
        print(f"Warning: Could not fetch from Hugging Face ({e}). Checking for a cached local copy...")

    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Critical error: Model file could not be downloaded or found at {local_model_path}")

    agent = DQNAgent(model_path=local_model_path)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()