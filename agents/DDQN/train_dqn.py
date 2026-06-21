__author__ = "Mário Antunes"
__version__ = "1.0.0"
__status__ = "Development"

import asyncio
import csv
import json
import logging
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import websockets

from agents.DDQN.dqn_common import ACTIONS, STATE_DIM, encode_state
from agents.DDQN.dqn_model import QNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s - TRAIN - %(levelname)s - %(message)s")

SERVER_URI = "ws://localhost:8765/ws"
N_ACTIONS = len(ACTIONS)

GAMMA = 0.99
LR = 5e-4
BATCH_SIZE = 64
BUFFER_SIZE = 200000
MIN_BUFFER = 3000
TAU = 0.005
TRAIN_EVERY = 4
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_EPISODES = 160
NUM_EPISODES = 200
MAX_STEPS_PER_EPISODE = 20000
SAVE_EVERY = 5
MODEL_PATH = "dqn_breakout.pt"
METRICS_PATH = "training_metrics.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = QNetwork(STATE_DIM, N_ACTIONS).to(device)
target_net = QNetwork(STATE_DIM, N_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = deque(maxlen=BUFFER_SIZE)
steps_done = 0


def select_action(vec, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    with torch.no_grad():
        t = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        return int(policy_net(t).argmax(dim=1).item())


def optimize():
    if len(buffer) < max(BATCH_SIZE, MIN_BUFFER):
        return None

    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)

    with torch.no_grad():
        next_actions = policy_net(next_states_t).argmax(dim=1, keepdim=True)
        next_q = target_net(next_states_t).gather(1, next_actions).squeeze(1)
        target = rewards_t + GAMMA * next_q * (1.0 - dones_t)

    loss = nn.functional.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    return loss.item()


async def run_episode(epsilon: float):
    global steps_done
    total_reward = 0.0
    prev_state = None
    prev_vec = None
    action_idx = None
    episode_steps = 0
    losses = []
    final_score = 0

    async with websockets.connect(SERVER_URI) as ws:
        await ws.send(json.dumps({"client": "agent"}))

        async for message in ws:
            data = json.loads(message)
            if data.get("type") == "setup":
                continue

            state = data
            episode_steps += 1
            if episode_steps >= MAX_STEPS_PER_EPISODE:
                break

            vec = encode_state(state, prev_state)
            done = bool(state.get("game_over"))

            if prev_vec is not None:
                reward = state["score"] - prev_state["score"]
                if state["lives"] < prev_state["lives"]:
                    reward -= 10.0
                if done:
                    reward -= 20.0
                reward -= 0.01 * abs(vec[2] - vec[1])
                total_reward += reward

                buffer.append((prev_vec, action_idx, reward, vec, float(done)))
                steps_done += 1
                if steps_done % TRAIN_EVERY == 0:
                    loss_val = optimize()
                    if loss_val is not None:
                        losses.append(loss_val)

            final_score = state["score"]

            if done:
                break

            action_idx = select_action(vec, epsilon)
            action = ACTIONS[action_idx]
            if action is not None:
                await ws.send(json.dumps(action))

            prev_state = state
            prev_vec = vec

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return total_reward, episode_steps, final_score, avg_loss


async def main():
    with open(METRICS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "epsilon", "buffer_size", "steps", "score", "avg_loss"])
        f.flush()

        for episode in range(1, NUM_EPISODES + 1):
            epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * episode / EPS_DECAY_EPISODES)

            try:
                total_reward, ep_steps, final_score, avg_loss = await run_episode(epsilon)
            except Exception as e:
                logging.error(f"Episódio {episode} falhou: {e}")
                await asyncio.sleep(1.0)
                continue

            writer.writerow([episode, total_reward, epsilon, len(buffer), ep_steps, final_score, avg_loss])
            f.flush()

            logging.info(
                f"Episódio {episode} | reward={total_reward:.1f} | score={final_score} | "
                f"epsilon={epsilon:.3f} | buffer={len(buffer)}"
            )

            if episode % SAVE_EVERY == 0:
                torch.save(policy_net.state_dict(), MODEL_PATH)
                logging.info(f"Modelo guardado em {MODEL_PATH}")

    torch.save(policy_net.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        torch.save(policy_net.state_dict(), MODEL_PATH)
        logging.info("Treino interrompido manualmente, modelo guardado.")