import argparse
import asyncio
import csv
import json
import logging
import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import websockets

from src.agents.ddqn.dqn_common import ACTIONS, STATE_DIM, encode_state
from src.agents.ddqn.dqn_model import QNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s - TRAIN - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Double DQN agent on Breakout.")
    parser.add_argument("--server-uri", type=str, default="ws://localhost:8765/ws", help="Server WebSocket URI")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--buffer-size", type=int, default=200000, help="Maximum capacity of replay buffer")
    parser.add_argument("--min-buffer", type=int, default=3000, help="Minimum buffer size before training starts")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update parameter for target network")
    parser.add_argument("--train-every", type=int, default=4, help="Step frequency to run optimization")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--eps-end", type=float, default=0.1, help="Ending epsilon for exploration")
    parser.add_argument("--eps-decay-episodes", type=int, default=170, help="Number of episodes over which epsilon decays")
    parser.add_argument("--num-episodes", type=int, default=200, help="Total number of episodes to train")
    parser.add_argument("--max-steps", type=int, default=20000, help="Max steps allowed per episode")
    parser.add_argument("--save-every", type=int, default=5, help="Save model weights every N episodes")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root directory for outputs")
    parser.add_argument("--model-name", type=str, default="dqn_breakout.pt", help="Filename for the saved model weights")
    parser.add_argument("--metrics-name", type=str, default="training_metrics.csv", help="Filename for training logs")
    return parser.parse_args()


args = parse_args()
N_ACTIONS = len(ACTIONS)

# Setup subdirectories under output_dir
MODELS_DIR = os.path.join(args.output_dir, "models")
LOGS_DIR = os.path.join(args.output_dir, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, args.model_name)
METRICS_PATH = os.path.join(LOGS_DIR, args.metrics_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = QNetwork(STATE_DIM, N_ACTIONS).to(device)
target_net = QNetwork(STATE_DIM, N_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
buffer = deque(maxlen=args.buffer_size)
steps_done = 0


def select_action(vec, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    with torch.no_grad():
        t = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        return int(policy_net(t).argmax(dim=1).item())


def optimize():
    if len(buffer) < max(args.batch_size, args.min_buffer):
        return None

    batch = random.sample(buffer, args.batch_size)
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
        target = rewards_t + args.gamma * next_q * (1.0 - dones_t)

    loss = nn.functional.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

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

    async with websockets.connect(args.server_uri) as ws:
        await ws.send(json.dumps({"client": "agent"}))

        async for message in ws:
            data = json.loads(message)
            if data.get("type") == "setup":
                continue

            state = data
            episode_steps += 1
            if episode_steps >= args.max_steps:
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
                if steps_done % args.train_every == 0:
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

        for episode in range(1, args.num_episodes + 1):
            epsilon = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * episode / args.eps_decay_episodes)

            try:
                total_reward, ep_steps, final_score, avg_loss = await run_episode(epsilon)
            except Exception as e:
                logging.error(f"Episode {episode} failed: {e}")
                await asyncio.sleep(1.0)
                continue

            writer.writerow([episode, total_reward, epsilon, len(buffer), ep_steps, final_score, avg_loss])
            f.flush()

            logging.info(
                f"Episode {episode} | reward={total_reward:.1f} | score={final_score} | "
                f"epsilon={epsilon:.3f} | buffer={len(buffer)}"
            )

            if episode % args.save_every == 0:
                torch.save(policy_net.state_dict(), MODEL_PATH)
                logging.info(f"Model checkpoint saved to {MODEL_PATH}")

    torch.save(policy_net.state_dict(), MODEL_PATH)
    logging.info(f"Final model successfully saved to {MODEL_PATH}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        torch.save(policy_net.state_dict(), MODEL_PATH)
        logging.info("Training manually interrupted, saving weights before exit.")