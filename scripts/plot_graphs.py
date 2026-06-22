import argparse
import csv
import os
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DQN training metrics.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root directory for outputs")
    parser.add_argument("--metrics-name", type=str, default="training_metrics.csv", help="Name of log metrics file")
    parser.add_argument("--window", type=int, default=10, help="Moving average window size")
    return parser.parse_args()


def load_metrics(path):
    episodes, rewards, epsilons, buffer_sizes, steps, scores, losses = [], [], [], [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            epsilons.append(float(row["epsilon"]))
            buffer_sizes.append(int(row["buffer_size"]))
            steps.append(int(row["steps"]))
            scores.append(float(row["score"]))
            losses.append(float(row["avg_loss"]))
    return episodes, rewards, epsilons, buffer_sizes, steps, scores, losses


def moving_average(values, window):
    if len(values) < window:
        return [], []
    avgs = []
    for i in range(window - 1, len(values)):
        avgs.append(sum(values[i - window + 1:i + 1]) / window)
    return list(range(window, len(values) + 1)), avgs


def plot_convergence(episodes, rewards, plots_dir, window):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Reward per episode")
    ma_x, ma_y = moving_average(rewards, window)
    if ma_y:
        plt.plot(ma_x, ma_y, color="red", linewidth=2, label=f"Moving Avg ({window} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Convergence (Reward per Episode)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "convergence_plot.png"), dpi=150)
    plt.close()


def plot_score(episodes, scores, plots_dir, window):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores, alpha=0.3, label="Score per episode")
    ma_x, ma_y = moving_average(scores, window)
    if ma_y:
        plt.plot(ma_x, ma_y, color="green", linewidth=2, label=f"Moving Avg ({window} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Final Game Score")
    plt.title("Game Score along Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "score_plot.png"), dpi=150)
    plt.close()


def plot_loss(episodes, losses, plots_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, losses, color="purple", alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss (Smooth L1)")
    plt.title("Neural Network Loss Evolution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_plot.png"), dpi=150)
    plt.close()


def plot_epsilon(episodes, epsilons, plots_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilons, color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay (Exploration vs Exploitation)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "epsilon_plot.png"), dpi=150)
    plt.close()


def plot_episode_length(episodes, steps, plots_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, steps, color="brown", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Survived Frames")
    plt.title("Episode Duration (Survival Proxy)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "episode_duration_plot.png"), dpi=150)
    plt.close()


def plot_buffer_growth(episodes, buffer_sizes, plots_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, buffer_sizes, color="teal")
    plt.xlabel("Episode")
    plt.ylabel("Replay Buffer Transitions")
    plt.title("Replay Buffer Growth Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "buffer_growth_plot.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    logs_dir = os.path.join(args.output_dir, "logs")
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metrics_path = os.path.join(logs_dir, args.metrics_name)
    
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        return

    episodes, rewards, epsilons, buffer_sizes, steps, scores, losses = load_metrics(metrics_path)

    plot_convergence(episodes, rewards, plots_dir, args.window)
    plot_score(episodes, scores, plots_dir, args.window)
    plot_loss(episodes, losses, plots_dir)
    plot_epsilon(episodes, epsilons, plots_dir)
    plot_episode_length(episodes, steps, plots_dir)
    plot_buffer_growth(episodes, buffer_sizes, plots_dir)

    print(f"Processed {len(episodes)} episodes from {metrics_path}.")
    print(f"Plots saved to: {plots_dir}/")


if __name__ == "__main__":
    main()