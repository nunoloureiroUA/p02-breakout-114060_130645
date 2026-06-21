__author__ = "Mário Antunes"
__version__ = "1.0.0"
__status__ = "Development"

import csv

import matplotlib.pyplot as plt

METRICS_PATH = "training_metrics.csv"
OUTPUT_DIR = "."
MOVING_AVG_WINDOW = 10


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


def plot_convergence(episodes, rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Reward por episódio")
    ma_x, ma_y = moving_average(rewards, MOVING_AVG_WINDOW)
    if ma_y:
        plt.plot(ma_x, ma_y, color="red", linewidth=2, label=f"Média móvel ({MOVING_AVG_WINDOW} episódios)")
    plt.xlabel("Episódio")
    plt.ylabel("Reward total")
    plt.title("Convergência do treino (reward por episódio)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/grafico_convergencia.png", dpi=150)
    plt.close()


def plot_score(episodes, scores):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores, alpha=0.3, label="Score por episódio")
    ma_x, ma_y = moving_average(scores, MOVING_AVG_WINDOW)
    if ma_y:
        plt.plot(ma_x, ma_y, color="green", linewidth=2, label=f"Média móvel ({MOVING_AVG_WINDOW} episódios)")
    plt.xlabel("Episódio")
    plt.ylabel("Score final do jogo")
    plt.title("Score do jogo ao longo do treino")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/grafico_score.png", dpi=150)
    plt.close()


def plot_loss(episodes, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, losses, color="purple", alpha=0.6)
    plt.xlabel("Episódio")
    plt.ylabel("Loss média (smooth L1)")
    plt.title("Evolução da loss da rede")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/grafico_loss.png", dpi=150)
    plt.close()


def plot_epsilon(episodes, epsilons):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilons, color="orange")
    plt.xlabel("Episódio")
    plt.ylabel("Epsilon")
    plt.title("Decaimento de epsilon (exploração vs exploração do que já sabe)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/grafico_epsilon.png", dpi=150)
    plt.close()


def plot_episode_length(episodes, steps):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, steps, color="brown", alpha=0.7)
    plt.xlabel("Episódio")
    plt.ylabel("Frames sobrevividos")
    plt.title("Duração do episódio (proxy de sobrevivência)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/grafico_duracao_episodio.png", dpi=150)
    plt.close()


def plot_buffer_growth(episodes, buffer_sizes):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, buffer_sizes, color="teal")
    plt.xlabel("Episódio")
    plt.ylabel("Transições no buffer")
    plt.title("Crescimento do replay buffer")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/grafico_buffer.png", dpi=150)
    plt.close()


def main():
    episodes, rewards, epsilons, buffer_sizes, steps, scores, losses = load_metrics(METRICS_PATH)

    plot_convergence(episodes, rewards)
    plot_score(episodes, scores)
    plot_loss(episodes, losses)
    plot_epsilon(episodes, epsilons)
    plot_episode_length(episodes, steps)
    plot_buffer_growth(episodes, buffer_sizes)

    print(f"{len(episodes)} episódios lidos de {METRICS_PATH}. Gráficos guardados em {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
