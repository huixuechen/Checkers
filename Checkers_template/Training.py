import matplotlib

matplotlib.use('TkAgg')  # Use the TkAgg backend to prevent segmentation fault
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os


class ModelTracker:
    def __init__(self, log_file="training_log.json"):
        self.log_file = log_file
        self.logs = []
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                self.logs = [json.loads(line) for line in f]

    def log_training(self, episode, reward, win_rate, exploration_rate):
        log_entry = {
            "episode": episode,
            "reward": reward,
            "win_rate": win_rate,
            "exploration_rate": exploration_rate,
        }
        self.logs.append(log_entry)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def plot_win_rate(self):
        episodes = [log["episode"] for log in self.logs]
        win_rates = [log["win_rate"] for log in self.logs]
        plt.plot(episodes, win_rates, label="Win Rate")
        plt.xlabel("Episodes")
        plt.ylabel("Win Rate")
        plt.title("Win Rate Over Time")
        plt.legend()
        plt.show()

    def plot_average_reward(self):
        episodes = [log["episode"] for log in self.logs]
        rewards = [log["reward"] for log in self.logs]
        plt.plot(episodes, rewards, label="Average Reward", color="orange")
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.title("Average Reward Over Time")
        plt.legend()
        plt.show()

    def plot_exploration_rate(self):
        episodes = [log["episode"] for log in self.logs]
        exploration_rates = [log["exploration_rate"] for log in self.logs]
        plt.plot(episodes, exploration_rates, label="Exploration Rate", color="green")
        plt.xlabel("Episodes")
        plt.ylabel("Exploration Rate")
        plt.title("Exploration Rate Over Time")
        plt.legend()
        plt.show()

    def display_logs(self):
        print("Training Logs:")
        for log in self.logs:
            print(log)

    def save_summary(self, summary_file="training_summary.json"):
        summary = {
            "total_episodes": len(self.logs),
            "final_win_rate": self.logs[-1]["win_rate"] if self.logs else 0,
            "final_average_reward": np.mean([log["reward"] for log in self.logs]) if self.logs else 0,
            "final_exploration_rate": self.logs[-1]["exploration_rate"] if self.logs else 0,
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)
