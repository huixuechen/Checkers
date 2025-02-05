import matplotlib

matplotlib.use('TkAgg')

import tkinter as tk
from checkers_env import CheckersEnv
from CheckerGUI import CheckerGUI
from LearningAgent import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def smooth_rewards(rewards, alpha=0.01):
    """Apply exponential moving average to smooth rewards."""
    smoothed = []
    last_value = rewards[0]
    for r in rewards:
        last_value = alpha * r + (1 - alpha) * last_value
        smoothed.append(last_value)
    return smoothed


def train_agent(env, agent1, agent2, num_episodes=10000):
    """Train AI agents and debug Agent 2's issue."""
    total_rewards = []  # Tracks cumulative rewards
    win_history = []  # Tracks how often Agent 1 wins

    for episode in range(num_episodes):
        env.reset()
        state = env.board.copy()
        done = False
        episode_reward = 0  # Tracks reward per episode

        while not done:
            current_agent = agent1 if env.player == 1 else agent2
            action = current_agent.choose_action(state)

            # 🚨 Debugging: Print chosen action
            print(f"🔹 Agent {env.player} chose action: {action}")

            if action is None:
                print(f"⚠️ Agent {env.player} has no valid moves! Skipping turn.")
                break  # Avoid infinite loops if no moves available

            next_state, raw_reward, done = env.step(action, env.player)

            # 🚨 Debugging: Ensure turn switch is working
            print(f"🔄 Switching to Agent {3 - env.player} after move.")

            # Properly switch turns
            env.player = 3 - env.player

            if done:
                winner = env.game_winner()
                if winner == 1:
                    reward = 10  # Agent 1 wins
                    win_history.append(1)
                elif winner == 2:
                    reward = -5  # Agent 2 wins
                    win_history.append(0)
                else:
                    reward = 2  # Draw
                    win_history.append(0.5)
            else:
                # Reward for capturing a piece
                if env.capture_piece(action, env.player):
                    reward = 3
                else:
                    reward = 0.1  # Encourage legal moves

            current_agent.learn(state, action, reward, next_state)
            state = next_state
            episode_reward += reward

            # Update exploration rates
            agent1.update_exploration_rate()
            agent2.update_exploration_rate()

        total_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = sum(total_rewards[-100:]) / 100
            print(f"📈 Episode {episode + 1}/{num_episodes}, Avg reward: {avg_reward}")

    return total_rewards, win_history, dict(agent1.q_table)


def plot_training_results(total_rewards, window_size=200):
    """Plot smoothed training rewards over time."""
    plt.figure(figsize=(10, 5))
    smoothed_rewards = smooth_rewards(total_rewards)
    plt.plot(smoothed_rewards, label=f"Smoothed (EMA, alpha=0.01)", color='blue', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Time (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_win_rate(win_history, window_size=200):
    """Plot AI win rate over time."""
    if not win_history:
        print("⚠️ No win history available! Skipping plot.")
        return
    plt.figure(figsize=(10, 5))
    smoothed_win_rate = smooth_rewards(win_history)
    plt.plot(smoothed_win_rate, label=f"Smoothed Win Rate (EMA, alpha=0.01)", color='green', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("AI Win Rate Over Time (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    gui = CheckerGUI(root, difficulty='medium')
    root.mainloop()

    env = CheckersEnv(board_size=8)
    agent1 = QLearningAgent(env, player=1, difficulty="easy")
    agent2 = QLearningAgent(env, player=2, difficulty="hard")

    total_rewards, win_history, _ = train_agent(env, agent1, agent2, num_episodes=10000)
    plot_training_results(total_rewards)
    plot_win_rate(win_history)
