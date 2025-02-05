import matplotlib

matplotlib.use('TkAgg')

import tkinter as tk
from checkers_env import CheckersEnv
from CheckerGUI import CheckerGUI
from LearningAgent import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

            reward = raw_reward
            if done:
                winner = env.game_winner()
                if winner == 1:
                    reward = 10
                    win_history.append(1)
                elif winner == 2:
                    reward = -5
                    win_history.append(0)
                else:
                    reward = 1
                    win_history.append(0.5)

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

    smoothed_rewards = np.convolve(total_rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(smoothed_rewards, label=f"Smoothed (window={window_size})", color='blue', linewidth=2)

    sampled_episodes = np.arange(window_size - 1, len(total_rewards), 200)
    sampled_rewards = [np.mean(total_rewards[max(0, i - window_size):i]) for i in sampled_episodes]
    plt.scatter(sampled_episodes, sampled_rewards, color='red', s=5, label="Sampled Points")

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
    smoothed_win_rate = np.convolve(win_history, np.ones(window_size) / window_size, mode='valid')
    plt.plot(smoothed_win_rate, label=f"Smoothed Win Rate (window={window_size})", color='green', linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("AI Win Rate Over Time (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_q_values(q_table):
    """Plot Q-value distribution."""
    q_values = []
    if not q_table:
        print("⚠️ Q-table is empty! No Q-values to plot.")
        return

    for state in q_table:
        if np.any(q_table[state] != 0):
            q_values.extend(q_table[state])

    if not q_values:
        print("⚠️ No meaningful Q-values found in Q-table!")
        return

    plt.figure(figsize=(10, 5))
    sns.histplot(q_values, bins=50, kde=True, color='purple')
    plt.xlabel("Q Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Q Values in Q-Learning Agent")
    plt.grid(True)
    plt.show()


def compare_ai_performance(env, agent_before, agent_after, num_games=100):
    """Compare AI performance before and after training."""
    before_wins, after_wins = 0, 0

    for _ in range(num_games):
        env.reset()
        done = False

        while not done:
            if env.player == 1:
                action = agent_before.choose_action(env.board)
            else:
                action = agent_after.choose_action(env.board)

            if action is None:
                break

            _, _, done = env.step(action, env.player)

        winner = env.game_winner()
        if winner == 1:
            before_wins += 1
        elif winner == 2:
            after_wins += 1

    print(f"🏆 Untrained AI Win Rate: {before_wins / num_games:.2%}")
    print(f"🎯 Trained AI Win Rate: {after_wins / num_games:.2%}")


if __name__ == "__main__":
    # Run the game GUI
    root = tk.Tk()
    gui = CheckerGUI(root, difficulty='medium')
    root.mainloop()

    # Initialize AI and train
    env = CheckersEnv(board_size=8)
    agent1 = QLearningAgent(env, player=1, difficulty="easy")
    agent2 = QLearningAgent(env, player=2, difficulty="medium")

    # 🚨 Debugging: Ensure agents can choose moves
    test_action_1 = agent1.choose_action(env.board)
    test_action_2 = agent2.choose_action(env.board)
    print(f"🔍 Agent 1 Test Action: {test_action_1}")
    print(f"🔍 Agent 2 Test Action: {test_action_2}")

    total_rewards, win_history, _ = train_agent(env, agent1, agent2, num_episodes=10000)

    # 🚨 Debugging: Check Q-table sizes
    print(f"📌 Agent 1 Q-table size: {len(agent1.q_table)}")
    print(f"📌 Agent 2 Q-table size: {len(agent2.q_table)}")

    agent1.save_q_table(agent1.q_table_file)
    agent2.save_q_table(agent2.q_table_file)

    # Plot training results
    plot_training_results(total_rewards)
    plot_win_rate(win_history)
    plot_q_values(agent2.q_table)
    compare_ai_performance(env, agent1, agent2, num_games=100)
