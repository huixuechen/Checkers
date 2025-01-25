import tkinter as tk
from checkers_env import CheckersEnv
from CheckerGUI import CheckerGUI
from LearningAgent import QLearningAgent
import matplotlib.pyplot as plt


def train_agent(env, agent, num_episodes=10000):
    total_rewards = []  # Track cumulative rewards

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0  # Current episode cumulative reward

        while not done:
            action = agent.choose_action(state)
            next_state, raw_reward, done = env.step(action)
            if done and env.game_winner() == agent.player:
                reward = 100  # High reward for winning
            elif done:
                reward = -50  # Severe penalty for losing
            elif env.is_dangerous_action(action):
                reward = -10  # Mild penalty for dangerous move
            else:
                reward = 1  # Small default reward

            agent.learn(state, action, reward, next_state)
            state = next_state
            episode_reward += reward  # Accumulate reward for the episode
            agent.update_exploration_rate()
        total_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = sum(total_rewards[-100:]) / 100
            print(f"Episode {episode + 1}/{num_episodes}, Average reward: {avg_reward}")

    return total_rewards


def plot_training_results(total_rewards):
    plt.figure()
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Time')
    plt.show()


if __name__ == "__main__":
    # Run the game GUI
    root = tk.Tk()
    gui = CheckerGUI(root, difficulty='easy')  # Change difficulty here
    root.mainloop()

    # Initialize the environment and the Q-learning agent
    env = CheckersEnv(board_size=6)
    agent = QLearningAgent(env, player=1)  # Pass the player argument

    # Train the agent and plot the results
    total_rewards = train_agent(env, agent, num_episodes=10000)
    plot_training_results(total_rewards)