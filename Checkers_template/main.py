# main.py
import tkinter as tk
from checkers_env import checkers_env
from CheckerGUI import CheckerGUI
from LearningAgent import QLearningAgent

if __name__ == "__main__":
    root = tk.Tk()
    gui = CheckerGUI(root, difficulty='medium')  # Change difficulty here
    root.mainloop()

    # Initialize the environment and the Q-learning agent
    env = checkers_env(board_size=6)
    agent = QLearningAgent(env, player=1)  # Pass the player argument
    num_episodes = 10000
    total_rewards = []  # Track cumulative rewards

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0  # Current episode cumulative reward

        while not done:
            action = agent.choose_action(state)
            next_state, raw_reward, done = env.step(action, env.player)
            if done and env.is_winner(agent.player):
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