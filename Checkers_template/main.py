import tkinter as tk
from checkers_env import CheckersEnv
from CheckerGUI import CheckerGUI
from LearningAgent import QLearningAgent
import matplotlib.pyplot as plt

def train_agent(env, agent1, agent2, num_episodes=10000):
    total_rewards = []  # Track cumulative rewards

    for episode in range(num_episodes):
        env.reset()
        state = env.board.copy()
        done = False
        episode_reward = 0  # Current episode cumulative reward

        while not done:
            current_agent = agent1 if env.player == 1 else agent2
            action = current_agent.choose_action(state)
            if action is None:
                break  # No valid moves available, end game

            next_state, raw_reward, done = env.step(action, env.player)

            if done and env.game_winner() == env.player:
                reward = 100  # High reward for winning
            elif done:
                reward = -50  # Severe penalty for losing
            else:
                reward = raw_reward  # Use default reward from env

            current_agent.learn(state, action, reward, next_state)
            state = next_state
            episode_reward += reward  # Accumulate reward for the episode
            agent1.update_exploration_rate()
            agent2.update_exploration_rate()
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

    # 初始化 AI 并训练
    env = CheckersEnv(board_size=6)
    agent1 = QLearningAgent(env, player=1, difficulty="medium")  # AI Player 1
    agent2 = QLearningAgent(env, player=2, difficulty="medium")  # AI Player 2
    total_rewards = train_agent(env, agent1, agent2, num_episodes=10000)
    plot_training_results(total_rewards)
