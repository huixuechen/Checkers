import tkinter as tk
from checkers_env import checkers_env
from CheckerGUI import CheckerGUI
from LearningAgent import QLearningAgent

if __name__ == "__main__":
    root = tk.Tk()
    gui = CheckerGUI(root, difficulty='low')
    root.mainloop()

    # Initialize the environment and the Q-learning agent
    env = checkers_env(board_size=6)
    agent = QLearningAgent(env, player=1)  # Pass the player argument

    num_episodes = 1000

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action, env.player)
            agent.learn(state, action, reward, next_state)
            state = next_state

        agent.update_exploration_rate()
        print(f"Episode {episode + 1}/{num_episodes} completed.")


