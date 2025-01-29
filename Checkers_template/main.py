import tkinter as tk
from checkers_env import CheckersEnv
from CheckerGUI import CheckerGUI
from LearningAgent import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def train_agent(env, agent1, agent2, num_episodes=10000):
    """训练 AI 并记录奖励和胜率"""
    total_rewards = []  # 累计奖励
    win_history = []  # 记录 AI1 胜利的次数

    for episode in range(num_episodes):
        env.reset()
        state = env.board.copy()
        done = False
        episode_reward = 0  # 当前回合总奖励

        while not done:
            current_agent = agent1 if env.player == 1 else agent2
            action = current_agent.choose_action(state)
            if action is None:
                break  # 没有合法移动，结束游戏

            next_state, raw_reward, done = env.step(action, env.player)

            # 计算奖励
            if done and env.game_winner() == 1:
                reward = 100  # AI1 获胜
                win_history.append(1)  # 记录胜利
            elif done and env.game_winner() == 2:
                reward = -50  # AI2 获胜
                win_history.append(0)  # 记录失败
            else:
                reward = raw_reward  # 正常奖励

            current_agent.learn(state, action, reward, next_state)
            state = next_state
            episode_reward += reward  # 累计当前回合奖励
            agent1.update_exploration_rate()
            agent2.update_exploration_rate()

        total_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = sum(total_rewards[-100:]) / 100
            print(f"Episode {episode + 1}/{num_episodes}, Average reward: {avg_reward}")

    return total_rewards, win_history, dict(agent1.q_table)  # **确保返回3个值**


def plot_training_results(total_rewards, window_size=100):
    """优化训练奖励的可视化"""
    plt.figure(figsize=(10, 5))

    # 计算滑动平均
    smoothed_rewards = np.convolve(total_rewards, np.ones(window_size)/window_size, mode='valid')

    # 绘制平滑曲线
    plt.plot(smoothed_rewards, label=f"Smoothed (window={window_size})", color='blue', linewidth=2)

    # 仅每 100 个回合采样一个点，减少点的密集度
    sampled_episodes = np.arange(window_size - 1, len(total_rewards), 100)
    sampled_rewards = [np.mean(total_rewards[max(0, i - window_size):i]) for i in sampled_episodes]
    plt.scatter(sampled_episodes, sampled_rewards, color='red', s=5, label="Sampled Points")

    # 添加标题、标签和网格
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Time (Smoothed)")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_win_rate(win_history, window_size=100):
    """绘制 AI 的胜率随时间变化的曲线"""
    plt.figure(figsize=(10, 5))

    # 计算滑动平均胜率
    smoothed_win_rate = np.convolve(win_history, np.ones(window_size)/window_size, mode='valid')

    plt.plot(smoothed_win_rate, label=f"Smoothed Win Rate (window={window_size})", color='green', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("AI Win Rate Over Time (Smoothed)")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_q_values(q_table):
    """绘制 Q 值分布"""
    q_values = []

    # **检查 Q 表是否为空**
    if not q_table:
        print("⚠️ Q-table is empty! No Q-values to plot.")
        return

    # **提取 Q 值**
    for state in q_table:
        if np.any(q_table[state] != 0):  # **跳过全是 0 的 Q 值**
            q_values.extend(q_table[state])

    # **检查 Q 值是否有效**
    if not q_values:
        print("⚠️ No meaningful Q-values found in Q-table!")
        return

    # **绘制 Q 值直方图**
    plt.figure(figsize=(10, 5))
    sns.histplot(q_values, bins=50, kde=True, color='purple')
    plt.xlabel("Q Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Q Values in Q-Learning Agent")
    plt.grid(True)
    plt.show()

def compare_ai_performance(env, agent_before, agent_after, num_games=100):
    """Compare AI performance before and after training"""
    before_wins = 0
    after_wins = 0

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

        if env.game_winner() == 1:
            before_wins += 1
        elif env.game_winner() == 2:
            after_wins += 1

    print(f"Untrained AI Win Rate: {before_wins / num_games:.2%}")
    print(f"Trained AI Win Rate: {after_wins / num_games:.2%}")




if __name__ == "__main__":
    # Run the game GUI
    root = tk.Tk()
    gui = CheckerGUI(root, difficulty='easy')  # Change difficulty here
    root.mainloop()

    # 初始化 AI 并训练
    env = CheckersEnv(board_size=6)
    difficulty = "medium"  # 选择训练难度
    agent1 = QLearningAgent(env, player=1, difficulty="easy")  # AI Player 1
    agent2 = QLearningAgent(env, player=2, difficulty="medium")  # AI Player 2
    total_rewards, win_history, _ = train_agent(env, agent1, agent2, num_episodes=10000)


    agent1.save_q_table(agent1.q_table_file)
    agent2.save_q_table(agent2.q_table_file)


    plot_training_results(total_rewards)
    plot_win_rate(win_history)
    plot_q_values(agent2.q_table)  # Q 值分布
    compare_ai_performance(env, agent1, agent2, num_games=100)
