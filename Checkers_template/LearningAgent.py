import numpy as np
import random
from collections import defaultdict
from TaskSimilarity import TaskSimilarity
import json
import os

class QLearningAgent:
    def __init__(self, env, player, board_size=6, difficulty='easy'):
        self.env = env
        self.player = player
        self.board_size = board_size
        self.difficulty = difficulty
        self.task_similarity = TaskSimilarity()
        self.q_table = defaultdict(lambda: np.zeros(len(self.env.valid_moves(self.player) or [0])))
        self.exploration_log = []
        self.visits = defaultdict(int)

        # 设置难度
        self.set_difficulty(self.difficulty)

    def set_difficulty(self, difficulty):
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.learning_rate = 0.1
            self.discount_factor = 0.6
            self.exploration_rate = 0.9
            self.exploration_decay = 0.995
            self.attack_bonus = 20

        elif difficulty == "medium":
            self.learning_rate = 0.15
            self.discount_factor = 0.75
            self.exploration_rate = 0.7
            self.exploration_decay = 0.99
            self.attack_bonus = 30

        elif difficulty == "hard":
            self.learning_rate = 0.25
            self.discount_factor = 0.9
            self.exploration_rate = 0.5
            self.exploration_decay = 0.995
            self.attack_bonus = 50
        else:
            raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', or 'hard'.")

        self.min_exploration_rate = 0.1

        self.q_table_file = f"q_table_{difficulty}.json"


        self.load_q_table(self.q_table_file)



    def choose_action(self, state, use_ucb=False):
        valid_moves = self.env.valid_moves(self.player)

        if not valid_moves:
            return None  # 没有合法移动

        state_hash = self.state_to_hash(state)
        self.visits[state_hash] += 1  # 记录访问次数

        # **1. 先分类普通移动和跳跃**
        jump_moves = [move for move in valid_moves if abs(move[2] - move[0]) == 2]  # 只能跳跃相邻棋子
        normal_moves = [move for move in valid_moves if abs(move[2] - move[0]) == 1]  # 普通移动只能前进一步

        action = None

        # **2. 强制 AI 先跳跃**
        if jump_moves:
            print(f"⚠️ AI Player {self.player} must jump: {jump_moves}")
            action = random.choice(jump_moves)  # **必须先吃子**

        else:  # **只有当没有跳跃时，才允许普通移动**
            # **严格限制普通移动只能前进一步**
            if normal_moves:
                print(f"✅ AI Player {self.player} performs normal move: {normal_moves}")
                action = random.choice(normal_moves)  # **只能移动 1 步**

            else:
                return None  # **如果没有可用的普通移动，也不能乱选**

        # **最终安全检查**
        if action not in valid_moves:
            print(f"⚠️ AI picked an invalid move {action}, forcing a random valid move!")
            action = random.choice(valid_moves)

        return action

    def learn(self, state, action, reward, next_state):
        """更新 Q 表并存储任务相似性"""
        valid_moves = self.env.valid_moves(self.player)
        if not valid_moves or action not in valid_moves:
            return  # 没有合法移动，直接返回

        state_hash = self.state_to_hash(state)
        next_state_hash = self.state_to_hash(next_state)

        # **确保 Q-table 初始化**
        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.full(len(valid_moves), -1.0)  # 仍然使用 -1.0 作为初始值

        if next_state_hash not in self.q_table:
            self.q_table[next_state_hash] = np.zeros(len(valid_moves))  # 初始化 Q 值

        try:
            action_index = valid_moves.index(action)
        except ValueError:
            print(f"⚠️ Action {action} not found in valid_moves: {valid_moves}")
            return  # **如果动作无效，直接返回**

        # **获取最佳下一步动作**
        if len(self.q_table[next_state_hash]) > 0:  # **确保 next_state_hash 有数据**
            best_next_action = np.argmax(self.q_table[next_state_hash])
        else:
            best_next_action = 0  # **如果 `q_table[next_state_hash]` 为空，默认 0**

        # **TD 目标计算**
        td_target = reward + self.discount_factor * self.q_table[next_state_hash][best_next_action]
        td_error = td_target - self.q_table[state_hash][action_index]
        self.q_table[state_hash][action_index] += self.learning_rate * td_error  # **更新 Q 值**

        # **打印调试信息**
        print(
            f"Updated Q-table: {state_hash} | Action {action} | New Q-value: {self.q_table[state_hash][action_index]}")

        try:
            self.task_similarity.store_state(state, action)  # **存储任务相似性**
        except Exception as e:
            print(f"⚠️ Error storing state in task_similarity: {e}")

    def state_to_hash(self, state):
        return hash(tuple(state.flatten())) if state is not None else 0

    def update_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def select_ucb_action(self, state, valid_moves):
        state_hash = self.state_to_hash(state)
        total_visits = sum(self.visits.values()) + 1
        ucb_values = []

        for action_index in range(len(valid_moves)):
            q_value = self.q_table[state_hash][action_index]
            visit_count = self.visits[state_hash] + 1
            ucb_value = q_value + 2 * np.sqrt(np.log(total_visits) / visit_count)
            ucb_values.append(ucb_value)

        return valid_moves[np.argmax(ucb_values)]

    def save_q_table(self, filepath):
        """使用 JSON 存储 Q-table"""
        q_table_dict = {str(state): q_values.tolist() for state, q_values in self.q_table.items()}  # 转换 JSON 格式

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(q_table_dict, f, indent=4)  # **格式化存储，方便手动修改**
        print(f"✅ Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        """从 JSON 加载 Q-table，如果文件不存在，则创建一个空白 Q-table"""
        if not os.path.exists(filepath):
            print(f"⚠️ Q-table file not found ({filepath}), creating a new empty Q-table.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.env.valid_moves(self.player) or [0])))
            self.save_q_table(filepath)  # **创建空白 Q-table 并保存**
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded_q_table = json.load(f)

            # 还原 Q-table
            self.q_table = defaultdict(
                lambda: np.zeros(len(self.env.valid_moves(self.player) or [0])),
                {int(state): np.array(q_values) for state, q_values in loaded_q_table.items()},
            )
            print(f"✅ Q-table loaded from {filepath}")
        except json.JSONDecodeError:
            print("❌ Error: Invalid JSON format! Please check the Q-table file.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.env.valid_moves(self.player) or [0])))

    def update_exploration_rate(self):
        """更新探索率，防止探索率降到 0"""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
