import numpy as np
import random
from collections import defaultdict
from TaskSimilarity import TaskSimilarity

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
            self.learning_rate = 0.05
            self.discount_factor = 0.5
            self.exploration_rate = 0.9
            self.exploration_decay = 0.995
        elif difficulty == "medium":
            self.learning_rate = 0.1
            self.discount_factor = 0.7
            self.exploration_rate = 0.5
            self.exploration_decay = 0.99
        elif difficulty == "hard":
            self.learning_rate = 0.2
            self.discount_factor = 0.9
            self.exploration_rate = 0.2
            self.exploration_decay = 0.995
        else:
            raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', or 'hard'.")
        self.min_exploration_rate = 0.05

    import numpy as np
    import random

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
        valid_moves = self.env.valid_moves(self.player)
        if not valid_moves or action not in valid_moves:
            return

        state_hash = self.state_to_hash(state)
        next_state_hash = self.state_to_hash(next_state)
        action_index = valid_moves.index(action)

        # **防止访问越界**
        if len(valid_moves) > 0 and state_hash not in self.q_table:
            self.q_table[state_hash] = np.full(len(valid_moves), -1.0)

        best_next_action = np.argmax(self.q_table[next_state_hash][:len(valid_moves)])
        td_target = reward + self.discount_factor * self.q_table[next_state_hash][best_next_action]
        td_error = td_target - self.q_table[state_hash][action_index]

        self.q_table[state_hash][action_index] += self.learning_rate * td_error
        self.task_similarity.store_state(state, action)

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
        with open(filepath, "wb") as f:
            np.save(f, dict(self.q_table))

    def load_q_table(self, filepath):
        with open(filepath, "rb") as f:
            loaded_q_table = np.load(f, allow_pickle=True).item()
            self.q_table = defaultdict(lambda: np.zeros(len(self.env.valid_moves(self.player) or [0])), loaded_q_table)
