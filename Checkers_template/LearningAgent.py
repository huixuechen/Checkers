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

        # Set difficulty based on the provided difficulty level
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

    def choose_action(self, state, use_ucb=False):
        valid_moves = self.env.valid_moves(self.player)

        # 🔥 确保 AI 不能执行 invalid move
        print(f"AI Player {self.player} Valid Moves: {valid_moves}")

        if not valid_moves:
            print(f"AI Player {self.player} has no valid moves.")
            return None  # 没有合法移动

        state_hash = self.state_to_hash(state)
        self.visits[state_hash] += 1  # 记录访问次数

        # 🔥 确保 AI 只从 valid_moves 选择动作
        action = None

        # **1. 尝试从相似状态学习**
        similar_state = self.task_similarity.find_similar_state(state)
        if similar_state:
            best_action = self.task_similarity.get_best_action(similar_state, valid_moves)
            if best_action and best_action in valid_moves:  # 确保合法
                print(f"AI chooses from similar state: {best_action}")
                action = best_action

        # **2. 探索模式 (随机选择)**
        if action is None and random.uniform(0, 1) < self.exploration_rate:
            self.exploration_log.append("explore")
            action = random.choice(valid_moves)
            print(f"AI explores and picks: {action}")

        # **3. UCB 选择 (增强版)**
        if action is None and use_ucb:
            self.exploration_log.append("ucb")
            action = self.select_ucb_action(state, valid_moves)
            print(f"AI uses UCB and picks: {action}")

        # **4. 直接利用 Q 值**
        if action is None:
            self.exploration_log.append("exploit")
            q_values = self.q_table[state_hash]
            best_index = np.argmax(q_values[:len(valid_moves)])
            action = valid_moves[best_index]
            print(f"AI chooses best Q-value action: {action}")

        # 🔥 确保最终 action 一定在 valid_moves 里
        if action not in valid_moves:
            print(f"⚠️ AI picked an invalid move {action}, forcing a random valid move!")
            action = random.choice(valid_moves)

        return action

    def learn(self, state, action, reward, next_state):
        valid_moves = self.env.valid_moves(self.player)
        if not valid_moves or action not in valid_moves:
            return  # **确保 AI 只学习合法的动作**

        state_hash = self.state_to_hash(state)
        next_state_hash = self.state_to_hash(next_state)
        action_index = valid_moves.index(action)

        # **防止访问越界**
        if len(valid_moves) > 0 and state_hash not in self.q_table:
            self.q_table[state_hash] = np.full(len(valid_moves), -1.0)

        best_next_action = np.argmax(self.q_table[next_state_hash][:len(valid_moves)])  # **防止超出索引**
        td_target = reward + self.discount_factor * self.q_table[next_state_hash][best_next_action]
        td_error = td_target - self.q_table[state_hash][action_index]

        self.q_table[state_hash][action_index] += self.learning_rate * td_error
        self.task_similarity.store_state(state, action)  # Store learned state-action pairs

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

        return valid_moves[np.argmax(ucb_values)]  # **确保 UCB 选择的动作也是合法的**

    def save_q_table(self, filepath):
        with open(filepath, "wb") as f:
            np.save(f, dict(self.q_table))

    def load_q_table(self, filepath):
        with open(filepath, "rb") as f:
            loaded_q_table = np.load(f, allow_pickle=True).item()
            self.q_table = defaultdict(lambda: np.zeros(len(self.env.valid_moves(self.player) or [0])), loaded_q_table)
