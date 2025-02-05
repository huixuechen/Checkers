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
        self.set_difficulty(self.difficulty)

    def set_difficulty(self, difficulty):
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.learning_rate = 0.15  # Increase learning speed
            self.discount_factor = 0.8  # Make AI think long-term
            self.exploration_rate = 0.3  # Reduce randomness
            self.exploration_decay = 0.999  # Slow decay for steady learning
            self.attack_bonus = 15

        elif difficulty == "medium":
            self.learning_rate = 0.2
            self.discount_factor = 0.9
            self.exploration_rate = 0.1
            self.exploration_decay = 0.9995  # Even slower decay
            self.attack_bonus = 25

        elif difficulty == "hard":
            self.learning_rate = 0.3
            self.discount_factor = 0.95
            self.exploration_rate = 0.05  # Almost fully deterministic
            self.exploration_decay = 0.9997
            self.attack_bonus = 35
        else:
            raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', or 'hard'.")

        self.min_exploration_rate = 0.1

        self.q_table_file = f"q_table_{difficulty}.json"


        self.load_q_table(self.q_table_file)



    def choose_action(self, state, use_ucb=False):
        valid_moves = self.env.valid_moves(self.player)

        if not valid_moves:
            return None

        state_hash = self.task_similarity.find_similar_state(state)
        if state_hash is not None:
            best_action = self.task_similarity.get_best_action(state_hash, valid_moves)
            if best_action:
                return best_action

        state_hash = self.state_to_hash(state)
        self.visits[state_hash] += 1


        jump_moves = [move for move in valid_moves if abs(move[2] - move[0]) == 2]
        normal_moves = [move for move in valid_moves if abs(move[2] - move[0]) == 1]

        action = None

        if jump_moves:
            print(f"⚠️ AI Player {self.player} must jump: {jump_moves}")
            action = random.choice(jump_moves)

        else:
            if normal_moves:
                print(f"✅ AI Player {self.player} performs normal move: {normal_moves}")
                action = random.choice(normal_moves)

            else:
                return None

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

        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.full(len(valid_moves), -1.0)

        if next_state_hash not in self.q_table:
            self.q_table[next_state_hash] = np.zeros(len(valid_moves))

        try:
            action_index = valid_moves.index(action)
        except ValueError:
            print(f"⚠️ Action {action} not found in valid_moves: {valid_moves}")
            return

        if len(self.q_table[next_state_hash]) > 0:
            best_next_action = np.argmax(self.q_table[next_state_hash])
        else:
            best_next_action = 0


        td_target = reward + self.discount_factor * self.q_table[next_state_hash][best_next_action]
        td_error = td_target - self.q_table[state_hash][action_index]
        self.q_table[state_hash][action_index] += self.learning_rate * td_error  # **更新 Q 值**

        print(
            f"Updated Q-table: {state_hash} | Action {action} | New Q-value: {self.q_table[state_hash][action_index]}")

        try:
            self.task_similarity.store_state(state, action)
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
            json.dump(q_table_dict, f, indent=4)
        print(f"✅ Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        if not os.path.exists(filepath):
            print(f"⚠️ Q-table file not found ({filepath}), creating a new empty Q-table.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.env.valid_moves(self.player) or [0])))
            self.save_q_table(filepath)
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
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
