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
        if not valid_moves:
            return None  # No valid moves available

        state_hash = self.state_to_hash(state)
        self.visits[state_hash] += 1  # Track state visits

        similar_state = self.task_similarity.find_similar_state(state)
        if similar_state:
            best_action = self.task_similarity.get_best_action(similar_state, valid_moves)
            if best_action:
                return best_action

        if random.uniform(0, 1) < self.exploration_rate:
            self.exploration_log.append("explore")
            return random.choice(valid_moves)
        elif use_ucb:
            self.exploration_log.append("ucb")
            return self.select_ucb_action(state, valid_moves)
        else:
            self.exploration_log.append("exploit")
            return valid_moves[np.argmax(self.q_table[state_hash])]

    def learn(self, state, action, reward, next_state):
        valid_moves = self.env.valid_moves(self.player)
        if not valid_moves or action not in valid_moves:
            return

        state_hash = self.state_to_hash(state)
        next_state_hash = self.state_to_hash(next_state)
        action_index = valid_moves.index(action)
        best_next_action = np.argmax(self.q_table[next_state_hash])

        td_target = reward + self.discount_factor * self.q_table[next_state_hash][best_next_action]
        td_error = td_target - self.q_table[state_hash][action_index]

        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.full(len(valid_moves), -1.0)

        self.q_table[state_hash][action_index] += self.learning_rate * td_error
        self.task_similarity.store_state(state, action)  # Store learned state-action pairs

    def state_to_hash(self, state):
        return hash(tuple(state.flatten())) if state is not None else 0

    def update_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def select_ucb_action(self, state, valid_moves):
        state_hash = self.state_to_hash(state)
        ucb_values = []
        total_visits = sum(self.visits.values()) + 1
        for action_index in range(len(valid_moves)):
            q_value = self.q_table[state_hash][action_index]
            ucb_value = q_value + 2 * np.sqrt(np.log(total_visits) / (1 + self.visits[state_hash]))
            ucb_values.append(ucb_value)
        return valid_moves[np.argmax(ucb_values)]

    def save_q_table(self, filepath):
        with open(filepath, "wb") as f:
            np.save(f, dict(self.q_table))

    def load_q_table(self, filepath):
        with open(filepath, "rb") as f:
            self.q_table = defaultdict(
                lambda: np.zeros(len(self.env.valid_moves(self.player) or [0])),
                np.load(f, allow_pickle=True),
            )
