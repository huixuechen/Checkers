import numpy as np
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(self, env, player, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0,
                 exploration_decay=0.995):
        self.env = env
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.1
        self.exploration_log = []
        self.q_table = defaultdict(lambda: np.zeros(len(self.env.valid_moves(self.player) or [0])))
        self.complexity_threshold = 0.5  # Threshold for task complexity

    def task_similarity(self, state, next_state, complexity):

        vector1 = np.asarray(state)
        vector2 = np.asarray(next_state)
        if complexity <= self.complexity_threshold:
         return 1 / (1 + np.linalg.norm(vector1 - vector2))  # Euclidean distance
        else:
         norm1 = np.linalg.norm(vector1)
         norm2 = np.linalg.norm(vector2)
        return np.dot(vector1, vector2) / (norm1 * norm2)  # Cosine similarity

    def choose_action(self, state, use_ucb=False):
        valid_moves = self.env.valid_moves(self.player)
        if not valid_moves:
            return None  # No valid moves available
        if random.uniform(0, 1) < self.exploration_rate:
            self.exploration_log.append("explore")
            return random.choice(valid_moves)
        elif use_ucb:
            self.exploration_log.append("ucb")
            return self.select_ucb_action(state, valid_moves)
        else:
            self.exploration_log.append("exploit")
            state_hash = self.state_to_hash(state)
            return valid_moves[np.argmax(self.q_table[state_hash])]


def learn(self, state, action, reward, next_state, task_complexity=None):
    valid_moves = self.env.valid_moves(self.player)
    if action not in valid_moves:
        return
    if action is None:
        return
    if task_complexity is not None:
        similarity_score = self.task_similarity(state, next_state, task_complexity)
        reward *= similarity_score
    state_hash = self.state_to_hash(state)
    next_state_hash = self.state_to_hash(next_state)
    action_index = valid_moves.index(action)
    best_next_action = np.argmax(self.q_table[next_state_hash])
    td_target = reward + self.discount_factor * self.q_table[next_state_hash][best_next_action]
    td_error = td_target - self.q_table[state_hash][action_index]

    if state_hash not in self.q_table:
        self.q_table[state_hash] = np.full(len(valid_moves), -1.0)
    self.q_table[state_hash][action_index] += self.learning_rate * td_error
    pass  # Prevent duplicate update
    if task_complexity is not None and task_complexity > self.complexity_threshold:
        self.q_table[state_hash][action_index] += similarity_score * self.learning_rate


def state_to_hash(self, state):
    return hash(tuple(state)) if state else 0


def update_exploration_rate(self):
    self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)


def select_ucb_action(self, state, valid_moves):
    state_hash = self.state_to_hash(state)
    ucb_values = []
    total_visits = sum(self.q_table[state_hash]) + 1
    total_visits = total_visits if total_visits > 0 else 1
    for action_index in range(len(valid_moves)):
        q_value = self.q_table[state_hash][action_index]
        ucb_value = q_value + 2 * np.sqrt(np.log(total_visits) / (1 + self.q_table[state_hash][action_index]))
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
