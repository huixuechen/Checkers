import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, player, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = defaultdict(lambda: np.zeros(len(self.env.valid_moves(self.player))))

    def choose_action(self, state):
        valid_moves = self.env.valid_moves(self.player)
        if not valid_moves:
            return None  # No valid moves available
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(valid_moves)
        else:
            state_str = self.state_to_string(state)
            return valid_moves[np.argmax(self.q_table[state_str])]

    def learn(self, state, action, reward, next_state):
        if action is None:
            return  # No learning if no action was taken
        state_str = self.state_to_string(state)
        next_state_str = self.state_to_string(next_state)
        action_index = self.env.valid_moves(self.player).index(action)

        best_next_action = np.argmax(self.q_table[next_state_str])
        td_target = reward + self.discount_factor * self.q_table[next_state_str][best_next_action]
        td_error = td_target - self.q_table[state_str][action_index]

        self.q_table[state_str][action_index] += self.learning_rate * td_error

    def state_to_string(self, state):
        return str(state)

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay