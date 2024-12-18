from django.contrib.admin import action

import checkers_env
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

class LearningAgent:

    def __init__(self, step_size, epsilon, env):
        '''
        :param step_size:
        :param epsilon:
        :param env:
        '''

        self.step_size = step_size
        self.epsilon = epsilon
        self.env = env
        self.q_table = np.zeros(len(env.state_space), len(env.action_space))


    def evaluation(self):
        '''
        evaluate the score of the board, i.e., reward function
        '''
        board = self.env.board
        player1_pieces = np.sum(board == 1)
        player2_pieces = np.sum(board == -1)
        return player1_pieces - player2_pieces

    def learning(self, state, action, reward, next_state, gamma):
        '''
        Q-learning update rule.
        '''
        start_row, start_col, end_row, end_col = action
        best_next_action = np.max(self.q_table[next_state])
        self.q_table[start_row, start_col, end_row, end_col] += self.step_size * (
                reward + gamma * best_next_action - self.q_table[start_row, start_col, end_row, end_col]
        )

    def select_action(self, valid_actions):
        '''
        make the movement decision
        '''
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)  # Exploration
        else:
            best_action = max(valid_actions, key=lambda a: self.q_table[a[0], a[1], a[2], a[3]])
            return best_action








