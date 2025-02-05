import numpy as np
from collections import defaultdict

class TaskSimilarity:
    def __init__(self):
        self.state_memory = {}
        self.similarity_threshold = 0.85  # Dynamically adjustable similarity threshold
        self.zobrist_table = np.random.randint(1, 2**32, size=(8, 8, 5), dtype=np.uint32)  # Zobrist hash table

    def zobrist_hash(self, state):
        """ Compute Zobrist hash value """
        hash_value = 0
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                piece = state[i, j]
                if piece != 0:  # Only compute hash for non-empty squares
                    hash_value ^= self.zobrist_table[i, j, piece]
        return hash_value

    def store_state(self, state, action):
        """ Store the board state and its best action """
        state_hash = self.zobrist_hash(state)
        state_bytes = state.tobytes()

        if state_hash not in self.state_memory:
            self.state_memory[state_hash] = {
                'actions': defaultdict(int),  # Store multiple actions and their visit counts
                'count': 1,
                'state': state_bytes
            }
        else:
            self.state_memory[state_hash]['count'] += 1
        self.state_memory[state_hash]['actions'][tuple(action)] += 1

    def find_similar_state(self, state):
        """ Find if there is a similar board state """
        state_bytes = state.tobytes()
        state_hash = self.zobrist_hash(state)

        if state_hash in self.state_memory:
            return state_hash

        for stored_hash, stored_data in self.state_memory.items():
            similarity = self.compute_state_similarity(state_bytes, stored_data['state'])
            if similarity > self.similarity_threshold:
                return stored_hash
        return None

    def compute_state_similarity(self, state_bytes, stored_bytes):
        """ Compute state similarity, giving higher weight to king pieces """
        stored_state = np.frombuffer(stored_bytes, dtype=np.int8).reshape(-1)
        state = np.frombuffer(state_bytes, dtype=np.int8).reshape(-1)

        if state.shape != stored_state.shape:
            return 0

        # Weight strategy: Kings (x1.5), Normal pieces (x1.2), Empty spaces (x1.0)
        weights = np.where(state == 0, 1.0, np.where((state == 3) | (state == 4), 1.5, 1.2))
        similarity = np.sum((state == stored_state) * weights) / np.sum(weights)

        return similarity

    def get_best_action(self, state_hash, valid_moves):
        """ Select the most frequently chosen best action from similar states """
        if state_hash in self.state_memory:
            action_counts = self.state_memory[state_hash]['actions']
            best_action = max(action_counts, key=action_counts.get)  # Choose the most frequently selected action
            if list(best_action) in valid_moves:
                return list(best_action)
        return None

    def adjust_similarity_threshold(self, win_rate):
        """ Dynamically adjust similarity threshold based on AI win rate """
        self.similarity_threshold = min(0.95, max(0.7, win_rate))
