import numpy as np
from collections import defaultdict

class TaskSimilarity:
    def __init__(self):
        self.state_memory = {}
        self.similarity_threshold = 0.9

    def state_to_hash(self, state):
        return hash(state.tobytes()) if state is not None else 0

    def store_state(self, state, action):
        state_hash = self.state_to_hash(state)
        state_bytes = state.tobytes()

        if state_hash in self.state_memory:
            self.state_memory[state_hash]['count'] += 1
        else:
            self.state_memory[state_hash] = {
                'action': action,
                'count': 1,
                'state': state_bytes
            }

    def find_similar_state(self, state):

        state_bytes = state.tobytes()
        state_hash = self.state_to_hash(state)

        if state_hash in self.state_memory:
            return state_hash

        for stored_hash, stored_data in self.state_memory.items():
            similarity = self.compute_state_similarity(state_bytes, stored_data['state'])
            if similarity > self.similarity_threshold:
                return stored_hash
        return None

    def compute_state_similarity(self, state_bytes, stored_bytes):
        stored_state = np.frombuffer(stored_bytes, dtype=np.int8)
        state = np.frombuffer(state_bytes, dtype=np.int8)

        if state.shape != stored_state.shape:
            return 0

        matching_cells = np.sum(state == stored_state)
        return matching_cells / state.size

    def get_best_action(self, state_hash, valid_moves):
        if state_hash in self.state_memory:
            best_action = self.state_memory[state_hash]['action']
            if best_action in valid_moves:
                return best_action
        return None
