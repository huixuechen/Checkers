import numpy as np

class TaskSimilarity:
    def __init__(self):
        self.state_memory = {}

    def state_to_hash(self, state):
        """Convert a board state into a unique hashable format."""
        return hash(tuple(state.flatten())) if state is not None else 0

    def store_state(self, state, action):
        """Store the best action for a given state."""
        state_hash = self.state_to_hash(state)
        self.state_memory[state_hash] = action

    def find_similar_state(self, state):
        """Find the most similar stored state."""
        state_hash = self.state_to_hash(state)
        if state_hash in self.state_memory:
            return state_hash
        return None

    def get_best_action(self, state_hash, valid_moves):
        """Retrieve the best action from a similar state if available."""
        if state_hash in self.state_memory:
            best_action = self.state_memory[state_hash]
            if best_action in valid_moves:
                return best_action
        return None
