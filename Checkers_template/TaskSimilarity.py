import numpy as np
from collections import defaultdict

class TaskSimilarity:
    def __init__(self):
        self.state_memory = {}  # 存储状态 -> (最佳动作, 访问计数)
        self.similarity_threshold = 0.9  # 设定相似度阈值

    def state_to_hash(self, state):
        """将棋盘状态转换为哈希格式"""
        return hash(state.tobytes()) if state is not None else 0

    def store_state(self, state, action):
        """存储某个状态的最佳行动"""
        state_hash = self.state_to_hash(state)
        if state_hash in self.state_memory:
            self.state_memory[state_hash]['count'] += 1  # 增加访问次数
        else:
            self.state_memory[state_hash] = {'action': action, 'count': 1}  # 初始化存储

    def find_similar_state(self, state):
        """查找最相近的已存储状态"""
        state_hash = self.state_to_hash(state)
        if state_hash in self.state_memory:
            return state_hash  # 直接返回匹配的哈希值

        # 尝试找到相似的状态（这里可以使用更复杂的搜索算法）
        for stored_hash in self.state_memory:
            similarity = self.compute_state_similarity(state, stored_hash)
            if similarity > self.similarity_threshold:
                return stored_hash
        return None

    def compute_state_similarity(self, state, stored_hash):
        """计算状态相似性（简单实现）"""
        stored_state = np.frombuffer(stored_hash, dtype=state.dtype).reshape(state.shape)
        matching_cells = np.sum(state == stored_state)
        return matching_cells / state.size  # 计算相似度比例

    def get_best_action(self, state_hash, valid_moves):
        """获取最优动作（如果存在）"""
        if state_hash in self.state_memory:
            best_action = self.state_memory[state_hash]['action']
            if best_action in valid_moves:
                return best_action  # 只返回有效动作
        return None
