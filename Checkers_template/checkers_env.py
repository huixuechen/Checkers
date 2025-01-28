import numpy as np


class CheckersEnv:
    def __init__(self, board_size=8, player=1):
        """
        Initialize the environment.
        :param board_size: Size of the board (6 or 8)
        :param player: Current player (default is 1)
        """
        self.board_size = board_size  # 支持 6x6 或 8x8 棋盘
        self.board = self.initialize_board()
        self.player = player

    def initialize_board(self):
        """初始化棋盘"""
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        rows = (self.board_size // 2) - 1

        for row in range(rows):
            for col in range(row % 2, self.board_size, 2):
                board[row, col] = 2  # 玩家 2 的棋子

        for row in range(self.board_size - rows, self.board_size):
            for col in range(row % 2, self.board_size, 2):
                board[row, col] = 1  # 玩家 1 的棋子
        return board

    def reset(self):
        """重置棋盘"""
        self.board = self.initialize_board()
        self.player = 1

    def valid_moves(self, player):
        """返回当前玩家所有合法移动，包括普通移动和吃子"""
        moves = []
        jump_moves = []
        forward_directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        king_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board[row, col]
                if piece == player or piece == player + 2:  # 普通棋子 或 王棋
                    piece_directions = king_directions if piece > 2 else forward_directions

                    for dr, dc in piece_directions:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                            if self.board[new_row, new_col] == 0:
                                moves.append([row, col, new_row, new_col])

                        cap_row, cap_col = row + dr, col + dc
                        end_row, end_col = row + 2 * dr, col + 2 * dc

                        # 只有跳过对方棋子，且落点为空时，才能吃子
                        if (
                                0 <= cap_row < self.board_size and 0 <= cap_col < self.board_size and
                                0 <= end_row < self.board_size and 0 <= end_col < self.board_size and
                                self.board[cap_row, cap_col] in [3 - player, (3 - player) + 2] and  # 确保中间是对方棋子
                                self.board[end_row, end_col] == 0 and  # 落点必须为空
                                abs(cap_row - row) == 1 and abs(cap_col - col) == 1  # 必须相邻，防止跨空位跳跃
                        ):
                            jump_moves.append([row, col, end_row, end_col])

        return jump_moves if jump_moves else moves  # 强制吃子规则

    def capture_piece(self, action):
        """确保吃子时必须跳过相邻的对方棋子"""
        start_row, start_col, end_row, end_col = action

        # 只有斜向跳跃两格才算吃子，确保棋子相邻
        if abs(end_row - start_row) == 2 and abs(end_col - start_col) == 2:
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2

            # 确保中间位置是对方棋子，并且必须紧邻当前棋子
            if (
                    self.board[mid_row, mid_col] in [3 - self.player, (3 - self.player) + 2] and  # 确保中间是对方棋子
                    abs(mid_row - start_row) == 1 and abs(mid_col - start_col) == 1  # 确保是相邻棋子
            ):
                self.board[mid_row, mid_col] = 0  # 移除被吃掉的棋子

    def promote_to_king(self):
        """达到对方底线后封王"""
        for col in range(self.board_size):
            if self.board[0, col] == 1:  # 玩家 1 到达底线
                self.board[0, col] = 3  # 升级为王棋
            if self.board[self.board_size - 1, col] == 2:  # 玩家 2 到达底线
                self.board[self.board_size - 1, col] = 4  # 升级为王棋

    def get_additional_jumps(self, row, col, player):
        """检查是否可以继续跳跃吃子"""
        additional_moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            mid_row, mid_col = row + dr, col + dc
            end_row, end_col = row + 2 * dr, col + 2 * dc
            if (
                    0 <= mid_row < self.board_size and 0 <= mid_col < self.board_size and
                    0 <= end_row < self.board_size and 0 <= end_col < self.board_size and
                    self.board[mid_row, mid_col] in [3 - player, (3 - player) + 2] and  # 确保中间是对方棋子
                    self.board[end_row, end_col] == 0  # 落点必须为空
            ):
                additional_moves.append([row, col, end_row, end_col])
        return additional_moves

    def step(self, action, player):
        """执行一步棋，并返回新状态"""
        start_row, start_col, end_row, end_col = action
        self.board[end_row, end_col] = self.board[start_row, start_col]
        self.board[start_row, start_col] = 0
        self.capture_piece(action)  # 处理吃子

        # 处理连跳
        self.handle_multiple_jumps(end_row, end_col, player)

        self.promote_to_king()

        reward = 1 if abs(end_row - start_row) == 2 else 0
        winner = self.game_winner()
        done = winner is not None

        if done:
            reward = 10 if winner == player else -10  # 正确给予奖励
        else:
            self.player = 1 if player == 2 else 2  # 正确切换玩家

        return self.board.copy(), reward, done

    def game_winner(self):
        """检查是否有玩家获胜"""
        player1_pieces = np.sum(self.board == 1) + np.sum(self.board == 3)
        player2_pieces = np.sum(self.board == 2) + np.sum(self.board == 4)

        if player1_pieces == 0:
            return 2  # 玩家 2 胜利
        elif player2_pieces == 0:
            return 1  # 玩家 1 胜利
        elif not self.valid_moves(1) and not self.valid_moves(2):
            return 0  # 平局
        elif not self.valid_moves(1):
            return 2  # 玩家 1 无法移动，玩家 2 胜利
        elif not self.valid_moves(2):
            return 1  # 玩家 2 无法移动，玩家 1 胜利
        return None

    def handle_multiple_jumps(self, row, col, player):
        """递归处理连跳吃子"""
        additional_moves = self.get_additional_jumps(row, col, player)
        while additional_moves:
            move = additional_moves[0]
            end_row, end_col = move[2], move[3]
            self.board[end_row, end_col] = self.board[row, col]
            self.board[row, col] = 0
            self.capture_piece(move)
            row, col = end_row, end_col
            additional_moves = self.get_additional_jumps(row, col, player)
