import numpy as np

class CheckersEnv:
    def __init__(self, board_size=8, player=1):
        self.board_size = board_size  # 6x6 或 8x8 棋盘
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
        """返回当前玩家所有合法移动"""
        moves = []
        jump_moves = []
        forward_directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        king_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # **王棋可以四向移动**

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board[row, col]

                if piece == player or piece == player + 2:  # **普通棋子 或 王棋**
                    piece_directions = king_directions if piece in [3, 4] else forward_directions

                    # **检查普通移动**
                    for dr, dc in piece_directions:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                            if self.board[new_row, new_col] == 0:
                                moves.append([row, col, new_row, new_col])

                    # **检查吃子 (普通棋子必须严格跳±2)**
                    for dr, dc in piece_directions:
                        cap_row, cap_col = row + dr, col + dc  # 被跳过的棋子
                        end_row, end_col = row + 2 * dr, col + 2 * dc  # 落点

                        # **确保普通棋子只能吃相邻棋子，且坐标变化必须是 (±2, ±2)**
                        if (
                                0 <= cap_row < self.board_size and 0 <= cap_col < self.board_size and
                                0 <= end_row < self.board_size and 0 <= end_col < self.board_size and
                                self.board[cap_row, cap_col] in [3 - player, (3 - player) + 2] and  # **确保中间有对方棋子**
                                self.board[end_row, end_col] == 0 and  # **落点必须为空**
                                (end_row - row, end_col - col) in [(2, 2), (2, -2), (-2, 2), (-2, -2)]  # **普通棋子严格限制跳跃**
                        ):
                            jump_moves.append([row, col, end_row, end_col])

                    # **王棋允许长距离跳跃吃子**
                    if piece in [3, 4]:
                        for dr, dc in king_directions:
                            temp_row, temp_col = row + dr, col + dc
                            captured = False  # **确保王棋必须跳过棋子**
                            while 0 <= temp_row < self.board_size and 0 <= temp_col < self.board_size:
                                if self.board[temp_row, temp_col] in [3 - player, (3 - player) + 2]:
                                    captured = True  # **标记王棋吃子**
                                elif self.board[temp_row, temp_col] == 0 and captured:
                                    jump_moves.append([row, col, temp_row, temp_col])
                                else:
                                    break
                                temp_row += dr
                                temp_col += dc

        return jump_moves if jump_moves else moves  # **强制吃子规则**

    def capture_piece(self, action, player):
        """确保吃子时必须跳过相邻的对方棋子"""
        start_row, start_col, end_row, end_col = action

        if (end_row - start_row, end_col - start_col) in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:  # **严格限制普通棋子**
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2

            if self.board[mid_row, mid_col] in [3 - player, (3 - player) + 2]:  # **确保跳过的是对方棋子**
                self.board[mid_row, mid_col] = 0  # **移除被吃掉的棋子**

    def promote_to_king(self):
        """达到对方底线后封王"""
        for col in range(self.board_size):
            if self.board[0, col] == 1:  # 玩家 1 到达底线
                self.board[0, col] = 3  # **升级为王棋**
            if self.board[self.board_size - 1, col] == 2:  # 玩家 2 到达底线
                self.board[self.board_size - 1, col] = 4  # **升级为王棋**

    def step(self, action, player):
        """执行一步棋，并返回新状态"""
        start_row, start_col, end_row, end_col = action
        self.board[end_row, end_col] = self.board[start_row, start_col]
        self.board[start_row, start_col] = 0
        self.capture_piece(action, player)  # **修正：确保正确的吃子逻辑**

        # **修正：确保连跳生效**
        self.handle_multiple_jumps(end_row, end_col, player)

        self.promote_to_king()

        reward = 1 if abs(end_row - start_row) == 2 else 0
        winner = self.game_winner()
        done = winner is not None

        if done:
            reward = 10 if winner == player else -10  # **正确给予奖励**
        else:
            self.player = 1 if player == 2 else 2  # **正确切换玩家**

        return self.board.copy(), reward, done

    def handle_multiple_jumps(self, row, col, player):
        """递归处理连跳吃子"""
        additional_moves = self.get_additional_jumps(row, col, player)

        # **确保所有可能的跳跃都被执行**
        for move in additional_moves:
            end_row, end_col = move[2], move[3]
            self.board[end_row, end_col] = self.board[row, col]
            self.board[row, col] = 0
            self.capture_piece(move, player)
            self.handle_multiple_jumps(end_row, end_col, player)  # **递归处理连跳**

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
                self.board[mid_row, mid_col] in [3 - player, (3 - player) + 2] and
                self.board[end_row, end_col] == 0
            ):
                additional_moves.append([row, col, end_row, end_col])
        return additional_moves

    def game_winner(self):
        """检查是否有玩家获胜"""
        player1_pieces = np.sum(self.board == 1) + np.sum(self.board == 3)
        player2_pieces = np.sum(self.board == 2) + np.sum(self.board == 4)

        if player1_pieces == 0:
            return 2
        elif player2_pieces == 0:
            return 1
        elif not self.valid_moves(1) and not self.valid_moves(2):
            return 0
        elif not self.valid_moves(1):
            return 2
        elif not self.valid_moves(2):
            return 1
        return None
