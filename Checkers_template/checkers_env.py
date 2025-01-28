import numpy as np


class CheckersEnv:
    def __init__(self, board_size=8, player=1):
        """
        Initialize the environment.
        :param board_size: Size of the board (6 or 8)
        :param player: Current player (default is 1)
        """
        self.board_size = board_size  # Support 6x6 or 8x8 board
        self.board = self.initialize_board()
        self.player = player

    def initialize_board(self):
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        rows = (self.board_size // 2) - 1

        for row in range(rows):
            for col in range(row % 2, self.board_size, 2):
                board[row, col] = 2  # Player 2's pieces

        for row in range(self.board_size - rows, self.board_size):
            for col in range(row % 2, self.board_size, 2):
                board[row, col] = 1  # Player 1's pieces
        return board

    def reset(self):
        self.board = self.initialize_board()
        self.player = 1

    def valid_moves(self, player):
        moves = []
        directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board[row, col]
                if piece == player or piece == player * 2:  # 普通棋子或王棋
                    piece_directions = directions
                    if piece == 3 or piece == 4:  # 3 和 4 是王棋
                        piece_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # 王棋可以四个方向移动

                    for dr, dc in piece_directions:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                            if self.board[new_row, new_col] == 0:
                                moves.append([row, col, new_row, new_col])

                        cap_row, cap_col = row + dr, col + dc
                        end_row, end_col = row + 2 * dr, col + 2 * dc
                        if (
                                0 <= cap_row < self.board_size and 0 <= cap_col < self.board_size and
                                0 <= end_row < self.board_size and 0 <= end_col < self.board_size and
                                self.board[cap_row, cap_col] == (3 - player) and
                                self.board[end_row, end_col] == 0
                        ):
                            moves.append([row, col, end_row, end_col])
        return moves

    def capture_piece(self, action):
        start_row, start_col, end_row, end_col = action
        if abs(end_row - start_row) == 2:  # Capture move
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            self.board[mid_row, mid_col] = 0

    def promote_to_king(self):
        for col in range(self.board_size):
            if self.board[0, col] == 1:  # Player 1 reaches the back row
                self.board[0, col] = 3  # Promote to player 1's king
            if self.board[self.board_size - 1, col] == 2:  # Player 2 reaches the back row
                self.board[self.board_size - 1, col] = 4  # Promote to player 2's king

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
                    self.board[mid_row, mid_col] == (3 - player) and
                    self.board[end_row, end_col] == 0
            ):
                additional_moves.append([row, col, end_row, end_col])
        return additional_moves

    def step(self, action, player):
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
        player1_pieces = np.sum(self.board == 1) + np.sum(self.board == 3)
        player2_pieces = np.sum(self.board == 2) + np.sum(self.board == 4)

        if player1_pieces == 0:
            return 2  # Player 2 wins
        elif player2_pieces == 0:
            return 1  # Player 1 wins
        elif not self.valid_moves(1) and not self.valid_moves(2):
            return 0  # Draw
        elif not self.valid_moves(1):
            return 2  # Player 1 has no moves
        elif not self.valid_moves(2):
            return 1  # Player 2 has no moves
        return None

    def handle_multiple_jumps(self, row, col, player):
        """
        递归处理连跳吃子
        """
        additional_moves = self.get_additional_jumps(row, col, player)
        if not additional_moves:
            return

        for move in additional_moves:
            end_row, end_col = move[2], move[3]
            self.board[end_row, end_col] = self.board[row, col]
            self.board[row, col] = 0
            self.capture_piece(move)
            self.handle_multiple_jumps(end_row, end_col, player)  # 递归处理连跳

    def is_dangerous_action(self, action):
        """
        检查这个动作是否会让对方棋子可以立即吃掉当前棋子。
        """
        start_row, start_col, end_row, end_col = action
        self.board[end_row, end_col] = self.board[start_row, start_col]
        self.board[start_row, start_col] = 0

        opponent = 1 if self.player == 2 else 2
        opponent_moves = self.valid_moves(opponent)

        # 还原棋盘状态
        self.board[start_row, start_col] = self.board[end_row, end_col]
        self.board[end_row, end_col] = 0

        return any(abs(move[2] - move[0]) == 2 for move in opponent_moves)
