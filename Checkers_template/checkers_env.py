import numpy as np

class CheckersEnv:
    def __init__(self, board_size=8, player=1):
        self.has_moved = False
        self.must_jump = False
        self.board_size = board_size
        self.board = self.initialize_board()
        self.player = player
        self.last_move_was_jump = False

    def initialize_board(self):
        """初始化棋盘"""
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        rows = (self.board_size // 2) - 1

        for row in range(rows):
            for col in range(row % 2, self.board_size, 2):
                board[row, col] = 2

        for row in range(self.board_size - rows, self.board_size):
            for col in range(row % 2, self.board_size, 2):
                board[row, col] = 1
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
        king_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board[row, col]

                if piece == player or piece == player + 2:
                    piece_directions = king_directions if piece in [3, 4] else forward_directions


                    for dr, dc in piece_directions:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                            if self.board[new_row, new_col] == 0:
                                moves.append([row, col, new_row, new_col])


                    for dr, dc in piece_directions:
                        cap_row, cap_col = row + dr, col + dc
                        end_row, end_col = row + 2 * dr, col + 2 * dc

                        if (
                                0 <= cap_row < self.board_size and 0 <= cap_col < self.board_size and
                                0 <= end_row < self.board_size and 0 <= end_col < self.board_size and
                                self.board[cap_row, cap_col] in [3 - player, (3 - player) + 2] and  # **确保中间有对方棋子**
                                self.board[end_row, end_col] == 0 and  # **落点必须为空**
                                (end_row - row, end_col - col) in [(2, 2), (2, -2), (-2, 2), (-2, -2)]  # **普通棋子严格限制跳跃**
                        ):
                            jump_moves.append([row, col, end_row, end_col])


                    if piece in [3, 4]:
                        for dr, dc in king_directions:
                            temp_row, temp_col = row + dr, col + dc
                            captured = False
                            while 0 <= temp_row < self.board_size and 0 <= temp_col < self.board_size:
                                if self.board[temp_row, temp_col] in [3 - player, (3 - player) + 2]:
                                    captured = True
                                elif self.board[temp_row, temp_col] == 0 and captured:
                                    jump_moves.append([row, col, temp_row, temp_col])
                                else:
                                    break
                                temp_row += dr
                                temp_col += dc
            if moves and not jump_moves:
                self.has_moved = True

        return jump_moves if jump_moves else moves  # **强制吃子规则**

    def capture_piece(self, action, player):
        start_row, start_col, end_row, end_col = action

        if (end_row - start_row, end_col - start_col) in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2

            if self.board[mid_row, mid_col] in [3 - player, (3 - player) + 2]:
                self.board[mid_row, mid_col] = 0

    def step(self, action, player):
        """Execute a move and return the new state, shaped rewards, and game status"""
        start_row, start_col, end_row, end_col = action
        self.board[end_row, end_col] = self.board[start_row, start_col]
        self.board[start_row, start_col] = 0

        is_jump = abs(end_row - start_row) == 2  # Check if it's a jump

        reward = 0  # Initialize reward

        # Handle piece capture and reward for jumps
        if is_jump:
            self.capture_piece(action, player)  # Remove the captured piece
            reward += 2  # Reward for capturing a piece

            additional_jumps = self.get_additional_jumps(end_row, end_col, player)
            if additional_jumps:
                self.must_jump = True
                return self.board.copy(), reward, False  # Player must continue turn
            else:
                self.must_jump = False
        else:
            if self.must_jump:
                # If a jump was possible earlier, normal moves are not allowed
                return self.board.copy(), -1, False

            self.must_jump = False

        # Handle king promotion and reward
        king_promotion = self.promote_to_king()
        if king_promotion:
            reward += 5  # Reward for king promotion

        # Check for game end
        winner = self.game_winner()
        done = winner is not None

        if done:
            reward += 10 if winner == player else -10  # Endgame reward

        # Switch players
        if not done:
            self.player = 1 if player == 2 else 2

        return self.board.copy(), reward, done

    def promote_to_king(self):
        """Promote pieces to king and return if promotion happened"""
        promotion_happened = False
        for col in range(self.board_size):
            if self.board[0, col] == 1:  # Player 1 promotion
                self.board[0, col] = 3
                promotion_happened = True
            if self.board[self.board_size - 1, col] == 2:  # Player 2 promotion
                self.board[self.board_size - 1, col] = 4
                promotion_happened = True
        return promotion_happened

    def handle_multiple_jumps(self, row, col, player):

        additional_moves = self.get_additional_jumps(row, col, player)

        # **确保所有可能的跳跃都被执行**
        for move in additional_moves:
            end_row, end_col = move[2], move[3]
            self.board[end_row, end_col] = self.board[row, col]
            self.board[row, col] = 0
            self.capture_piece(move, player)
            self.handle_multiple_jumps(end_row, end_col, player)  # **递归处理连跳**

    def get_additional_jumps(self, row, col, player):

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
