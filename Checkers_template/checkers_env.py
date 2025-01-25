import numpy as np

class CheckersEnv:
    def __init__(self, board_size=8, player=1):
        """
        Initialize the checkers environment.
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
                if piece == player or piece == player * 2:  # Include kings
                    piece_directions = directions
                    if abs(piece) == 2:
                        piece_directions = directions + [(d[0] * -1, d[1] * -1) for d in directions]

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
                                self.board[cap_row, cap_col] == 3 - player and
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
                self.board[0, col] = 2  # Promote to player 1's king
            if self.board[self.board_size - 1, col] == 2:  # Player 2 reaches the back row
                self.board[self.board_size - 1, col] = -2  # Promote to player 2's king

    def game_winner(self):
        player1_pieces = np.sum(self.board == 1) + np.sum(self.board == 2)
        player2_pieces = np.sum(self.board == 2) + np.sum(self.board == -2)

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

    def step(self, action, player):
        start_row, start_col, end_row, end_col = action
        self.board[end_row, end_col] = self.board[start_row, start_col]
        self.board[start_row, start_col] = 0
        self.capture_piece(action)  # Handle capture
        self.promote_to_king()

        reward = 1 if abs(end_row - start_row) == 2 else 0
        winner = self.game_winner()
        done = winner is not None
        if done:
            reward = 10 if winner == self.player else -10
        self.player *= -1  # Switch turns
        return self.board, reward, done

    def render(self):
        for row in self.board:
            for square in row:
                if square == 1:
                    piece = "|P"
                elif square == -1:
                    piece = "|O"
                elif square == 2:
                    piece = "|K"
                elif square == -2:
                    piece = "|Q"
                else:
                    piece = "| "
                print(piece, end="")
            print("|")
