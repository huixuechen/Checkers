import numpy as np

class checkers_env:
    def __init__(self, board_size=6, player=1):
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
                board[row, col] = -1  # Player -1's pieces

        for row in range(self.board_size - rows, self.board_size):
            for col in range(row % 2, self.board_size, 2):
                board[row, col] = 1  # Player 1's pieces
        return board
    def reset(self):
        """
        Reset the board to its initial state.
        """
        self.board = self.initialize_board()
        self.player = 1

    def valid_moves(self, player):
        """
        Calculate all valid moves for a given player.
        """
        moves = []
        directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]

        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == player:  # 只检查属于当前玩家的棋子
                    for dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        # 检查普通移动
                        if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                            if self.board[new_row, new_col] == 0:
                                moves.append([row, col, new_row, new_col])
                        # 检查捕获移动
                        cap_row, cap_col = row + dr, col + dc
                        end_row, end_col = row + 2 * dr, col + 2 * dc
                        if (
                                0 <= cap_row < self.board_size and 0 <= cap_col < self.board_size and
                                0 <= end_row < self.board_size and 0 <= end_col < self.board_size and
                                self.board[cap_row, cap_col] == -player and
                                self.board[end_row, end_col] == 0
                        ):
                            moves.append([row, col, end_row, end_col])
        print(f"Valid moves for player {player}: {moves}")  # 调试输出
        return moves

    def capture_piece(self, action):
        """
        Capture a piece during a valid move.
        """
        start_row, start_col, end_row, end_col = action
        if abs(end_row - start_row) == 2:  # Capture move
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            self.board[mid_row, mid_col] = 0
            print(f"Captured piece at: ({mid_row}, {mid_col})")  # 调试输出



    def game_winner(self, board=None):
        """
        Determine the winner of the game.
        """
        board = board if board is not None else self.board
        player1_pieces = np.sum(board == 1)
        player2_pieces = np.sum(board == -1)

        if player1_pieces == 0:
            return -1  # Player -1 wins
        elif player2_pieces == 0:
            return 1  # Player 1 wins
        elif not self.valid_moves(1) and not self.valid_moves(-1):
            return 0  # Draw
        return None

    def step(self, action, player):
        """
        Execute a move and return the new board, reward, and game status.
        """
        start_row, start_col, end_row, end_col = action
        print(f"Player {player} executes move: {action}")

        self.board[end_row, end_col] = player
        self.board[start_row, start_col] = 0
        self.capture_piece(action)

        print("Board after move:")
        self.render()

        reward = 1 if abs(end_row - start_row) == 2 else 0  # Reward for capturing a piece
        done = self.game_winner(self.board) is not None
        return self.board, reward if not done else 10

    def render(self):
        """
        Render the current state of the board in the terminal.
        """
        for row in self.board:
            for square in row:
                if square == 1:
                    piece = "|0"
                elif square == -1:
                    piece = "|X"
                else:
                    piece = "| "
                print(piece, end='')
            print("|")
