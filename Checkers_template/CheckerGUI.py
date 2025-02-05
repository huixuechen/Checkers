import tkinter as tk
from tkinter import messagebox
import numpy as np
from checkers_env import CheckersEnv
from LearningAgent import QLearningAgent

class CheckerGUI:
    def __init__(self, root, difficulty='easy'):
        self.root = root
        self.root.title("Checkers")
        self.difficulty = difficulty
        self.board_size = 6 if difficulty == 'easy' else 8
        self.env = CheckersEnv(board_size=self.board_size)
        self.canvas_size = 500
        self.cell_size = self.canvas_size // self.board_size
        self.current_player = 1
        self.selected_piece = None
        self.valid_destinations = []
        self.history = []
        self.agent = self.create_agent()


        self.setup_ui()
        self.render_board()

    def create_agent(self):
        return QLearningAgent(self.env, player=2, difficulty=self.difficulty)

    def setup_ui(self):

        main_frame = tk.Frame(self.root, bg="#D0E4C8")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame, bg="#D0E4C8", relief=tk.GROOVE, bd=2)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        tk.Label(control_frame, text="Checkers", font=("Arial", 12, "bold"), bg="#D0E4C8").pack(pady=5)
        tk.Label(control_frame, text="Difficulty", font=("Arial", 12, "bold"), bg="#D0E4C8").pack(pady=5)
        tk.Button(control_frame, text="Easy", command=lambda: self.set_difficulty('easy'), width=15).pack(pady=5)
        tk.Button(control_frame, text="Medium", command=lambda: self.set_difficulty('medium'), width=15).pack(pady=5)
        tk.Button(control_frame, text="Hard", command=lambda: self.set_difficulty('hard'), width=15).pack(pady=5)

        tk.Button(control_frame, text="Reset Game", command=self.reset_game, width=15).pack(pady=10)
        tk.Button(control_frame, text="Undo Last Move", command=self.regret_move, width=15).pack(pady=5)

        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, bg="#F0F5F1", relief=tk.SUNKEN, bd=2)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<ButtonPress-1>", self.on_piece_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_piece_release)

    def on_piece_press(self, event):
        """处理棋子按下事件，并高亮合法移动位置"""
        col, row = event.x // self.cell_size, event.y // self.cell_size
        piece = self.env.board[row, col]

        if piece in [1, 2, 3, 4]:
            self.selected_piece = (row, col)

            all_valid_moves = self.env.valid_moves(self.current_player)

            jump_moves = [move for move in all_valid_moves if abs(move[2] - move[0]) == 2]  # 普通吃子
            king_jump_moves = [move for move in all_valid_moves if
                               piece in [3, 4] and abs(move[2] - move[0]) > 2]  # 王棋远跳

            if jump_moves:
                self.valid_destinations = [move[2:4] for move in jump_moves if move[:2] == [row, col]]
            elif king_jump_moves:
                self.valid_destinations = [move[2:4] for move in king_jump_moves if move[:2] == [row, col]]
            else:
                self.valid_destinations = [move[2:4] for move in all_valid_moves if move[:2] == [row, col]]

            print(f"Valid moves for player {self.current_player} from ({row}, {col}): {self.valid_destinations}")

            self.render_board()

    def on_piece_release(self, event):
        """Handle piece release event and execute the move"""

        if self.selected_piece:
            start_row, start_col = self.selected_piece
            end_row, end_col = event.y // self.cell_size, event.x // self.cell_size
            action = [start_row, start_col, end_row, end_col]

            valid_moves = self.env.valid_moves(self.current_player)

            if action in valid_moves:
                self.history.append((self.env.board.copy(), self.current_player))
                self.env.step(action, self.current_player)

                # Check if the move was a capture
                if abs(end_row - start_row) == 2:
                    # Check for additional jumps
                    additional_jumps = [
                        move for move in self.env.valid_moves(self.current_player)
                        if move[:2] == [end_row, end_col] and abs(move[2] - move[0]) == 2
                    ]

                    if additional_jumps:
                        self.selected_piece = (end_row, end_col)  # Keep the current piece selected
                        self.valid_destinations = [move[2:4] for move in additional_jumps]
                        print(f"Forced multi-jump available: {self.valid_destinations}")
                    else:
                        self.current_player = 2 if self.current_player == 1 else 1  # Switch player
                        self.selected_piece = None  # Deselect piece
                        self.valid_destinations = []  # Clear highlights
                else:
                    self.current_player = 2 if self.current_player == 1 else 1  # Switch player
                    self.selected_piece = None  # Deselect piece
                    self.valid_destinations = []  # Clear highlights

                self.render_board()
                self.check_winner()

                # If it's AI's turn, let the AI make a move
                if self.current_player == 2:
                    self.ai_move()

    def regret_move(self):
        if self.history:
            last_state, last_player = self.history.pop()
            self.env.board = last_state
            self.current_player = last_player
            self.render_board()
        else:
            messagebox.showinfo("Undo Move", "No moves to undo.")

    def set_difficulty(self, difficulty):

        self.difficulty = difficulty
        self.board_size = 6 if difficulty == 'easy' else 8
        self.cell_size = self.canvas_size // self.board_size
        self.env = CheckersEnv(board_size=self.board_size)
        self.agent = self.create_agent()
        self.reset_game()

    def render_board(self):
        self.canvas.delete("all")

        for row in range(self.board_size):
            for col in range(self.board_size):
                x0, y0 = col * self.cell_size, row * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                color = "#D0E4C8" if (row + col) % 2 == 0 else "#F0F5F1"
                if (row, col) in self.valid_destinations:
                    color = "#B0E57C"

                if self.selected_piece == (row, col):
                    color = "#FFD700"

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")

                piece = self.env.board[row, col]
                if piece == 1:
                    self.canvas.create_oval(x0 + 8, y0 + 8, x1 - 8, y1 - 8, fill="black")
                elif piece == 2:
                    self.canvas.create_oval(x0 + 8, y0 + 8, x1 - 8, y1 - 8, fill="red")
                elif piece == 3:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="black", outline="gold", width=5)
                elif piece == 4:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="red", outline="gold", width=5)

    def reset_game(self):

        self.history.clear()
        self.env.reset()
        self.current_player = 1
        self.selected_piece = None
        self.valid_destinations = []
        self.render_board()

    def check_winner(self):
        winner = self.env.game_winner()
        if winner is not None:
            messagebox.showinfo("Game Over", f"Player {winner} Wins!")
            self.root.quit()

    def ai_move(self):
        while self.current_player == 2:  # **AI 需要连跳**
            action = self.agent.choose_action(self.env.board)
            if action is not None:
                self.history.append((self.env.board.copy(), self.current_player))
                self.env.step(action, self.current_player)
                self.render_board()
                self.check_winner()
                if self.env.has_moved:
                    print(f"⚠️ AI Player {self.current_player} has already moved. Skipping turn.")
                    self.current_player = 1
                    return


                additional_jumps = [
                    move for move in self.env.valid_moves(self.current_player)
                    if move[:2] == action[2:4] and abs(move[2] - move[0]) == 2
                ]

                if not additional_jumps:
                    self.current_player = 1
            else:

                winner = self.env.game_winner()
                if winner is not None:
                    messagebox.showinfo("Game Over", f"Player {winner} Wins!")
                    self.root.quit()


