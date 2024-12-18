
import checkers_env
import LearningAgent
import random
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox
from checkers_env import checkers_env



env = checkers_env(player=1)

print("Initialized Board:")
env.render()
current_player = 1  # 1 代表玩家1，-1代表玩家2


class CheckersGUI:
    def __init__(self, root, difficulty='low'):
        self.root = root
        self.root.title("Checkers")
        self.difficulty = difficulty  # Difficulty level: low, medium, high
        self.board_size = 6 if difficulty == 'low' else 8
        self.env = checkers_env(board_size=self.board_size)
        self.canvas_size = 500  # Canvas size for drawing the board
        self.cell_size = self.canvas_size // self.board_size
        self.current_player = 1
        self.selected_piece = None  # Track selected piece for movement
        self.valid_destinations = []  # Track valid moves for selected piece
        self.history = []  # To store the history of moves

        self.setup_ui()
        self.render_board()

    def setup_ui(self):
        """ Set up buttons and canvas """
        # Main Frame
        main_frame = tk.Frame(self.root, bg="#D0E4C8")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Control Panel
        control_frame = tk.Frame(main_frame, bg="#D0E4C8", relief=tk.GROOVE, bd=2)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        # Difficulty Selection
        tk.Label(control_frame, text="Select Difficulty", font=("Arial", 12, "bold"), bg="#D0E4C8").pack(pady=5)
        tk.Button(control_frame, text="Low", command=lambda: self.set_difficulty('low'), width=15).pack(pady=5)
        tk.Button(control_frame, text="Medium", command=lambda: self.set_difficulty('medium'), width=15).pack(pady=5)
        tk.Button(control_frame, text="High", command=lambda: self.set_difficulty('high'), width=15).pack(pady=5)

        # Game Information
        tk.Label(control_frame, text="Game Info", font=("Arial", 12, "bold"), bg="#D0E4C8").pack(pady=5)
        self.difficulty_label = tk.Label(control_frame, text=f"Difficulty: {self.difficulty.capitalize()}", font=("Arial", 10), bg="#D0E4C8")
        self.difficulty_label.pack(pady=5)
        self.status_label = tk.Label(control_frame, text="Player 1's Turn", font=("Arial", 10), bg="#D0E4C8")
        self.status_label.pack(pady=5)

        # Buttons for actions
        tk.Button(control_frame, text="Reset Game", command=self.reset_game, width=15).pack(pady=10)
        tk.Button(control_frame, text="Show Valid Moves", command=self.show_valid_moves, width=15).pack(pady=5)
        tk.Button(control_frame, text="Regret Chess", command=self.regret_move, width=15).pack(pady=5)
        tk.Button(control_frame, text="History Record", command=self.show_history, width=15).pack(pady=5)

        # Board Canvas
        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, bg="#F0F5F1", relief=tk.SUNKEN, bd=2)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Right Panel Placeholder
        right_frame = tk.Frame(main_frame, bg="#D0E4C8")
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        tk.Label(right_frame, text="Checkers", font=("Arial", 20, "bold"), bg="#D0E4C8").pack()

    def set_difficulty(self, difficulty):
        """ Change the difficulty level and reset the game. """
        self.difficulty = difficulty
        self.board_size = 6 if difficulty == 'low' else 8
        self.cell_size = self.canvas_size // self.board_size
        self.env = checkers_env(board_size=self.board_size)
        self.difficulty_label.config(text=f"Difficulty: {self.difficulty.capitalize()}")
        self.reset_game()

    def render_board(self):
        """ Renders the checkers board on the canvas. """
        self.canvas.delete("all")
        for row in range(self.board_size):
            for col in range(self.board_size):
                x0 = col * self.cell_size
                y0 = row * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                # Draw cells
                color = "#D0E4C8" if (row + col) % 2 == 0 else "#F0F5F1"  # Light green and white
                if (row, col) in self.valid_destinations:
                    color = "light green"  # Highlight valid moves
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")

                # Draw pieces
                if self.env.board[row, col] == 1:  # Player 1's piece
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="black")
                elif self.env.board[row, col] == -1:  # Player -1's piece
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="red")

    def reset_game(self):
        """ Resets the game board. """
        self.history.clear()
        self.env.reset()
        self.current_player = 1
        self.selected_piece = None
        self.valid_destinations = []
        self.status_label.config(text="Player 1's Turn")
        self.render_board()

    def computer_move(self):
        valid_moves = self.env.valid_moves(-1)
        action = self.agent.get_best_move(valid_moves)
        if action:
            self.env.step(action, -1)
            self.check_winner()
            self.current_player = 1
            self.status_label.config(text="Player 1's Turn")
        self.render_board()

    def show_valid_moves(self):
        """ Shows valid moves for the current player. """
        moves = self.env.valid_moves(self.current_player)
        move_str = "\n".join([f"{m[:2]} -> {m[2:]}" for m in moves])
        if moves:
            messagebox.showinfo("Valid Moves", f"Valid Moves for Player {self.current_player}:\n{move_str}")
        else:
            messagebox.showinfo("No Moves", f"No valid moves for Player {self.current_player}.")

    def regret_move(self):
        """ Undo the last move. """
        if self.history:
            last_move = self.history.pop()
            self.env.board = last_move[0]
            self.current_player = last_move[1]
            self.status_label.config(text=f"Player {self.current_player}'s Turn")
            self.render_board()
        else:
            messagebox.showinfo("Regret Move", "No moves to regret.")

    def show_history(self):
        """ Display the history of moves. """
        history_str = "\n".join([f"Move {i+1}: Player {h[1]}" for i, h in enumerate(self.history)])
        if history_str:
            messagebox.showinfo("Move History", history_str)
        else:
            messagebox.showinfo("Move History", "No moves have been made yet.")

    def on_canvas_click(self, event):
        """ Handles click events on the canvas to move pieces. """
        col = event.x // self.cell_size  # Convert x-coordinate to column index
        row = event.y // self.cell_size  # Convert y-coordinate to row index
        print(f"Clicked position: Row {row}, Col {col}")

        # Check for out-of-bound clicks
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return  # Ignore invalid clicks outside the board

        if self.selected_piece:
            # Try to move the piece
            start_row, start_col = self.selected_piece
            action = [start_row, start_col, row, col]
            valid_moves = self.env.valid_moves(self.current_player)

            # Debug: print valid moves for troubleshooting
            print("Valid Moves:", valid_moves)
            print("Attempted Action:", action)

            if action in valid_moves:
                # Save the state to history before making the move
                self.history.append((self.env.board.copy(), self.current_player))
                self.env.step(action, self.current_player)
                self.current_player = -self.current_player  # Switch player
                self.status_label.config(text=f"Player {-self.current_player}'s Turn")
                self.valid_destinations = []
                self.render_board()
            else:
                messagebox.showwarning("Invalid Move", "This is not a valid move.")
            self.selected_piece = None  # Clear the selected piece
        elif self.env.board[row, col] == self.current_player:
            # Select the piece and highlight valid moves
            self.selected_piece = (row, col)
            self.valid_destinations = [tuple(m[2:]) for m in self.env.valid_moves(self.current_player) if
                                       m[:2] == [row, col]]
            self.render_board()
        else:
            messagebox.showwarning("Invalid Selection", "Select your own piece to move.")

    def check_winner(self):
        winner = self.env.game_winner(self.env.board)
        if winner is not None:
            if winner == 1:
                messagebox.showinfo("Game Over", "Player 1 Wins!")
            elif winner == -1:
                messagebox.showinfo("Game Over", "Computer Wins!")
            else:
                messagebox.showinfo("Game Over", "It's a Draw!")
            self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    gui = CheckersGUI(root, difficulty='low')
    root.mainloop()

if __name__ == "__main__":
    env = checkers_env(player=1)
    env.render()

    print("\nPlayer 1's valid moves:")
    moves = env.valid_moves(1)

    if moves:
        action = moves[0]
        print(f"\nExecuting move: {action}")
        env.step(action, player=1)
        print("\nBoard after Player 1's move:")
        env.render()

        # 切换到 Player -1
        print("\nPlayer -1's valid moves:")
        moves = env.valid_moves(-1)
        if moves:
            action = moves[0]
            print(f"\nExecuting move: {action}")
            env.step(action, player=-1)
            print("\nBoard after Player -1's move:")
            env.render()
    else:
        print("No valid moves for Player 1.")






