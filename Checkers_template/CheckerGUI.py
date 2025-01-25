import tkinter as tk
from tkinter import messagebox
from checkers_env import CheckersEnv
from LearningAgent import QLearningAgent


class CheckerGUI:
    def __init__(self, root, difficulty='easy'):
        self.root = root
        self.root.title("Checkers")
        self.difficulty = difficulty

        # Board and environment setup
        self.board_size = 6 if difficulty == 'easy' else 8
        self.env = CheckersEnv(board_size=self.board_size)
        self.canvas_size = 500
        self.cell_size = self.canvas_size // self.board_size
        self.current_player = 1
        self.selected_piece = None
        self.valid_destinations = []
        self.history = []

        # Initialize agent
        self.agent = self.create_agent()

        # Setup UI and bindings
        self.setup_ui()
        self.render_board()


    def create_agent(self):
        """Initialize QLearningAgent based on current difficulty."""
        return QLearningAgent(self.env, player=2, difficulty=self.difficulty)

    def setup_ui(self):
        """Set up the UI layout and controls."""
        main_frame = tk.Frame(self.root, bg="#D0E4C8")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame, bg="#D0E4C8", relief=tk.GROOVE, bd=2)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        tk.Label(control_frame, text="Checkers", font=("Arial", 12, "bold"), bg="#D0E4C8").pack(pady=5)
        tk.Label(control_frame, text="Difficulty", font=("Arial", 12, "bold"), bg="#D0E4C8").pack(pady=5)
        tk.Button(control_frame, text="Easy", command=lambda: self.set_difficulty('easy'), width=15).pack(pady=5)
        tk.Button(control_frame, text="Medium", command=lambda: self.set_difficulty('medium'), width=15).pack(pady=5)
        tk.Button(control_frame, text="Hard", command=lambda: self.set_difficulty('hard'), width=15).pack(pady=5)

        tk.Label(control_frame, text="Game Info", font=("Arial", 12, "bold"), bg="#D0E4C8").pack(pady=5)
        self.difficulty_label = tk.Label(control_frame, text=f"Difficulty: {self.difficulty.capitalize()}", bg="#D0E4C8")
        self.difficulty_label.pack(pady=5)
        self.status_label = tk.Label(control_frame, text="Player 1's Turn", bg="#D0E4C8")
        self.status_label.pack(pady=5)

        tk.Button(control_frame, text="Reset Game", command=self.reset_game, width=15).pack(pady=10)
        tk.Button(control_frame, text="Show Valid Moves", command=self.show_valid_moves, width=15).pack(pady=5)
        tk.Button(control_frame, text="Undo Last Move", command=self.regret_move, width=15).pack(pady=5)

        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, bg="#F0F5F1", relief=tk.SUNKEN, bd=2)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<ButtonPress-1>", self.on_piece_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_piece_release)

    def set_difficulty(self, difficulty):
        """Adjust settings based on difficulty."""
        self.difficulty = difficulty
        self.board_size = 6 if difficulty == 'easy' else 8
        self.cell_size = self.canvas_size // self.board_size
        self.env = CheckersEnv(board_size=self.board_size)
        self.agent = self.create_agent()
        self.difficulty_label.config(text=f"Difficulty: {self.difficulty.capitalize()}")
        self.reset_game()

    def render_board(self):
        self.canvas.delete("all")
        for row in range(self.board_size):
            for col in range(self.board_size):
                x0, y0 = col * self.cell_size, row * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size  # Corrected y1 assignment
                color = "#D0E4C8" if (row + col) % 2 == 0 else "#F0F5F1"
                if (row, col) in self.valid_destinations:
                    color = "#FFFF00"  # Highlight valid destinations in yellow
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
                if self.env.board[row, col] == 1:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="black")
                elif self.env.board[row, col] == 2:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="red")

    def reset_game(self):
        """Reset the game to its initial state."""
        self.history.clear()
        self.env.reset()
        self.current_player = 1
        self.selected_piece = None
        self.valid_destinations = []
        self.status_label.config(text="Player 1's Turn")
        self.render_board()

    def show_valid_moves(self):
        """Display valid moves for the current player."""
        moves = self.env.valid_moves(self.current_player)
        if moves:
            move_str = "\n".join([f"{m[:2]} -> {m[2:]}" for m in moves])
            messagebox.showinfo("Valid Moves", f"Player {self.current_player}'s valid moves:\n{move_str}")
        else:
            self.check_winner()

    def regret_move(self):
        """Undo the last move."""
        if self.history:
            last_state, last_player = self.history.pop()
            self.env.board = last_state
            self.current_player = last_player
            self.status_label.config(text=f"Player {self.current_player}'s Turn")
            self.render_board()
        else:
            messagebox.showinfo("Undo Move", "No moves to undo.")

    def on_piece_press(self, event):
        """Handle the selection of a piece."""
        col, row = event.x // self.cell_size, event.y // self.cell_size
        if self.env.board[row, col] == self.current_player:
            self.selected_piece = (row, col)
            self.valid_destinations = [tuple(m[2:]) for m in self.env.valid_moves(self.current_player) if
                                       m[:2] == [row, col]]
            self.render_board()

    def on_piece_release(self, event):
        """Handle piece placement."""
        if self.selected_piece:
            start_row, start_col = self.selected_piece
            end_row, end_col = event.y // self.cell_size, event.x // self.cell_size
            action = [start_row, start_col, end_row, end_col]
            if action in self.env.valid_moves(self.current_player):
                self.history.append((self.env.board.copy(), self.current_player))
                self.env.step(action, self.current_player)
                self.current_player = -self.current_player
                self.status_label.config(text=f"Player {-self.current_player}'s Turn")
                self.valid_destinations = []
                self.render_board()
                self.check_winner()

                if self.current_player == -1:
                    self.root.after(1000, self.agent_play)
            else:
                messagebox.showinfo("Invalid Move", "The selected move is not valid.")
            self.selected_piece = None

    def agent_play(self):
        """Make the agent play its turn."""
        action = self.agent.choose_action(self.env.board)
        if action:
            self.history.append((self.env.board.copy(), self.current_player))
            self.env.step(action, self.current_player)
            self.current_player = 1  # Switch back to player 1
            self.render_board()
            self.check_winner()

    def check_winner(self):
        winner = self.env.game_winner()
        if winner is not None:
            if winner == 1:
                messagebox.showinfo("Game Over", "Player 1 Wins!")
            elif winner == 2:
                messagebox.showinfo("Game Over", "Player 2 Wins!")
            else:
                messagebox.showinfo("Game Over", "It's a Draw!")
            self.root.quit()
