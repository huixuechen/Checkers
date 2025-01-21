import tkinter as tk
from tkinter import messagebox
from checkers_env import checkers_env
from LearningAgent import QLearningAgent

class CheckerGUI:
    def __init__(self, root, difficulty='low'):
        self.root = root
        self.root.title("Checkers")
        self.difficulty = difficulty
        self.board_size = 6 if difficulty == 'low' else 8
        self.env = checkers_env(board_size=self.board_size)
        self.canvas_size = 500
        self.cell_size = self.canvas_size // self.board_size
        self.current_player = 1
        self.selected_piece = None
        self.valid_destinations = []
        self.history = []

        self.agent = self.create_agent(difficulty)

        self.setup_ui()
        self.render_board()

        self.canvas.bind("<ButtonPress-1>", self.on_piece_press)
        self.canvas.bind("<B1-Motion>", self.on_piece_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_piece_release)

    def create_agent(self, difficulty):
        if difficulty == 'low':
            return QLearningAgent(self.env, player=-1, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0)
        elif difficulty == 'medium':
            return QLearningAgent(self.env, player=-1, learning_rate=0.05, discount_factor=0.95, exploration_rate=0.5)
        else:  # high difficulty
            return QLearningAgent(self.env, player=-1, learning_rate=0.01, discount_factor=0.99, exploration_rate=0.1)

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#D0E4C8")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame, bg="#D0E4C8", relief=tk.GROOVE, bd=2)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        tk.Label(control_frame, text="Checkers", font=("Arial", 12, "bold"), bg="#D0E4C8").pack(pady=5)
        tk.Label(control_frame, text="Difficulty", font=("Arial", 12, "bold"), bg="#D0E4C8").pack(pady=5)
        tk.Button(control_frame, text="Low", command=lambda: self.set_difficulty('low'), width=15).pack(pady=5)
        tk.Button(control_frame, text="Medium", command=lambda: self.set_difficulty('medium'), width=15).pack(pady=5)
        tk.Button(control_frame, text="High", command=lambda: self.set_difficulty('high'), width=15).pack(pady=5)

        tk.Label(control_frame, text="Game Info", font=("Arial", 12, "bold"), bg="#D0E4C8").pack(pady=5)
        self.difficulty_label = tk.Label(control_frame, text=f"Difficulty: {self.difficulty.capitalize()}", font=("Arial", 10), bg="#D0E4C8")
        self.difficulty_label.pack(pady=5)
        self.status_label = tk.Label(control_frame, text="Player 1's Turn", font=("Arial", 10), bg="#D0E4C8")
        self.invalid_move_label = tk.Label(control_frame, text="", font=("Arial", 10, "italic"), fg="red", bg="#D0E4C8")
        self.status_label.pack(pady=5)
        self.invalid_move_label.pack(pady=5)


        tk.Button(control_frame, text="Reset Game", command=self.reset_game, width=15).pack(pady=10)
        tk.Button(control_frame, text="Show Valid Moves", command=self.show_valid_moves, width=15).pack(pady=5)
        tk.Button(control_frame, text="Regret Chess", command=self.regret_move, width=15).pack(pady=5)
        tk.Button(control_frame, text="History Record", command=self.show_history, width=15).pack(pady=5)

        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, bg="#F0F5F1", relief=tk.SUNKEN, bd=2)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        right_frame = tk.Frame(main_frame, bg="#D0E4C8")
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        self.board_size = 6 if difficulty == 'low' else 8
        self.cell_size = self.canvas_size // self.board_size
        self.env = checkers_env(board_size=self.board_size)
        self.agent = self.create_agent(difficulty)
        self.difficulty_label.config(text=f"Difficulty: {self.difficulty.capitalize()}")
        self.reset_game()

    def render_board(self):
        self.canvas.delete("all")
        for row in range(self.board_size):
            for col in range(self.board_size):
                x0 = col * self.cell_size
                y0 = row * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                color = "#D0E4C8" if (row + col) % 2 == 0 else "#F0F5F1"
                if (row, col) in self.valid_destinations:
                    color = "light green"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")

                if self.env.board[row, col] == 1:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="black")
                elif self.env.board[row, col] == -1:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="red")

    def reset_game(self):
        self.history.clear()
        self.env.reset()
        self.current_player = 1
        self.selected_piece = None
        self.valid_destinations = []
        self.status_label.config(text="Player 1's Turn")
        self.render_board()

    def show_valid_moves(self):
        moves = self.env.valid_moves(self.current_player)
        if not moves:
            self.check_winner()
        else:
            move_str = "\n".join([f"{m[:2]} -> {m[2:]}" for m in moves])
            messagebox.showinfo("Valid Moves", f"Valid Moves for Player {self.current_player}:\n{move_str}")

    def regret_move(self):
        if self.history:
            last_move = self.history.pop()
            self.env.board = last_move[0]
            self.current_player = last_move[1]
            self.status_label.config(text=f"Player {self.current_player}'s Turn")
            self.render_board()
        else:
            messagebox.showinfo("Regret Move", "No moves to regret.")

    def show_history(self):
        history_str = "\n".join([f"Move {i+1}: Player {h[1]}" for i, h in enumerate(self.history)])
        if history_str:
            messagebox.showinfo("Move History", history_str)
        else:
            messagebox.showinfo("Move History", "No moves have been made yet.")

    def on_piece_press(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if self.env.board[row, col] == self.current_player:
            self.selected_piece = (row, col)
            self.valid_destinations = [tuple(m[2:]) for m in self.env.valid_moves(self.current_player) if m[:2] == [row, col]]
            self.render_board()

    def on_piece_motion(self, event):
        if self.selected_piece:
            self.render_board()
            x0 = event.x - self.cell_size // 2
            y0 = event.y - self.cell_size // 2
            x1 = event.x + self.cell_size // 2
            y1 = event.y + self.cell_size // 2
            color = "black" if self.current_player == 1 else "red"
            self.canvas.create_oval(x0, y0, x1, y1, fill=color, tags="drag_piece")

    def on_piece_release(self, event):
        if self.selected_piece:
            start_row, start_col = self.selected_piece
            end_col = event.x // self.cell_size
            end_row = event.y // self.cell_size
            action = [start_row, start_col, end_row, end_col]
            valid_moves = self.env.valid_moves(self.current_player)

            if action in valid_moves:
                self.history.append((self.env.board.copy(), self.current_player))
                self.env.step(action, self.current_player)
                self.current_player = -self.current_player
                self.status_label.config(text=f"Player {-self.current_player}'s Turn")
                self.valid_destinations = []
                self.render_board()
                self.check_winner()

                if self.current_player == -1:
                    self.agent_play()

            else:
                self.invalid_move_label.config(text="Invalid Move")
            self.selected_piece = None
            self.render_board()

    def agent_play(self):
        state = self.env.board
        action = self.agent.choose_action(state)
        if action:
            self.history.append((self.env.board.copy(), self.current_player))
            self.env.step(action, self.current_player)
            self.current_player = -self.current_player
            self.status_label.config(text=f"Player {self.current_player}'s Turn")
            self.render_board()
            self.check_winner()

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