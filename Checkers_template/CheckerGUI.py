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

        # 初始化棋盘大小
        self.board_size = 6 if difficulty == 'easy' else 8
        self.env = CheckersEnv(board_size=self.board_size)
        self.canvas_size = 500
        self.cell_size = self.canvas_size // self.board_size
        self.current_player = 1
        self.selected_piece = None
        self.valid_destinations = []
        self.history = []

        # 初始化 AI 代理
        self.agent = self.create_agent()

        # 设置 UI 界面
        self.setup_ui()
        self.render_board()

    def create_agent(self):
        return QLearningAgent(self.env, player=2, difficulty=self.difficulty)

    def setup_ui(self):
        """设置 UI 界面"""
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
        """处理棋子按下事件"""
        col, row = event.x // self.cell_size, event.y // self.cell_size
        if self.env.board[row, col] == self.current_player:
            valid_moves = self.env.valid_moves(self.current_player)
            print(f"Valid moves for player {self.current_player}: {valid_moves}")  # 调试代码
            self.selected_piece = (row, col)

    def on_piece_release(self, event):
        """处理棋子释放事件并执行移动"""

        if self.selected_piece:
            start_row, start_col = self.selected_piece
            end_row, end_col = event.y // self.cell_size, event.x // self.cell_size
            action = [start_row, start_col, end_row, end_col]
            if action in self.env.valid_moves(self.current_player):
                self.history.append((self.env.board.copy(), self.current_player))
                self.env.step(action, self.current_player)
                self.current_player = 2 if self.current_player == 1 else 1
                self.render_board()
                self.check_winner()

                # 如果轮到AI，自动执行AI的回合
                if self.current_player == 2:
                    self.ai_move()

            self.selected_piece = None



    def regret_move(self):
        """撤销上一步移动"""
        if self.history:
            last_state, last_player = self.history.pop()
            self.env.board = last_state
            self.current_player = last_player
            self.render_board()
        else:
            messagebox.showinfo("Undo Move", "No moves to undo.")

    def set_difficulty(self, difficulty):
        """调整难度并重置游戏"""
        self.difficulty = difficulty
        self.board_size = 6 if difficulty == 'easy' else 8
        self.cell_size = self.canvas_size // self.board_size
        self.env = CheckersEnv(board_size=self.board_size)
        self.agent = self.create_agent()
        self.reset_game()

    def render_board(self):
        """渲染棋盘"""
        self.canvas.delete("all")
        for row in range(self.board_size):
            for col in range(self.board_size):
                x0, y0 = col * self.cell_size, row * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                color = "#D0E4C8" if (row + col) % 2 == 0 else "#F0F5F1"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
                piece = self.env.board[row, col]
                if piece == 1:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="black")
                elif piece == 2:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="red")
                elif piece == 3:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="black", outline="gold", width=3)
                elif piece == 4:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="red", outline="gold", width=3)

    def reset_game(self):
        """重置游戏"""
        self.history.clear()
        self.env.reset()
        self.current_player = 1
        self.selected_piece = None
        self.valid_destinations = []
        self.render_board()

    def check_winner(self):
        """检查游戏是否结束"""
        winner = self.env.game_winner()
        if winner is not None:
            messagebox.showinfo("Game Over", f"Player {winner} Wins!")
            self.root.quit()

    def ai_move(self):
        """让AI执行回合"""
        if self.current_player == 2:
            action = self.agent.choose_action(self.env.board)
            if action is not None:
                self.history.append((self.env.board.copy(), self.current_player))
                self.env.step(action, self.current_player)
                self.current_player = 1  # 轮到玩家
                self.render_board()
                self.check_winner()
            else:
                # AI 无法移动，游戏可能结束
                winner = self.env.game_winner()
                if winner is not None:
                    messagebox.showinfo("Game Over", f"Player {winner} Wins!")
                    self.root.quit()

