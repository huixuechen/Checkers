import tkinter as tk
from checkers_env import checkers_env
from CheckerGUI import CheckerGUI

if __name__ == "__main__":
    root = tk.Tk()
    gui = CheckerGUI(root, difficulty='low')
    root.mainloop()





