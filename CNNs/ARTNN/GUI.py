import Application
import tkinter as tk

class Gui():
    def __init__(self):
        self.root = tk.Tk()
        self.title = "Art NN 1.0 (Beta) by Daniel Rossi"
        self.root.title(self.title)

        self.root.resizable(width=True, height=True)

        app = Application.Application(master=self.root)
        app.mainloop()

