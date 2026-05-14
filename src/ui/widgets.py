import tkinter as tk
from src.ui.theme import PANEL_BG, ACCENT, TEXT, DIM, GREEN


def divider(parent):
    tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill="x", padx=8, pady=4)


def section_label(parent, text):
    tk.Label(parent, text=text, font=("Arial", 9, "bold"),
             bg=PANEL_BG, fg=DIM).pack(anchor="w", padx=14, pady=(2, 0))


def slider(parent, label, var, lo, hi, cmd):
    row = tk.Frame(parent, bg=PANEL_BG)
    row.pack(fill="x", padx=10, pady=1)
    tk.Label(row, text=label, width=7, anchor="w",
             font=("Arial", 8), bg=PANEL_BG, fg=TEXT).pack(side="left")
    tk.Scale(row, variable=var, from_=lo, to=hi,
             orient="horizontal", length=108, showvalue=False,
             bg=PANEL_BG, fg=TEXT, troughcolor=ACCENT,
             highlightthickness=0, sliderlength=14, bd=0,
             command=lambda _: cmd()).pack(side="left")
    tk.Label(row, textvariable=var, width=4,
             font=("Consolas", 8), bg=PANEL_BG, fg=GREEN).pack(side="left")
