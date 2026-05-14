import tkinter as tk
from config.configs import BG, PANEL_BG, ACCENT, TEXT, DIM, GREEN, RED, COLOR_NAMES
from src.ui import widgets


class UIBuilderMixin:
    def _build_ui(self):
        hdr = tk.Frame(self.window, bg=ACCENT, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Ohm-Vision Analyzer",
                 font=("Arial", 15, "bold"), bg=ACCENT, fg="white").pack(
            side="left", padx=16, pady=12)
        self.status_badge = tk.Label(hdr, text="  SEARCHING  ",
                                     font=("Arial", 10, "bold"),
                                     bg=RED, fg="white", padx=6)
        self.status_badge.pack(side="right", padx=16, pady=13)

        body = tk.Frame(self.window, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg="#000",
                                highlightthickness=2, highlightbackground=ACCENT)
        self.canvas.pack(fill="both", expand=True)
        tk.Label(left, text="[Q] Quit   [G] Grid   [F11] Fullscreen   [Esc] Exit FS",
                 font=("Consolas", 9), bg=BG, fg=DIM).pack(pady=4)

        panel_outer = tk.Frame(body, bg=PANEL_BG, width=230)
        panel_outer.pack(side="right", fill="y", padx=(10, 0))
        panel_outer.pack_propagate(False)
        cs = tk.Canvas(panel_outer, bg=PANEL_BG, highlightthickness=0, width=214)
        sb = tk.Scrollbar(panel_outer, orient="vertical", command=cs.yview)
        cs.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cs.pack(side="left", fill="both", expand=True)
        panel = tk.Frame(cs, bg=PANEL_BG)
        pw = cs.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda _: cs.configure(scrollregion=cs.bbox("all")))
        cs.bind("<Configure>",   lambda e: cs.itemconfig(pw, width=e.width))
        panel_outer.bind_all("<MouseWheel>",
                             lambda e: cs.yview_scroll(int(-1 * e.delta / 120), "units"))

        tk.Label(panel, text="Controls", font=("Arial", 12, "bold"),
                 bg=PANEL_BG, fg=TEXT).pack(pady=(14, 6))

        widgets.divider(panel); widgets.section_label(panel, "Warp")
        self.var_margin  = tk.IntVar(value=self._cfg["margin"])
        self.var_shift_x = tk.IntVar(value=self._cfg["shift_x"])
        self.var_shift_y = tk.IntVar(value=self._cfg["shift_y"])
        widgets.slider(panel, "Margin",  self.var_margin,  0, 300, self._apply_warp)
        widgets.slider(panel, "Shift X", self.var_shift_x, 0, 200, self._apply_warp)
        widgets.slider(panel, "Shift Y", self.var_shift_y, 0, 200, self._apply_warp)

        widgets.divider(panel); widgets.section_label(panel, "Grid")
        self.var_off_x   = tk.IntVar(value=self._cfg["off_x"])
        self.var_off_y   = tk.IntVar(value=self._cfg["off_y"])
        self.var_pitch_x = tk.IntVar(value=self._cfg["pitch_x"])
        self.var_pitch_y = tk.IntVar(value=self._cfg["pitch_y"])
        widgets.slider(panel, "Off X",   self.var_off_x,   0, 200, self._apply_grid)
        widgets.slider(panel, "Off Y",   self.var_off_y,   0, 200, self._apply_grid)
        widgets.slider(panel, "Pitch X", self.var_pitch_x, 0, 500, self._apply_grid)
        widgets.slider(panel, "Pitch Y", self.var_pitch_y, 0, 500, self._apply_grid)

        widgets.divider(panel); widgets.section_label(panel, "Color Cal")
        self._build_color_cal(panel)

        widgets.divider(panel); widgets.section_label(panel, "Circuit")
        cbox = tk.Frame(panel, bg=PANEL_BG)
        cbox.pack(fill="x", padx=12, pady=4)
        self.circuit_type_lbl = tk.Label(
            cbox, text="—", font=("Arial", 15, "bold"), bg=PANEL_BG, fg=TEXT,
            wraplength=185, justify="left")
        self.circuit_type_lbl.pack(anchor="w")
        self.circuit_formula_lbl = tk.Label(
            cbox, text="", font=("Consolas", 7), bg=PANEL_BG, fg=TEXT,
            wraplength=185, justify="left")
        self.circuit_formula_lbl.pack(anchor="w")
        self.circuit_ohms_lbl = tk.Label(
            cbox, text="", font=("Arial", 11, "bold"), bg=PANEL_BG, fg=GREEN,
            wraplength=185, justify="left")
        self.circuit_ohms_lbl.pack(anchor="w", pady=(0, 4))

        widgets.divider(panel)
        btn_area = tk.Frame(panel, bg=PANEL_BG)
        btn_area.pack(fill="x", padx=12, pady=10)
        self.btn_grid = tk.Button(btn_area, text="Grid  OFF",
                                  command=self._toggle_grid,
                                  bg=ACCENT, fg="white", font=("Arial", 10),
                                  relief="flat", activebackground="#1a4a8a",
                                  cursor="hand2", height=2)
        self.btn_grid.pack(fill="x", pady=(0, 6))
        tk.Button(btn_area, text="Quit", command=self.on_closing,
                  bg="#7f1d1d", fg="white", font=("Arial", 10),
                  relief="flat", activebackground="#991b1b",
                  cursor="hand2", height=2).pack(fill="x")

        widgets.divider(panel)
        self.readout = tk.Label(panel, text="", font=("Consolas", 8),
                                bg=PANEL_BG, fg=DIM, justify="left")
        self.readout.pack(padx=12, pady=6, anchor="w")

    def _build_color_cal(self, parent):
        cal = tk.Frame(parent, bg=PANEL_BG)
        cal.pack(fill="x", padx=10, pady=4)

        row = tk.Frame(cal, bg=PANEL_BG)
        row.pack(fill="x", pady=(0, 2))
        tk.Label(row, text="Band:", width=5, anchor="w",
                 font=("Arial", 8), bg=PANEL_BG, fg=TEXT).pack(side="left")
        self._cal_var = tk.StringVar(value="BLACK")
        m = tk.OptionMenu(row, self._cal_var, *COLOR_NAMES)
        m.config(bg=ACCENT, fg="white", activebackground="#1a4a8a",
                 font=("Arial", 8), relief="flat", width=7, highlightthickness=0)
        m["menu"].config(bg=ACCENT, fg="white", font=("Arial", 8))
        m.pack(side="left")

        sw_row = tk.Frame(cal, bg=PANEL_BG)
        sw_row.pack(fill="x", pady=2)
        tk.Label(sw_row, text="Color:", font=("Arial", 8),
                 bg=PANEL_BG, fg=DIM).pack(side="left")
        self._cal_swatch = tk.Label(sw_row, text="   ", width=3, relief="solid", bd=1)
        self._cal_swatch.pack(side="left", padx=4)
        self._cal_hsv_label = tk.Label(sw_row, text="",
                                       font=("Consolas", 7), bg=PANEL_BG, fg=GREEN)
        self._cal_hsv_label.pack(side="left")

        self._cal_btn = tk.Button(cal, text="Sample from Image",
                                  command=self._toggle_cal_mode,
                                  bg=ACCENT, fg="white", font=("Arial", 9),
                                  relief="flat", activebackground="#1a4a8a",
                                  cursor="hand2", height=1)
        self._cal_btn.pack(fill="x", pady=(4, 0))
        self._cal_status = tk.Label(cal, text="", font=("Arial", 7, "italic"),
                                    bg=PANEL_BG, fg=DIM, wraplength=180)
        self._cal_status.pack(pady=(2, 0))
        self._cal_var.trace_add("write", lambda *_: self._update_cal_swatch())
