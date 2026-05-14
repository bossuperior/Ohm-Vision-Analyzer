import cv2
import numpy as np
from src.vision import color_mapping
from config.configs import ACCENT, DIM, GREEN


class CallbackMixin:
    def _apply_warp(self):
        self.transformer.margin  = self.var_margin.get()
        self.transformer.shift_x = self.var_shift_x.get() - 100
        self.transformer.shift_y = self.var_shift_y.get() - 100

    def _apply_grid(self):
        self.grid_mapper.set_params(
            self.var_off_x.get(), self.var_off_y.get(),
            pitch_x=self.var_pitch_x.get() / 10.0,
            pitch_y=self.var_pitch_y.get() / 10.0,
        )

    def _toggle_grid(self):
        self.show_grid = not self.show_grid
        self.btn_grid.config(
            text="Grid  ON" if self.show_grid else "Grid  OFF",
            bg="#166534"    if self.show_grid else ACCENT)

    def _toggle_cal_mode(self):
        self._cal_active = not self._cal_active
        if self._cal_active:
            self._cal_btn.config(text="Cancel", bg="#7c3a1d")
            self._cal_status.config(
                text=f"Click image to sample '{self._cal_var.get()}'", fg="#f9a825")
            self.canvas.config(cursor="crosshair")
        else:
            self._cal_btn.config(text="Sample from Image", bg=ACCENT)
            self._cal_status.config(text="", fg=DIM)
            self.canvas.config(cursor="")

    def _on_canvas_click(self, event):
        if not self._cal_active or self._last_warped is None:
            return
        scale, x0, y0 = self._lb
        h_f, w_f = self._last_warped.shape[:2]
        fx = int(np.clip((event.x - x0) / scale, 0, w_f - 1))
        fy = int(np.clip((event.y - y0) / scale, 0, h_f - 1))
        hsv = cv2.cvtColor(np.uint8([[self._last_warped[fy, fx]]]), cv2.COLOR_BGR2HSV)[0][0]
        name = self._cal_var.get()
        color_mapping.REF_COLORS[name] = tuple(int(v) for v in hsv)
        self._update_cal_swatch()
        self._cal_active = False
        self._cal_btn.config(text="Sample from Image", bg=ACCENT)
        self._cal_status.config(text=f"Saved {name}: H{hsv[0]} S{hsv[1]} V{hsv[2]}", fg=GREEN)
        self.canvas.config(cursor="")

    def _update_cal_swatch(self):
        name = self._cal_var.get()
        h, s, v = (int(x) for x in color_mapping.REF_COLORS.get(name, (0, 0, 40)))
        bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
        r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
        hex_col = f'#{r:02x}{g:02x}{b:02x}'
        self._cal_swatch.config(bg=hex_col,
                                fg="#000000" if 0.299*r + 0.587*g + 0.114*b > 128 else "#ffffff")
        self._cal_hsv_label.config(text=f"H{h} S{s} V{v}")
