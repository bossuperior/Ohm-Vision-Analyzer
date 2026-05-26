import re
import json
import cv2
import time
import collections
import threading
from collections import Counter
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from src.vision.camera_loader import CameraLoader
from src.inference.model_engine import ModelEngine, ClassificationEngine
from src.vision.breadboard_warper import BreadboardWarper
from src.topology.grid_mapper import GridMapper
from src.topology.circuit_analyzer import CircuitAnalyzer
from src.vision.band_reader import BandReader
from src.utils.crop_from_dataset import crop_body_for_classifier
from src.ui.renderer import draw_results
from src.ui.build_ui import UIBuilderMixin
from src.ui.callback import CallbackMixin
from config.configs import (BG, GREEN, RED, DIM,
                            POSE_BACKEND, POSE_MODEL, POSE_CONF, POSE_IOU,
                            CLS_BACKEND, CLS_MODEL, CLS_DEVICE,
                            CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT,
                            CLS_RESISTOR, CLS_WIRE,
                            DEBUG_CROPS, DEBUG_CROP_DIR)
import config.configs as configs


_CIRCUIT_COLORS = {
    'Series':            '#fbbf24',
    'Parallel':          '#38bdf8',
    'Wheatstone Bridge': '#e879f9',
    'Mixed':             '#fb923c',
    'Ring':              '#c084fc',
    'Single':            '#a3e635',
    'Not Connected':     '#888888',
}

def _parse_ohms(label: str) -> float:
    m = re.match(r'([0-9.]+)(k|M)?\s+Ohm', label)
    if not m:
        return 0.0
    v, p = float(m.group(1)), m.group(2) or ''
    return v * (1e3 if p == 'k' else 1e6 if p == 'M' else 1.0)


def _cls_to_ohm_str(cls_name: str) -> str:
    """Convert '6k8_1pct' → '6.8k Ohm 1%' (format parseable by _parse_ohms)."""
    parts = cls_name.split('_')
    if len(parts) != 2 or not parts[1].endswith('pct'):
        return cls_name
    val_s = parts[0]
    tol   = parts[1].replace('pct', '%')
    m = re.match(r'^(\d+)[Rr](\d*)$', val_s)
    if m:
        num = float(f"{m.group(1)}.{m.group(2) or '0'}")
        return f"{num:g} Ohm {tol}" if num < 1000 else f"{num/1000:g}k Ohm {tol}"
    m = re.match(r'^(\d+)[kK](\d*)$', val_s)
    if m:
        num = float(f"{m.group(1)}.{m.group(2) or '0'}")
        return f"{num:g}k Ohm {tol}"
    m = re.match(r'^(\d+)[mM](\d*)$', val_s)
    if m:
        num = float(f"{m.group(1)}.{m.group(2) or '0'}")
        return f"{num:g}M Ohm {tol}"
    return cls_name


class OhmVisionApp(UIBuilderMixin, CallbackMixin):
    def __init__(self, window):
        self.window = window
        self.window.title("Ohm-Vision Analyzer")
        self.window.configure(bg=BG)
        self.window.resizable(True, True)
        self.window.minsize(900, 600)

        self._fullscreen = False
        self._cfg        = configs.load_ui_state()

        self._cal_active  = False
        self._last_warped = None
        self._lb          = (1.0, 0, 0)

        self.band_reader        = BandReader()
        self.circuit_analyzer   = CircuitAnalyzer()
        self._ohm_cache         = {}
        self._ohm_numeric       = {}  # idx → float (Ω)
        self._ohm_votes         = {}  # idx → deque of recent readings
        self._band_thread       = None
        self._kp_smooth         = {}  # track_key → {'kps', 'cx', 'cy'}

        self._topo_votes  = collections.deque(maxlen=5)   # majority-vote circuit type
        self._stable_info = {'type': '—', 'total_ohms': 0.0, 'formula': '', 'extra': {}}
        self._last_aruco_ok = False   # track board detection state for cache reset

        self._fps_times = collections.deque(maxlen=30)
        self._padded    = None

        self.camera      = CameraLoader(camera_id=CAMERA_ID, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        self.engine      = ModelEngine(POSE_BACKEND, POSE_MODEL, conf=POSE_CONF, iou=POSE_IOU)
        self.class_names = {0: 'resistor', 1: 'wire'}

        self.classifier = ClassificationEngine(
            backend=CLS_BACKEND,
            model_path=CLS_MODEL,
            device=CLS_DEVICE,
        )
        self.transformer = BreadboardWarper(output_width=810, output_height=540)
        self.grid_mapper = GridMapper(target_w=810, target_h=540)
        self.show_grid   = False

        # Inference pipeline runs in a background thread so tkinter never blocks
        self._stop_event  = threading.Event()
        self._raw_lock    = threading.Lock()
        self._raw_frame   = None
        self._result_lock = threading.Lock()
        self._infer_pkg   = None   # (success: bool, base: ndarray, results: DetectionResult)
        self._infer_new   = False

        self._build_ui()
        self._apply_warp()
        self._apply_grid()
        configs.load_color_refs()
        self._update_cal_swatch()

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.window.bind("<q>",      lambda e: self.on_closing())
        self.window.bind("<Q>",      lambda e: self.on_closing())
        self.window.bind("<g>",      lambda e: self._toggle_grid())
        self.window.bind("<G>",      lambda e: self._toggle_grid())
        self.window.bind("<F11>",    lambda e: self._toggle_fullscreen())
        self.window.bind("<Escape>", lambda e: self._exit_fullscreen())
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.camera.start()
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()
        self.window.after(33, self._update_frame)

    def _toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        self.window.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self):
        if self._fullscreen:
            self._fullscreen = False
            self.window.attributes("-fullscreen", False)

    def _start_band_worker(self, warped_snap, kp_snap, cls_snap):
        VOTE_SIZE   = 10
        VOTE_THRESH = 4   # ต้องการ 4/10 (40%) เพราะ crop=None เกิดบ่อย
        BAD = {"?", "ERR", "Unknown", "Read Error", "Calc Error", "Error"}

        def _kp_center(kp_arr):
            pts = [k[:2] for k in kp_arr if len(k) >= 3 and k[2] > 0.3]
            if not pts:
                return None
            return (sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts))

        _debug_dir = None
        if DEBUG_CROPS:
            import os
            _debug_dir = DEBUG_CROP_DIR
            os.makedirs(_debug_dir, exist_ok=True)

        def worker():
            for idx, cls_id in enumerate(cls_snap):
                if int(cls_id) != CLS_RESISTOR or idx >= len(kp_snap):
                    continue

                try:
                    kps     = kp_snap[idx]
                    visible = [kp for kp in kps if kp[2] >= 0.5]
                    crop    = crop_body_for_classifier(
                                  warped_snap, visible[0][:2], visible[-1][:2]
                              ) if len(visible) >= 2 else None
                    if crop is not None:
                        cls_name, conf = self.classifier.predict(crop)
                        res = _cls_to_ohm_str(cls_name)
                        print(f"[Cls{idx}] → {cls_name} ({conf:.2f}) → {res}")

                        if _debug_dir is not None:
                            ts   = int(time.perf_counter() * 1000) % 1_000_000
                            stem = f"{ts:06d}_r{idx}_{cls_name}_{int(conf*100)}pct"
                            # raw crop
                            cv2.imwrite(f"{_debug_dir}/{stem}_raw.jpg", crop)
                            # annotated: band scan overlay
                            try:
                                band_label, bands, _ = self.band_reader.calculate(crop)
                                annot = self.band_reader.annotate_crop(crop, bands, band_label)
                                cv2.imwrite(f"{_debug_dir}/{stem}_annot.jpg", annot)
                            except Exception:
                                pass
                    else:
                        res = "?"
                        print(f"[Cls{idx}] crop=None")
                except Exception as e:
                    res = "?"
                    print(f"[Cls] idx={idx} error: {e}")

                buf = self._ohm_votes.setdefault(idx, collections.deque(maxlen=VOTE_SIZE))
                buf.append(res)
                print(f"[Vote{idx}] buf={list(buf)}")

                valid_votes = [(v, f) for v, f in Counter(buf).most_common() if v not in BAD]
                if valid_votes and valid_votes[0][1] >= VOTE_THRESH:
                    winner = valid_votes[0][0]
                    self._ohm_cache[idx]   = winner
                    self._ohm_numeric[idx] = _parse_ohms(winner)
                elif all(r in BAD for r in buf):
                    self._ohm_cache[idx] = "?"
                    self._ohm_numeric.pop(idx, None)

        self._band_thread = threading.Thread(target=worker, daemon=True)
        self._band_thread.start()

    def _smooth_keypoints(self, keypoints, class_ids) -> np.ndarray:
        """EMA-smooth body-class keypoints matched by centroid proximity across frames."""
        ALPHA    = 0.35   # current-frame weight (lower = smoother)
        MAX_DIST = 50.0   # px — max centroid distance to match same detection

        try:
            kp_arr = np.array(keypoints, dtype=np.float32, copy=True)
        except Exception:
            return keypoints

        if kp_arr.ndim != 3:
            return kp_arr

        new_smooth = {}
        for idx in range(len(kp_arr)):
            if int(class_ids[idx]) != CLS_RESISTOR:
                continue
            kps      = kp_arr[idx]          # (K, 3)
            vis_mask = kps[:, 2] > 0.3
            if not vis_mask.any():
                continue
            cx = float(kps[vis_mask, 0].mean())
            cy = float(kps[vis_mask, 1].mean())

            # Find nearest previous track by centroid
            best_key, best_dist = None, MAX_DIST
            for key, prev in self._kp_smooth.items():
                d = float(np.hypot(cx - prev['cx'], cy - prev['cy']))
                if d < best_dist:
                    best_dist, best_key = d, key

            if best_key is not None:
                prev_kps  = self._kp_smooth[best_key]['kps']
                both_vis  = vis_mask & (prev_kps[:, 2] > 0.3)
                kp_arr[idx, both_vis, 0] = (ALPHA * kps[both_vis, 0]
                                             + (1.0 - ALPHA) * prev_kps[both_vis, 0])
                kp_arr[idx, both_vis, 1] = (ALPHA * kps[both_vis, 1]
                                             + (1.0 - ALPHA) * prev_kps[both_vis, 1])

            new_smooth[idx] = {'kps': kp_arr[idx].copy(), 'cx': cx, 'cy': cy}

        self._kp_smooth = new_smooth
        return kp_arr

    def _inference_loop(self):
        """Background thread: ArUco warp + YOLO inference. Never touches tkinter."""
        while not self._stop_event.is_set():
            with self._raw_lock:
                frame = self._raw_frame
            if frame is None:
                time.sleep(0.002)
                continue

            success, warped, _ = self.transformer.process(frame)
            if success:
                self._last_warped = warped
                results = self.engine.predict(warped)
                base    = warped
            else:
                base    = cv2.resize(frame, (810, 540), interpolation=cv2.INTER_NEAREST)
                results = self.engine.predict(base)

            with self._result_lock:
                self._infer_pkg = (success, base.copy(), results)
                self._infer_new = True

    def _update_frame(self):
        # Push latest camera frame to inference thread
        frame = self.camera.get_frame()
        if frame is not None:
            with self._raw_lock:
                self._raw_frame = frame

        # Pull latest inference result (non-blocking)
        with self._result_lock:
            pkg    = self._infer_pkg
            is_new = self._infer_new
            if is_new:
                self._infer_new = False

        if pkg is None:
            self.window.after(33, self._update_frame)
            return

        success, base, results = pkg

        # ── ล้าง cache ทันทีที่ board หาย (True→False) ─────────────────────
        # ป้องกัน vote เก่าจากตำแหน่งบอร์ดเดิมปนกับ vote ใหม่หลัง re-detect
        if self._last_aruco_ok and not success:
            self._ohm_cache.clear()
            self._ohm_votes.clear()
            self._ohm_numeric.clear()
            self._kp_smooth.clear()
            self._topo_votes.clear()
            self._stable_info = {'type': '—', 'total_ohms': 0.0, 'formula': '', 'extra': {}}
            self._update_circuit_ui(self._stable_info)
        self._last_aruco_ok = success

        # Trigger classification เมื่อมี resistor detection (class 0)
        has_resistor = any(int(c) == CLS_RESISTOR for c in results.class_ids)
        if is_new and success and has_resistor and (self._band_thread is None or not self._band_thread.is_alive()):
            kp_len = len(results.keypoints)
            smoothed_kps = (self._smooth_keypoints(results.keypoints, results.class_ids)
                            if kp_len > 0 else np.array([]))
            self._start_band_worker(
                base.copy(),
                smoothed_kps,
                results.class_ids.copy())

        display = base.copy()

        if success:
            if self.show_grid:
                display = self.grid_mapper.draw_grid_overlay(display)
            resistor_idxs = {i for i, cid in enumerate(results.class_ids)
                             if int(cid) == CLS_RESISTOR}
            ohm_map       = {k: v for k, v in self._ohm_cache.items() if k in resistor_idxs}
            draw_results(display, results, self.class_names, ohm_map=ohm_map, show_keypoints=False)
            cv2.putText(display, "BOARD OK",
                        (12, 32), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 80), 2)
            self.status_badge.config(text="  BOARD OK  ", bg=GREEN, fg="#052e16")

            # Circuit topology — wires merge electrical nodes via Union-Find
            resistor_ids = {idx for idx, cid in enumerate(results.class_ids) if int(cid) == CLS_RESISTOR}
            wire_ids     = {idx for idx, cid in enumerate(results.class_ids) if int(cid) == CLS_WIRE}
            all_kp_data  = [
                {'id': idx, 'keypoints': kps}
                for idx, (_, kps) in enumerate(
                    zip(results.class_ids, results.keypoints))
                if len(kps) >= 2
            ]
            if resistor_ids and all_kp_data:
                all_mapped = self.grid_mapper.map_to_holes(all_kp_data)
                resistors  = self.circuit_analyzer.apply_wires(all_mapped, wire_ids)
                resistors  = [c for c in resistors if c['id'] in resistor_ids]
                for c in resistors:
                    c['ohms'] = self._ohm_numeric.get(c['id'], 0.0)
                info = self.circuit_analyzer.analyze(resistors)

                # Majority-vote: show only when stable (≥3/5 agree)
                self._topo_votes.append(info['type'])
                cnt = Counter(self._topo_votes)
                voted_type, voted_freq = cnt.most_common(1)[0]
                is_stable = voted_freq >= 3

                if is_stable:
                    if info['type'] == voted_type:
                        self._stable_info = info
                    self._update_circuit_ui(self._stable_info)
                    t_display = self._stable_info['type']
                    if t_display not in ('—',):
                        col = _CIRCUIT_COLORS.get(t_display, '#888888')
                        bgr = tuple(int(col[i:i+2], 16) for i in (5, 3, 1))
                        cv2.putText(display, t_display,
                                    (12, 64), cv2.FONT_HERSHEY_DUPLEX, 0.9, bgr, 2)
        else:
            draw_results(display, results, self.class_names, show_keypoints=False)
            cv2.putText(display, "SEARCHING FOR ARUCO TAGS...",
                        (12, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (80, 80, 255), 2)
            self.status_badge.config(text="  SEARCHING  ", bg=RED, fg="white")

        # FPS counter tracks inference throughput, not display rate
        if is_new:
            self._fps_times.append(time.perf_counter())
        if len(self._fps_times) >= 2:
            fps = (len(self._fps_times) - 1) / \
                  (self._fps_times[-1] - self._fps_times[0] + 1e-9)
            cv2.putText(display, f"DET {fps:.1f}",
                        (display.shape[1] - 95, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw > 10 and ch > 10:
            src_h, src_w = display.shape[:2]
            scale   = min(cw / src_w, ch / src_h)
            new_w, new_h = int(src_w * scale), int(src_h * scale)
            resized = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            x0, y0  = (cw - new_w) // 2, (ch - new_h) // 2

            if self._padded is None or self._padded.shape[:2] != (ch, cw):
                self._padded = np.zeros((ch, cw, 3), dtype=np.uint8)
            else:
                if y0 > 0:
                    self._padded[:y0] = 0; self._padded[y0 + new_h:] = 0
                if x0 > 0:
                    self._padded[:, :x0] = 0; self._padded[:, x0 + new_w:] = 0

            self._padded[y0:y0 + new_h, x0:x0 + new_w] = resized
            self._lb = (scale, x0, y0)
            display  = self._padded

        t, gm = self.transformer, self.grid_mapper
        self.readout.config(text=(
            f"Margin  {t.margin}\n"
            f"Shift X {t.shift_x:+d}\n  Shift Y {t.shift_y:+d}\n"
            f"Off X   {gm.offset_x}\n  Off Y   {gm.offset_y}\n"
            f"Pitch X {gm.pitch_x:.1f}\n  Pitch Y {gm.pitch_y:.1f}"))

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        self.window.after(33, self._update_frame)

    def _update_circuit_ui(self, info: dict):
        t     = info['type']
        ohms  = info['total_ohms']
        extra = info.get('extra', {})
        color = _CIRCUIT_COLORS.get(t, DIM)

        self.circuit_type_lbl.config(text=t, fg=color)
        self.circuit_formula_lbl.config(text=info.get('formula', ''))

        if t in ('Wheatstone Bridge', 'Wheatstone Bridge (5R)') and extra.get('balanced') is not None:
            bal       = extra['balanced']
            bal_color = '#4ade80' if bal else '#f87171'
            bal_text  = '[OK] Balanced' if bal else '[!!] Unbalanced'
            req       = extra.get('req_ac', 0)
            if req > 0:
                if   req >= 1e6: r_str = f"{req/1e6:.2f} MOhms"
                elif req >= 1e3: r_str = f"{req/1e3:.2f} kOhms"
                else:             r_str = f"{req:.1f} Ohms"
                bal_text += f"\nRtotal={r_str}"
            self.circuit_ohms_lbl.config(text=bal_text, fg=bal_color)
        elif ohms > 0:
            if   ohms >= 1e6: s = f"= {ohms/1e6:.2f} MOhms"
            elif ohms >= 1e3: s = f"= {ohms/1e3:.2f} kOhms"
            else:              s = f"= {ohms:.1f} Ohms"
            self.circuit_ohms_lbl.config(text=s, fg=color)
        else:
            self.circuit_ohms_lbl.config(text='', fg=color)

    def on_closing(self):
        self._stop_event.set()
        configs.save_ui_state({k: v.get() for k, v in {
            "margin":  self.var_margin,  "shift_x": self.var_shift_x,
            "shift_y": self.var_shift_y, "off_x":   self.var_off_x,
            "off_y":   self.var_off_y,   "pitch_x": self.var_pitch_x,
            "pitch_y": self.var_pitch_y,
        }.items()})
        self.camera.stop()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    OhmVisionApp(root)
    root.mainloop()
    _calc_nodals = CircuitAnalyzer._calc_nodal_resistance
    _calc_nodals
