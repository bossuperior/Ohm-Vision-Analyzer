#!/usr/bin/env python3
"""
evaluate.py  —  Multi-model batch evaluation against ground_truth.json

Usage:
    cd /home/bossuperior/Desktop/Ohm-Vision-Analyzer
    source ohm-vision/bin/activate
    python -m src.inference.evaluate

Pose models tested (skipped automatically if file/library missing):
    1. YOLOv8n-pose  (ONNX fp32)
    2. YOLOv8n-pose  (ONNX int8 quantized)
    3. YOLOv8s-pose  (ONNX fp32)
    4. YOLOv8s-pose  (ONNX int8 quantized)
    5. RTMPose-s     (requires rtmlib + ONNX models)

Classification model: YOLOv8n-cls (fixed for all runs)

Outputs (in reports/<model_name>/):
    report.json, confusion_matrix.png, type_accuracy.png,
    ohm_scatter.png, error_distribution.png, summary_dashboard.png

Plus top-level:
    reports/model_comparison.png
    reports/type_model_matrix.png
"""
from __future__ import annotations
import sys, json, math, re
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.configs import CLS_BACKEND, CLS_MODEL, CLS_DEVICE
from src.inference.model_engine import ClassificationEngine
from src.vision.breadboard_warper import BreadboardWarper
from src.topology.grid_mapper import GridMapper
from src.topology.circuit_analyzer import CircuitAnalyzer
from src.utils.crop_from_dataset import crop_body_for_classifier

# ─────────────────────────────────────────────────────────────────────────────
# Pose model configurations
# ─────────────────────────────────────────────────────────────────────────────
POSE_CONFIGS: list[dict] = [
    {
        "name":       "YOLOv8n (fp32)",
        "backend":    "yolo",
        "model_path": ROOT / "models/Pose Model/Yolo_v8n/Yolo_v8n_pose_weights.onnx",
        "conf": 0.5, "iou": 0.45,
    },
    {
        "name":       "YOLOv8n (int8)",
        "backend":    "yolo",
        "model_path": ROOT / "models/Pose Model/Yolo_v8n/Yolo_v8n_pose_weights_int8.onnx",
        "conf": 0.5, "iou": 0.45,
    },
    {
        "name":       "YOLOv8s (fp32)",
        "backend":    "yolo",
        "model_path": ROOT / "models/Pose Model/Yolo_v8s/Yolo_v8s_pose_weights.onnx",
        "conf": 0.5, "iou": 0.45,
    },
    {
        "name":       "YOLOv8s (int8)",
        "backend":    "yolo",
        "model_path": ROOT / "models/Pose Model/Yolo_v8s/Yolo_v8s_pose_weights_int8.onnx",
        "conf": 0.5, "iou": 0.45,
    },
    {
        "name":       "RTMPose-s",
        "backend":    "rtmpose",
        "model_path": ROOT / "models/Pose Model/RTM_Pose/RTM_Pose_s.onnx",
        "det_model":  ROOT / "models/Pose Model/RTM_Pose/RTM_Pose_s_det.onnx",
        "device":     "cpu",
        "onnx_backend": "onnxruntime",
        "conf": 0.3,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
GT_PATH  = ROOT / "src/inference/ground_truth.json"
TEST_DIR = Path("/home/bossuperior/Desktop/test")
OUT_DIR  = ROOT / "reports"

CLS_RESISTOR = 0
CLS_WIRE     = 1

GT_TYPE_MAP = {
    "S":                       "Single",
    "Wheatstone (Unbalanced)": "Wheatstone Bridge",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ohms(label: str) -> float:
    m = re.match(r"([0-9.]+)(k|M)?\s+Ohm", label)
    if not m:
        return 0.0
    v, p = float(m.group(1)), m.group(2) or ""
    return v * (1e3 if p == "k" else 1e6 if p == "M" else 1.0)


def _cls_to_ohm_str(cls_name: str) -> str:
    parts = cls_name.split("_")
    if len(parts) != 2 or not parts[1].endswith("pct"):
        return cls_name
    val_s, tol = parts[0], parts[1].replace("pct", "%")
    for pat, mul in [(r"^(\d+)[Rr](\d*)$", 1), (r"^(\d+)[kK](\d*)$", 1e3), (r"^(\d+)[mM](\d*)$", 1e6)]:
        m = re.match(pat, val_s)
        if m:
            num = float(f"{m.group(1)}.{m.group(2) or '0'}") * mul
            if   num >= 1e6: return f"{num/1e6:g}M Ohm {tol}"
            elif num >= 1e3: return f"{num/1e3:g}k Ohm {tol}"
            else:             return f"{num:g} Ohm {tol}"
    return cls_name


def normalize_gt_type(gt_type: str) -> str:
    return GT_TYPE_MAP.get(gt_type, gt_type)


def find_image(gt_name: str, test_dir: Path) -> Path | None:
    exact = test_dir / gt_name
    if exact.exists():
        return exact
    candidates = sorted(test_dir.glob(f"{gt_name[:7]}*.jpg"))
    return candidates[0] if candidates else None


def _check_config(cfg: dict) -> str | None:
    mp = Path(cfg["model_path"])
    if not mp.exists():
        return f"model file not found: {mp.name}"
    if cfg["backend"] == "rtmpose":
        det = cfg.get("det_model")
        if det and not Path(det).exists():
            return f"det model not found: {Path(det).name}"
        try:
            import rtmlib  # noqa: F401
        except ImportError:
            return "rtmlib not installed  (pip install rtmlib)"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Voting parameters
# ─────────────────────────────────────────────────────────────────────────────
N_VOTES      = 10
VOTE_THRESH  = 6
TOPO_WINDOW  = 5
BAD_LABELS   = {"?", "ERR", "Unknown", "Read Error", "Calc Error", "Error", ""}

# ─────────────────────────────────────────────────────────────────────────────
# Pass/fail thresholds  (edit these to raise/lower the bar)
# ─────────────────────────────────────────────────────────────────────────────
TYPE_PASS_PCT = 80.0   # overall circuit-type accuracy must be ≥ this
OHM_PASS_PCT  = 70.0   # ohm accuracy (rated images only) must be ≥ this


def _augment(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out   = frame.copy()
    alpha = float(rng.uniform(0.90, 1.10))
    beta  = int(rng.integers(-12, 13))
    out   = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.30:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Single-model evaluator
# ─────────────────────────────────────────────────────────────────────────────

class OhmEvaluator:
    def __init__(self, pose_cfg: dict, cls_engine: ClassificationEngine,
                 warper: BreadboardWarper):
        from src.inference.model_engine import ModelEngine
        kwargs = {k: v for k, v in pose_cfg.items()
                  if k not in ("name", "backend", "model_path", "conf", "iou")}
        self.pose_engine = ModelEngine(
            pose_cfg["backend"], str(pose_cfg["model_path"]),
            conf=pose_cfg.get("conf", 0.5), iou=pose_cfg.get("iou", 0.45),
            **kwargs,
        )
        self.cls_engine       = cls_engine
        self.warper           = warper
        self.grid_mapper      = GridMapper(target_w=810, target_h=540)
        self.circuit_analyzer = CircuitAnalyzer()

    def _predict_once(self, base: np.ndarray, aruco_ok: bool) -> dict:
        from collections import Counter as _Counter

        self.grid_mapper._node_cache.clear()
        results = self.pose_engine.predict(base)

        if len(results.class_ids) == 0:
            return {"type": "Not Connected", "total_ohms": math.inf,
                    "aruco": aruco_ok, "ohm_cls": {}, "n_resistors": 0}

        ohm_cls      = {}
        ohm_numeric  = {}
        resistor_ids = set()
        wire_ids     = set()

        for idx, cls_id in enumerate(results.class_ids):
            if int(cls_id) == CLS_RESISTOR:
                resistor_ids.add(idx)
                if idx < len(results.keypoints):
                    kps     = results.keypoints[idx]
                    visible = [kp for kp in kps if kp[2] >= 0.5]
                    if len(visible) >= 2:
                        crop = crop_body_for_classifier(base, visible[0][:2], visible[-1][:2])
                        if crop is not None:
                            cls_name, _ = self.cls_engine.predict(crop)
                            ohm_cls[idx] = cls_name
                            if cls_name not in BAD_LABELS:
                                val = _parse_ohms(_cls_to_ohm_str(cls_name))
                                if val > 0:
                                    ohm_numeric[idx] = val
            elif int(cls_id) == CLS_WIRE:
                wire_ids.add(idx)

        if not resistor_ids:
            return {"type": "Not Connected", "total_ohms": math.inf,
                    "aruco": aruco_ok, "ohm_cls": {}, "n_resistors": 0}

        all_kp_data = [
            {"id": idx, "keypoints": kps}
            for idx, (_, kps) in enumerate(zip(results.class_ids, results.keypoints))
            if len(kps) >= 2
        ]
        all_mapped = self.grid_mapper.map_to_holes(all_kp_data)
        resistors  = self.circuit_analyzer.apply_wires(all_mapped, wire_ids)
        resistors  = [c for c in resistors if c["id"] in resistor_ids]
        for c in resistors:
            c["ohms"] = ohm_numeric.get(c["id"], 0.0)

        info = self.circuit_analyzer.analyze(resistors)
        return {
            "type":        info["type"],
            "total_ohms":  info["total_ohms"],
            "aruco":       aruco_ok,
            "ohm_cls":     ohm_cls,
            "n_resistors": len(resistor_ids),
        }

    def predict(self, img_path: Path) -> dict:
        from collections import Counter as _Counter

        frame = cv2.imread(str(img_path))
        if frame is None:
            return {"type": "Unknown", "total_ohms": 0.0, "aruco": False,
                    "vote_count": 0, "stable": False}

        rng         = np.random.default_rng(42)
        aruco_found = False

        type_votes: list[str]               = []
        ohm_cls_votes: dict[int, list]      = {}
        total_ohm_per_type: dict[str, list] = defaultdict(list)

        for i in range(N_VOTES):
            aug = _augment(frame, rng) if i > 0 else frame.copy()

            aruco_ok, base, _ = self.warper.process(aug)
            if aruco_ok:
                aruco_found = True
            else:
                base = cv2.resize(aug, (810, 540), interpolation=cv2.INTER_LINEAR)

            pr = self._predict_once(base, aruco_ok)
            type_votes.append(pr["type"])
            total_ohm_per_type[pr["type"]].append(pr["total_ohms"])

            for idx, lbl in pr["ohm_cls"].items():
                ohm_cls_votes.setdefault(idx, []).append(lbl)

        recent_votes   = type_votes[-TOPO_WINDOW:]
        topo_counter   = _Counter(type_votes)
        recent_counter = _Counter(recent_votes)
        voted_type, vote_count = topo_counter.most_common(1)[0]
        recent_top = recent_counter.most_common(1)[0][0]
        stable = (voted_type == recent_top) and (vote_count >= VOTE_THRESH)

        ohm_numeric_voted: dict[int, float] = {}
        for idx, labels in ohm_cls_votes.items():
            good_labels = [l for l in labels if l not in BAD_LABELS]
            if not good_labels:
                continue
            winner, freq = _Counter(good_labels).most_common(1)[0]
            if freq >= VOTE_THRESH:
                val = _parse_ohms(_cls_to_ohm_str(winner))
                if val > 0:
                    ohm_numeric_voted[idx] = val

        valid_totals = [
            v for v in total_ohm_per_type.get(voted_type, [])
            if v > 0 and not math.isinf(v)
        ]
        if valid_totals:
            voted_ohm = float(np.median(valid_totals))
        elif voted_type == "Not Connected":
            voted_ohm = math.inf
        else:
            voted_ohm = 0.0

        return {
            "type":        voted_type,
            "total_ohms":  voted_ohm,
            "aruco":       aruco_found,
            "vote_count":  vote_count,
            "stable":      stable,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Threshold check — shows what's dragging the score down
# ─────────────────────────────────────────────────────────────────────────────

def _print_failures(metrics: dict,
                    type_thresh: float = TYPE_PASS_PCT,
                    ohm_thresh:  float = OHM_PASS_PCT) -> None:
    name     = metrics["model_name"]
    type_acc = metrics["type_accuracy"]
    ohm_acc  = metrics["ohm_accuracy"]
    results  = metrics["results"]
    per_type = metrics["per_type"]

    type_pass = type_acc >= type_thresh
    ohm_pass  = ohm_acc  >= ohm_thresh
    overall   = type_pass and ohm_pass

    badge = "✓ PASS" if overall else "✗ FAIL"
    sep   = "─" * 70
    print(f"\n{sep}")
    print(f"  Threshold check — {name}")
    print(sep)
    print(f"  Type accuracy : {type_acc:6.1f}%  (need ≥{type_thresh:.0f}%)  "
          f"{'✓' if type_pass else '✗ BELOW THRESHOLD'}")
    print(f"  Ohm  accuracy : {ohm_acc:6.1f}%  (need ≥{ohm_thresh:.0f}%)  "
          f"{'✓' if ohm_pass else '✗ BELOW THRESHOLD'}")
    print(f"  Overall       : {badge}")

    if overall:
        print(sep)
        return

    # ── Per-type breakdown: show what's dragging the score ────────────────────
    print(f"\n  {'Circuit Type':<25} {'n':>4} {'Type Acc':>10} {'Ohm Acc':>10}  Status")
    print(f"  {'─'*65}")
    for t in sorted(per_type):
        d  = per_type[t]
        ta = d["type_ok"] / d["total"] * 100 if d["total"] else 0.0
        oa = (d["ohm_ok"] / d["ohm_rated"] * 100) if d["ohm_rated"] else None
        oa_s  = f"{oa:.1f}%" if oa is not None else "  —"
        flags = []
        if ta < type_thresh:
            flags.append(f"Type {ta:.0f}%<{type_thresh:.0f}%")
        if oa is not None and oa < ohm_thresh:
            flags.append(f"Ohm {oa:.0f}%<{ohm_thresh:.0f}%")
        flag_s = "  ← " + ", ".join(flags) if flags else ""
        print(f"  {t:<25} {d['total']:>4} {ta:>9.1f}% {oa_s:>10}{flag_s}")

    # ── Misclassified images ──────────────────────────────────────────────────
    type_wrong = [r for r in results if not r["type_ok"]]
    ohm_wrong  = [r for r in results if r["ohm_ok"] is False]

    if type_wrong:
        print(f"\n  Type misclassified ({len(type_wrong)} image(s)):")
        print(f"  {'Image':<44} {'GT':<20} {'Pred':<20}")
        print(f"  {'─'*84}")
        for r in type_wrong:
            print(f"  {r['name'][:43]:<44} {r['gt_type']:<20} {r['pred_type']:<20}")

    if ohm_wrong:
        print(f"\n  Ohm out of tolerance ({len(ohm_wrong)} image(s)):")
        print(f"  {'Image':<44} {'GT Ohm':>10} {'Pred Ohm':>10} {'Tol':>6}")
        print(f"  {'─'*72}")
        for r in ohm_wrong:
            gt_s   = "Inf" if math.isinf(r["gt_ohm"])   else f"{r['gt_ohm']:.1f}"
            pred_s = "Inf" if math.isinf(r["pred_ohm"]) else f"{r['pred_ohm']:.1f}"
            print(f"  {r['name'][:43]:<44} {gt_s:>10} {pred_s:>10} {r['tolerance']*100:>5.0f}%")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop for one pose model
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_model(pose_cfg: dict, ground_truth: dict,
                   cls_engine: ClassificationEngine,
                   warper: BreadboardWarper) -> dict:
    name = pose_cfg["name"]
    print(f"\n{'='*70}")
    print(f"  Model: {name}")
    print(f"{'='*70}")

    ev = OhmEvaluator(pose_cfg, cls_engine, warper)

    col = [44, 20, 20, 10, 10, 6, 6, 7, 7]
    hdr = (f"{'Image':<{col[0]}} {'GT Type':<{col[1]}} {'Pred Type':<{col[2]}}"
           f" {'GT Ohm':>{col[3]}} {'Pred Ohm':>{col[4]}} {'Typ':^{col[5]}} {'Ohm':^{col[6]}}"
           f" {'Vote':^{col[7]}} {'Stable':^{col[8]}}")
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    results: list[dict] = []
    for i, (gt_name, gt_val) in enumerate(ground_truth.items(), 1):
        img_path = find_image(gt_name, TEST_DIR)
        if img_path is None:
            continue

        print(f"  [{i:02d}/{len(ground_truth)}] {gt_name[:35]} …", end="\r", flush=True)
        pred = ev.predict(img_path)

        gt_type_norm = normalize_gt_type(gt_val["type"])
        pred_type    = pred["type"]
        gt_ohm       = math.inf if str(gt_val.get("total_ohm", gt_val.get("total_ohms", 0))) == "Infinity" \
                       else float(gt_val.get("total_ohm", gt_val.get("total_ohms", 0)))
        pred_ohm     = pred["total_ohms"]
        tolerance    = gt_val.get("tolerance", 0.05)
        type_ok      = (gt_type_norm == pred_type)
        vote_count   = pred.get("vote_count", 0)
        stable       = pred.get("stable", False)

        if tolerance == 0.0 and gt_ohm == 0.0:
            ohm_ok = None
        elif math.isinf(gt_ohm):
            ohm_ok = math.isinf(pred_ohm) or pred_ohm == 0.0
        elif pred_ohm <= 0 or math.isinf(pred_ohm):
            ohm_ok = False
        else:
            ohm_ok = abs(pred_ohm - gt_ohm) / gt_ohm <= tolerance

        results.append({
            "name": gt_name, "gt_type": gt_type_norm, "pred_type": pred_type,
            "gt_ohm": gt_ohm, "pred_ohm": pred_ohm, "tolerance": tolerance,
            "type_ok": type_ok, "ohm_ok": ohm_ok,
            "aruco": pred.get("aruco", False),
            "vote_count": vote_count, "stable": stable,
        })

        gt_s   = "Inf" if math.isinf(gt_ohm)   else f"{gt_ohm:.1f}"
        pred_s = "Inf" if math.isinf(pred_ohm) else f"{pred_ohm:.1f}"
        tok    = "OK" if type_ok else "X"
        ook    = ("OK" if ohm_ok else "X") if ohm_ok is not None else "—"
        stb    = "✓" if stable else "~"
        print(f"{gt_name[:col[0]-1]:<{col[0]}} {gt_type_norm:<{col[1]}} {pred_type:<{col[2]}}"
              f" {gt_s:>{col[3]}} {pred_s:>{col[4]}} {tok:^{col[5]}} {ook:^{col[6]}}"
              f" {vote_count:^{col[7]}} {stb:^{col[8]}}")

    n         = len(results)
    type_acc  = sum(r["type_ok"] for r in results) / n * 100 if n else 0.0
    ohm_rated = [r for r in results if r["ohm_ok"] is not None]
    ohm_acc   = sum(r["ohm_ok"] for r in ohm_rated) / len(ohm_rated) * 100 if ohm_rated else 0.0
    aruco_rate= sum(r["aruco"]   for r in results) / n * 100 if n else 0.0

    per_type: dict[str, dict] = defaultdict(lambda: {"total": 0, "type_ok": 0, "ohm_ok": 0, "ohm_rated": 0})
    for r in results:
        t = r["gt_type"]
        per_type[t]["total"]   += 1
        per_type[t]["type_ok"] += r["type_ok"]
        if r["ohm_ok"] is not None:
            per_type[t]["ohm_rated"] += 1
            per_type[t]["ohm_ok"]   += r["ohm_ok"]

    stable_count = sum(r["stable"] for r in results)
    print(sep)
    print(f"  Images: {n}  |  ArUco: {aruco_rate:.0f}%  |  "
          f"Stable votes: {stable_count}/{n} ({stable_count/n*100:.0f}%)")
    print(f"  Type acc: {type_acc:.1f}%  |  Ohm acc: {ohm_acc:.1f}% ({len(ohm_rated)} rated)")
    print(f"\n  {'Type':<25} {'n':>4} {'Type Acc':>10} {'Ohm Acc':>10}")
    print(f"  {'─'*51}")
    for t in sorted(per_type):
        d  = per_type[t]
        ta = d["type_ok"] / d["total"] * 100
        oa = (d["ohm_ok"] / d["ohm_rated"] * 100) if d["ohm_rated"] else float("nan")
        oa_s = f"{oa:.1f}%" if not math.isnan(oa) else "—"
        print(f"  {t:<25} {d['total']:>4} {ta:>9.1f}% {oa_s:>10}")

    return {
        "model_name":    name,
        "n_images":      n,
        "aruco_rate":    round(aruco_rate, 2),
        "type_accuracy": round(type_acc,   2),
        "ohm_accuracy":  round(ohm_acc,    2),
        "per_type":      dict(per_type),
        "results":       results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-type × all-models comparison (console table + chart)
# ─────────────────────────────────────────────────────────────────────────────

def _print_type_model_matrix(all_metrics: list[dict]):
    """Print a per-type × per-model accuracy matrix to console."""
    all_types  = sorted({t for m in all_metrics for t in m["per_type"]})
    model_names = [m["model_name"] for m in all_metrics]
    cell_w = 20

    print(f"\n{'='*70}")
    print("  Per-Type × All-Models Accuracy Matrix")
    print(f"{'='*70}")
    # header
    hdr = f"  {'Circuit Type':<22}"
    for name in model_names:
        short = name[:18]
        hdr += f"  {short:^{cell_w}}"
    print(hdr)
    sub = f"  {'':22}"
    for _ in model_names:
        sub += f"  {'Type%  Ohm%':^{cell_w}}"
    print(sub)
    print(f"  {'─'*22}" + f"  {'─'*cell_w}" * len(model_names))

    for ctype in all_types:
        row = f"  {ctype:<22}"
        for m in all_metrics:
            pt = m["per_type"].get(ctype, {})
            if not pt.get("total"):
                row += f"  {'—':^{cell_w}}"
                continue
            ta  = pt["type_ok"] / pt["total"] * 100
            oa  = (pt["ohm_ok"] / pt["ohm_rated"] * 100) if pt.get("ohm_rated") else float("nan")
            oa_s = f"{oa:.0f}%" if not math.isnan(oa) else "—"
            cell = f"{ta:.0f}%  {oa_s}"
            row += f"  {cell:^{cell_w}}"
        print(row)

    # summary row
    print(f"  {'─'*22}" + f"  {'─'*cell_w}" * len(model_names))
    row = f"  {'OVERALL':<22}"
    for m in all_metrics:
        cell = f"{m['type_accuracy']:.1f}%  {m['ohm_accuracy']:.1f}%"
        row += f"  {cell:^{cell_w}}"
    print(row)
    print()


def _save_type_model_comparison(all_metrics: list[dict], out_dir: Path):
    """Save per-type × per-model heatmap chart."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    all_types   = sorted({t for m in all_metrics for t in m["per_type"]})
    model_names = [m["model_name"] for m in all_metrics]
    n_types     = len(all_types)
    n_models    = len(model_names)

    # Build matrices: type_mat[model_idx][type_idx], ohm_mat same
    type_mat = np.full((n_models, n_types), np.nan)
    ohm_mat  = np.full((n_models, n_types), np.nan)
    for mi, m in enumerate(all_metrics):
        for ti, ctype in enumerate(all_types):
            pt = m["per_type"].get(ctype, {})
            if pt.get("total"):
                type_mat[mi, ti] = pt["type_ok"] / pt["total"] * 100
            if pt.get("ohm_rated"):
                ohm_mat[mi, ti] = pt["ohm_ok"] / pt["ohm_rated"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_types * 1.5 + 4), max(5, n_models * 0.9 + 3)))
    fig.patch.set_facecolor("#f8fafc")

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=100)

    for ax, mat, title in zip(axes, [type_mat, ohm_mat],
                               ["Type Accuracy (%)", "Ohm Accuracy (%)"]):
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(n_types))
        ax.set_xticklabels(all_types, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(model_names, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        for mi in range(n_models):
            for ti in range(n_types):
                v = mat[mi, ti]
                txt = f"{v:.0f}%" if not np.isnan(v) else "—"
                ax.text(ti, mi, txt, ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if (not np.isnan(v) and v < 50) else "black")
        plt.colorbar(im, ax=ax, shrink=0.85, label="%")

    fig.suptitle("Ohm-Vision — Per-Type × Model Accuracy Matrix",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = out_dir / "type_model_matrix.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Charts (per-model)
# ─────────────────────────────────────────────────────────────────────────────

def _save_charts(metrics: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(metrics["results"], out_dir)
    _plot_type_accuracy(metrics["per_type"], metrics["type_accuracy"], metrics["ohm_accuracy"], out_dir)
    _plot_ohm_scatter(metrics["results"], out_dir)
    _plot_error_distribution(metrics["results"], out_dir)
    _plot_dashboard(metrics, out_dir)


def _plot_confusion_matrix(results, out_dir):
    import matplotlib.pyplot as plt

    all_labels = sorted(set(r["gt_type"] for r in results) | set(r["pred_type"] for r in results))
    idx = {t: i for i, t in enumerate(all_labels)}
    n   = len(all_labels)
    mat = np.zeros((n, n), dtype=int)
    for r in results:
        gi = idx.get(r["gt_type"],   -1)
        pi = idx.get(r["pred_type"], -1)
        if gi >= 0 and pi >= 0:
            mat[gi][pi] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(all_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(all_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual (Ground Truth)", fontsize=11)
    ax.set_title("Circuit Type — Confusion Matrix", fontsize=13, fontweight="bold")
    thresh = mat.max() * 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, mat[i][j], ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if mat[i][j] > thresh else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def _plot_type_accuracy(per_type, type_acc, ohm_acc, out_dir):
    import matplotlib.pyplot as plt

    types  = sorted(per_type)
    ta     = [per_type[t]["type_ok"] / per_type[t]["total"] * 100 for t in types]
    oa     = [(per_type[t]["ohm_ok"] / per_type[t]["ohm_rated"] * 100)
              if per_type[t]["ohm_rated"] else 0 for t in types]
    counts = [per_type[t]["total"] for t in types]

    x, w = np.arange(len(types)), 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w / 2, ta, w, label="Type Accuracy", color="#3b82f6", alpha=0.85)
    b2 = ax.bar(x + w / 2, oa, w, label="Ohm Accuracy",  color="#10b981", alpha=0.85)
    for bar, v in list(zip(b1, ta)) + list(zip(b2, oa)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={c})" for t, c in zip(types, counts)], fontsize=9)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"Per-Type Accuracy  |  Type {type_acc:.1f}%  Ohm {ohm_acc:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.axhline(type_acc, color="#3b82f6", linestyle="--", alpha=0.5, linewidth=1.2)
    ax.axhline(ohm_acc,  color="#10b981", linestyle="--", alpha=0.5, linewidth=1.2)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "type_accuracy.png", dpi=150)
    plt.close()


def _plot_ohm_scatter(results, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    finite = [(r["gt_ohm"], r["pred_ohm"], r["type_ok"]) for r in results
              if not math.isinf(r["gt_ohm"]) and r["gt_ohm"] > 0
              and not math.isinf(r["pred_ohm"]) and r["pred_ohm"] > 0]
    if not finite:
        return

    gts, preds, tok = zip(*finite)
    gts, preds = np.array(gts, dtype=float), np.array(preds, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(gts, preds, c=["#22c55e" if t else "#ef4444" for t in tok],
               alpha=0.78, s=60, edgecolors="white", linewidths=0.6)
    lo = min(gts.min(), preds.min()) * 0.7
    hi = max(gts.max(), preds.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.3)
    ax.plot([lo, hi], [lo * 1.05, hi * 1.05], color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.plot([lo, hi], [lo * 0.95, hi * 0.95], color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Ground Truth (Ω)", fontsize=11)
    ax.set_ylabel("Predicted (Ω)",    fontsize=11)
    ax.set_title("Predicted vs. Ground Truth Resistance", fontsize=12, fontweight="bold")
    legend_elems = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#22c55e", markersize=9, label="Type correct"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#ef4444", markersize=9, label="Type wrong"),
        Line2D([0],[0], color="k", linestyle="--", label="y = x"),
        Line2D([0],[0], color="gray", linestyle=":", label="±5% band"),
    ]
    ax.legend(handles=legend_elems, fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / "ohm_scatter.png", dpi=150)
    plt.close()


def _plot_error_distribution(results, out_dir):
    import matplotlib.pyplot as plt

    errors = np.array([
        (r["pred_ohm"] - r["gt_ohm"]) / r["gt_ohm"] * 100
        for r in results
        if not math.isinf(r["gt_ohm"]) and r["gt_ohm"] > 0
        and not math.isinf(r["pred_ohm"]) and r["pred_ohm"] > 0
    ])
    if not len(errors):
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors, bins=20, color="#6366f1", alpha=0.82, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.3, label="Zero error")
    ax.axvline(errors.mean(), color="#f59e0b", linestyle="-", linewidth=1.6,
               label=f"Mean = {errors.mean():.1f}%")
    ax.axvline(np.median(errors), color="#ec4899", linestyle="-", linewidth=1.6,
               label=f"Median = {np.median(errors):.1f}%")
    ax.set_xlabel("Relative Error (%)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Ohm Value — Error Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    props = dict(boxstyle="round", facecolor="#e0e7ff", alpha=0.85)
    ax.text(0.97, 0.97, f"n={len(errors)}\nσ={errors.std():.1f}%\nMAE={np.abs(errors).mean():.1f}%",
            transform=ax.transAxes, fontsize=9, va="top", ha="right", bbox=props)
    plt.tight_layout()
    fig.savefig(out_dir / "error_distribution.png", dpi=150)
    plt.close()


def _plot_dashboard(metrics: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    results    = metrics["results"]
    per_type   = metrics["per_type"]
    type_acc   = metrics["type_accuracy"]
    ohm_acc    = metrics["ohm_accuracy"]
    aruco_rate = metrics["aruco_rate"]
    n          = metrics["n_images"]
    model_name = metrics["model_name"]

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#f8fafc")
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.48)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("#eff6ff"); ax0.axis("off")
    for i, (label, val, color) in enumerate([
        ("Images Tested",  str(n),              "#1e40af"),
        ("ArUco Detect",   f"{aruco_rate:.0f}%","#0369a1"),
        ("Type Accuracy",  f"{type_acc:.1f}%",  "#047857"),
        ("Ohm  Accuracy",  f"{ohm_acc:.1f}%",   "#7c3aed"),
    ]):
        y = 0.87 - i * 0.23
        ax0.text(0.5, y,       val,   transform=ax0.transAxes, ha="center", va="center",
                 fontsize=23, fontweight="bold", color=color)
        ax0.text(0.5, y - 0.09, label, transform=ax0.transAxes, ha="center", va="center",
                 fontsize=9, color="#374151")
    ax0.set_title("Overall Results", fontsize=11, fontweight="bold", pad=8)

    ax1 = fig.add_subplot(gs[0, 1])
    ok_t = sum(r["type_ok"] for r in results)
    ax1.pie([ok_t, n - ok_t], labels=[f"Correct\n{ok_t}", f"Wrong\n{n - ok_t}"],
            colors=["#22c55e", "#ef4444"], autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 10})
    ax1.set_title("Circuit Type Classification", fontsize=11, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 2])
    ohm_rated = [r for r in results if r["ohm_ok"] is not None]
    ok_o = sum(r["ohm_ok"] for r in ohm_rated)
    ax2.pie([ok_o, len(ohm_rated) - ok_o],
            labels=[f"Within tol.\n{ok_o}", f"Outside\n{len(ohm_rated) - ok_o}"],
            colors=["#3b82f6", "#f59e0b"], autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 10})
    ax2.set_title(f"Ohm Accuracy  ({len(ohm_rated)} rated)", fontsize=11, fontweight="bold")

    ax3 = fig.add_subplot(gs[1, :])
    types  = sorted(per_type)
    ta     = [per_type[t]["type_ok"] / per_type[t]["total"] * 100 for t in types]
    oa     = [(per_type[t]["ohm_ok"] / per_type[t]["ohm_rated"] * 100)
              if per_type[t]["ohm_rated"] else 0 for t in types]
    counts = [per_type[t]["total"] for t in types]
    x, w   = np.arange(len(types)), 0.38
    b1 = ax3.bar(x - w / 2, ta, w, label="Type Acc", color="#3b82f6", alpha=0.85)
    b2 = ax3.bar(x + w / 2, oa, w, label="Ohm Acc",  color="#10b981", alpha=0.85)
    for bar, v in list(zip(b1, ta)) + list(zip(b2, oa)):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{t}\n(n={c})" for t, c in zip(types, counts)], fontsize=9)
    ax3.set_ylim(0, 122); ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Per Circuit-Type Accuracy", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Ohm-Vision — Evaluation Report  [{model_name}]",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(out_dir / "summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_model_comparison(all_metrics: list[dict], out_dir: Path):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # ── Thai font setup ────────────────────────────────────────────────────────
    thai_candidates = ["Sarabun", "Noto Sans Thai", "TH Sarabun New",
                       "Garuda", "Loma", "Tlwg Typewriter"]
    chosen_font = None
    for fc in thai_candidates:
        if any(fc.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            chosen_font = fc
            break
    if chosen_font:
        plt.rcParams["font.family"] = chosen_font
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linestyle": "--", "grid.color": "#cbd5e1"})

    # ── Thai label maps ────────────────────────────────────────────────────────
    TYPE_TH = {
        "Not Connected":     "ไม่เชื่อมต่อ",
        "Series":            "อนุกรม",
        "Parallel":          "ขนาน",
        "Mixed":             "ผสม",
        "Single":            "ตัวเดียว",
        "Wheatstone Bridge": "วีทสโตนบริดจ์",
    }
    MODEL_SHORT = {
        "YOLOv8n (fp32)":  "v8n-fp32",
        "YOLOv8n (int8)":  "v8n-int8",
        "YOLOv8s (fp32)":  "v8s-fp32",
        "YOLOv8s (int8)":  "v8s-int8",
    }

    names     = [m["model_name"] for m in all_metrics]
    short_names = [MODEL_SHORT.get(n, n) for n in names]
    type_acc  = [m["type_accuracy"] for m in all_metrics]
    all_types = sorted({t for m in all_metrics for t in m["per_type"]})
    type_labels = [TYPE_TH.get(t, t) for t in all_types]

    PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444",
               "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]
    model_colors = [PALETTE[i % len(PALETTE)] for i in range(len(all_metrics))]

    best_idx = int(np.argmax(type_acc))

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16, 7),
        gridspec_kw={"width_ratios": [1, 1.8]},
    )
    fig.patch.set_facecolor("#ffffff")

    # ── กราฟซ้าย: ความแม่นยำรวมแต่ละโมเดล ─────────────────────────────────────
    x = np.arange(len(names))
    bars = ax_left.bar(x, type_acc, color=model_colors, width=0.55,
                       edgecolor="white", linewidth=1.2, zorder=3)
    for bi, (bar, v) in enumerate(zip(bars, type_acc)):
        bar.set_alpha(1.0 if bi == best_idx else 0.78)
        ax_left.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1.5,
                     f"{v:.1f}%",
                     ha="center", va="bottom",
                     fontsize=11, fontweight="bold",
                     color="#1e293b")

    # เส้นเป้าหมาย 80%
    ax_left.axhline(80, color="#ef4444", linewidth=1.4, linestyle="--", zorder=2)
    ax_left.text(len(names) - 0.45, 81.5, "เป้าหมาย 80%",
                 fontsize=9, color="#ef4444", ha="right")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(short_names, fontsize=11)
    ax_left.set_ylim(0, 115)
    ax_left.set_ylabel("ความแม่นยำ (%)", fontsize=12, labelpad=8)
    ax_left.set_title("ภาพรวมความแม่นยำในการจำแนกประเภทวงจร\nเปรียบเทียบระหว่างโมเดล",
                      fontsize=13, fontweight="bold", pad=14, color="#0f172a")
    ax_left.tick_params(axis="both", labelsize=10)

    # ── กราฟขวา: ความแม่นยำต่อประเภทวงจร ──────────────────────────────────────
    n_models = len(all_metrics)
    n_types  = len(all_types)
    total_w  = 0.75
    bar_w    = total_w / n_models

    for mi, m in enumerate(all_metrics):
        vals = []
        for t in all_types:
            pt  = m["per_type"].get(t, {})
            tot = pt.get("total", 0)
            vals.append(pt["type_ok"] / tot * 100 if tot else 0)
        xs   = np.arange(n_types) + (mi - n_models / 2 + 0.5) * bar_w
        alpha = 1.0 if mi == best_idx else 0.78
        ax_right.bar(xs, vals, bar_w,
                     label=short_names[mi],
                     color=model_colors[mi],
                     alpha=alpha,
                     edgecolor="white",
                     linewidth=0.8,
                     zorder=3)
        for xi, v in zip(xs, vals):
            if v > 0:
                ax_right.text(xi, v + 1.2, f"{v:.0f}",
                              ha="center", va="bottom",
                              fontsize=7.5, color="#374151")

    ax_right.set_xticks(range(n_types))
    ax_right.set_xticklabels(type_labels, fontsize=11, rotation=15, ha="right")
    ax_right.set_ylim(0, 120)
    ax_right.set_ylabel("ความแม่นยำ (%)", fontsize=12, labelpad=8)
    ax_right.set_title("ความแม่นยำในการจำแนกประเภทวงจรแยกตามประเภท\nเปรียบเทียบระหว่างโมเดล",
                       fontsize=13, fontweight="bold", pad=14, color="#0f172a")
    ax_right.legend(fontsize=10, loc="upper right",
                    framealpha=0.9, edgecolor="#e2e8f0")
    ax_right.tick_params(axis="both", labelsize=10)

    fig.suptitle("ผลการประเมินประสิทธิภาพโมเดลตรวจวัดวงจรไฟฟ้า  —  Ohm-Vision",
                 fontsize=15, fontweight="bold", y=1.02, color="#0f172a")
    fig.tight_layout(pad=2.5)

    out_path = out_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate():
    import matplotlib
    matplotlib.use("Agg")

    with open(GT_PATH) as f:
        ground_truth: dict = json.load(f)

    print("Loading shared resources …")
    cls_engine = ClassificationEngine(
        backend=CLS_BACKEND,
        model_path=str(ROOT / CLS_MODEL),
        device=CLS_DEVICE,
    )
    warper = BreadboardWarper(output_width=810, output_height=540)
    print("Classification engine ready.\n")

    all_metrics: list[dict] = []

    for cfg in POSE_CONFIGS:
        skip_reason = _check_config(cfg)
        if skip_reason:
            print(f"[SKIP] {cfg['name']}  —  {skip_reason}")
            continue

        metrics = _run_one_model(cfg, ground_truth, cls_engine, warper)
        all_metrics.append(metrics)
        _print_failures(metrics)

        safe = cfg["name"].replace(" ", "_").replace("(", "").replace(")", "")
        model_out = OUT_DIR / safe
        _save_charts(metrics, model_out)
        print(f"  Charts → {model_out}/")

        def _jv(v):
            return "Infinity" if isinstance(v, float) and math.isinf(v) else v

        report = {
            "model_name":    metrics["model_name"],
            "n_images":      metrics["n_images"],
            "aruco_rate":    metrics["aruco_rate"],
            "type_accuracy": metrics["type_accuracy"],
            "ohm_accuracy":  metrics["ohm_accuracy"],
            "per_type": {
                t: {
                    "count":    d["total"],
                    "type_acc": round(d["type_ok"] / d["total"] * 100, 2),
                    "ohm_acc":  round(d["ohm_ok"] / d["ohm_rated"] * 100, 2) if d["ohm_rated"] else None,
                }
                for t, d in metrics["per_type"].items()
            },
            "details": [{k: _jv(v) for k, v in r.items()} for r in metrics["results"]],
        }
        with open(model_out / "report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    if not all_metrics:
        print("\nNo models were evaluated.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_model_comparison(all_metrics, OUT_DIR)

    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON  ({len(all_metrics)} model(s) evaluated)")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Type Acc':>10} {'Ohm Acc':>10} {'ArUco':>8}")
    print(f"  {'─'*55}")
    for m in all_metrics:
        print(f"  {m['model_name']:<25} {m['type_accuracy']:>9.1f}% {m['ohm_accuracy']:>9.1f}% {m['aruco_rate']:>7.0f}%")

    best_type = max(all_metrics, key=lambda m: m["type_accuracy"])
    best_ohm  = max(all_metrics, key=lambda m: m["ohm_accuracy"])
    print(f"\n  Best Type Acc : {best_type['model_name']}  ({best_type['type_accuracy']:.1f}%)")
    print(f"  Best Ohm  Acc : {best_ohm['model_name']}  ({best_ohm['ohm_accuracy']:.1f}%)")
    print(f"\n  All outputs → {OUT_DIR}/")

    # Per-type × all-models matrix (console + chart)
    _print_type_model_matrix(all_metrics)
    _save_type_model_comparison(all_metrics, OUT_DIR)

    # Save combined CSV
    csv_path = _save_results_csv(all_metrics, OUT_DIR)
    print(f"\n  CSV → {csv_path}")


def _save_results_csv(all_metrics: list[dict], out_dir: Path) -> Path:
    import csv as _csv

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "evaluation_results.csv"

    model_names = [m["model_name"] for m in all_metrics]

    # Index each model's results by image name for O(1) lookup
    by_model: list[dict[str, dict]] = [
        {r["name"]: r for r in m["results"]} for m in all_metrics
    ]

    # Union of all image names, in GT order (use first model's result order)
    all_names: list[str] = []
    seen: set[str] = set()
    for m in all_metrics:
        for r in m["results"]:
            if r["name"] not in seen:
                all_names.append(r["name"])
                seen.add(r["name"])

    base_fields = ["image", "gt_type", "gt_ohm", "tolerance"]
    model_fields: list[str] = []
    for lbl in model_names:
        safe = lbl.replace(" ", "_").replace("(", "").replace(")", "")
        model_fields += [
            f"{safe}_pred_type", f"{safe}_pred_ohm",
            f"{safe}_type_ok",   f"{safe}_ohm_ok",
            f"{safe}_aruco",     f"{safe}_vote",    f"{safe}_stable",
        ]
    summary_fields = ["models_agree_type", "any_type_ok", "any_ohm_ok"]

    fieldnames = base_fields + model_fields + summary_fields

    rows: list[dict] = []
    for name in all_names:
        # Pull GT from whichever model has this image
        gt_type = gt_ohm_raw = tolerance = None
        for m_results in by_model:
            if name in m_results:
                r0 = m_results[name]
                gt_type    = r0["gt_type"]
                gt_ohm_raw = r0["gt_ohm"]
                tolerance  = r0["tolerance"]
                break
        if gt_type is None:
            continue

        gt_ohm_s = "Infinity" if math.isinf(gt_ohm_raw) else f"{gt_ohm_raw:.4f}"

        row: dict = {
            "image":     name,
            "gt_type":   gt_type,
            "gt_ohm":    gt_ohm_s,
            "tolerance": f"{tolerance*100:.0f}%",
        }

        pred_types: list[str] = []
        any_type_ok = False
        any_ohm_ok  = False

        for m_idx, lbl in enumerate(model_names):
            safe = lbl.replace(" ", "_").replace("(", "").replace(")", "")
            r = by_model[m_idx].get(name)
            if r is None:
                for key in [f"{safe}_pred_type", f"{safe}_pred_ohm",
                            f"{safe}_type_ok", f"{safe}_ohm_ok",
                            f"{safe}_aruco", f"{safe}_vote", f"{safe}_stable"]:
                    row[key] = ""
                continue

            pred_ohm_s = "Infinity" if math.isinf(r["pred_ohm"]) else f"{r['pred_ohm']:.4f}"
            row[f"{safe}_pred_type"] = r["pred_type"]
            row[f"{safe}_pred_ohm"]  = pred_ohm_s
            row[f"{safe}_type_ok"]   = "Yes" if r["type_ok"] else "No"
            row[f"{safe}_ohm_ok"]    = ("Yes" if r["ohm_ok"] else "No") if r["ohm_ok"] is not None else "—"
            row[f"{safe}_aruco"]     = "Yes" if r["aruco"]   else "No"
            row[f"{safe}_vote"]      = r.get("vote_count", "")
            row[f"{safe}_stable"]    = "Yes" if r.get("stable") else "No"

            pred_types.append(r["pred_type"])
            if r["type_ok"]:
                any_type_ok = True
            if r["ohm_ok"] is True:
                any_ohm_ok = True

        row["models_agree_type"] = "Yes" if len(set(pred_types)) == 1 else "No"
        row["any_type_ok"]       = "Yes" if any_type_ok else "No"
        row["any_ohm_ok"]        = "Yes" if any_ohm_ok else "No"
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return csv_path


if __name__ == "__main__":
    evaluate()
