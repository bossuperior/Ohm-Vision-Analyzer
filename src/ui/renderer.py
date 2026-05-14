import cv2
import numpy as np

# ── Box / keypoint color palette ──────────────────────────────────────────────
BOX_COLORS = {
    "resistor": (0, 200, 255),
    "wire":     (255, 180, 0),
}
DEFAULT_BOX_COLOR = (180, 180, 180)

RESISTOR_KP_COLORS = [
    (0, 0, 255),   # Leg_0
    (0, 0, 255),   # Leg_1
    (0, 255, 0),   # Body_2
    (0, 255, 0),   # Body_3
]


# ── Perspective helpers ───────────────────────────────────────────────────────
def transform_points(pts, matrix):
    if len(pts) == 0:
        return pts
    src = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(src, matrix).reshape(-1, 2)


def transform_box(box, matrix):
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1],
                        [x2, y2], [x1, y2]], dtype=np.float32)
    t = transform_points(corners, matrix)
    return (int(t[:, 0].min()), int(t[:, 1].min()),
            int(t[:, 0].max()), int(t[:, 1].max()))


# ── Main draw function ────────────────────────────────────────────────────────
def draw_results(frame, results, class_names, matrix=None, ohm_map=None):
    """Draw bounding boxes, keypoints, body line, and optional ohm readings."""
    for i, (box, cls_id, score) in enumerate(
            zip(results.boxes, results.class_ids, results.scores)):

        name  = class_names.get(int(cls_id), f"cls{int(cls_id)}")
        color = BOX_COLORS.get(name, DEFAULT_BOX_COLOR)

        if matrix is not None:
            x1, y1, x2, y2 = transform_box(box, matrix)
        else:
            x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {score:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Resistance label — bottom-center inside the bounding box
        if ohm_map is not None and i in ohm_map:
            ohm_txt  = ohm_map[i] or "?"
            good     = ohm_txt not in ("Unknown", "Error", "Read Error", "Calc Error", "?", "")
            txt_col  = (0, 230, 255) if good else (120, 120, 120)

            fscale = 0.55
            thick  = 2
            (tw, th), _ = cv2.getTextSize(ohm_txt, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)

            tx = max(x1 + 2, (x1 + x2) // 2 - tw // 2)
            ty = max(th + 6, min(y2 - 8, frame.shape[0] - 4))

            # Dark backing strip for legibility
            pad = 3
            bx1 = max(0, tx - pad)
            by1 = max(0, ty - th - pad)
            bx2 = min(frame.shape[1], tx + tw + pad)
            by2 = min(frame.shape[0], ty + pad)
            if bx2 > bx1 and by2 > by1:
                roi = frame[by1:by2, bx1:bx2]
                frame[by1:by2, bx1:bx2] = (roi * 0.3).astype(np.uint8)

            cv2.putText(frame, ohm_txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fscale, txt_col, thick)

        # Keypoints + body line
        if len(results.keypoints) > i:
            kp_data     = results.keypoints[i]
            is_resistor = (name == "resistor")
            raw_kps     = [(j, kp[:2]) for j, kp in enumerate(kp_data)
                           if kp[2] > 0.5]
            xy_only     = [kp for _, kp in raw_kps]

            if matrix is not None and xy_only:
                xy_only = transform_points(xy_only, matrix)

            kp_map = {j: tuple(map(int, xy_only[k]))
                      for k, (j, _) in enumerate(raw_kps)}

            for j, pt in kp_map.items():
                c = RESISTOR_KP_COLORS[j] \
                    if is_resistor and j < len(RESISTOR_KP_COLORS) \
                    else (0, 255, 255)
                cv2.circle(frame, pt, 5, c, -1)

            if is_resistor and 2 in kp_map and 3 in kp_map:
                cv2.line(frame, kp_map[2], kp_map[3], (0, 255, 0), 2)
