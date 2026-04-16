import cv2
import numpy as np


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_pts = pts[np.argsort(angles)]

    s = sorted_pts.sum(axis=1)
    tl_index = np.argmin(s)

    # Returns [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    return np.roll(sorted_pts, -tl_index, axis=0)


def img_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate actual pixel dimensions
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    max_width = max(int(width_top), int(width_bottom))
    max_height = max(int(height_left), int(height_right))
    target_w, target_h = 810, 540

    if max_height > max_width:
        # The physical board is VERTICAL.
        # We map the corners to automatically rotate it 90-degrees clockwise into a flat landscape.
        dst_pts = np.array([
            [target_w - 1, 0],  # Top-Left point moves to Top-Right
            [target_w - 1, target_h - 1],  # Top-Right point moves to Bottom-Right
            [0, target_h - 1],  # Bottom-Right point moves to Bottom-Left
            [0, 0]  # Bottom-Left point moves to Top-Left
        ], dtype="float32")
    else:
        # The physical board is already HORIZONTAL. Standard mapping.
        dst_pts = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype="float32")

    # Apply the transform. The output is now guaranteed to be 810x540 and perfectly straight.
    matrix = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (target_w, target_h))

    return warped, matrix