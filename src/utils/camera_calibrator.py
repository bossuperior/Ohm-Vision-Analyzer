import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import numpy as np
import glob

CHESSBOARD = (9, 6)   # inner corners (cols, rows) — adjust to your printed pattern
SAVE_PATH  = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'camera_calibration.npz')


def capture_calibration_images(camera_id=1, save_dir='data/raw/calibration'):
    """Live capture: press SPACE to save frame, Q to finish."""
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    count = 0
    print(f"Point camera at checkerboard ({CHESSBOARD[0]}x{CHESSBOARD[1]} inner corners)")
    print("SPACE = save frame  |  Q = finish capture")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, CHESSBOARD, corners, found)
            cv2.putText(display, "DETECTED - press SPACE", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(display, "not found", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(display, f"Saved: {count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Camera Calibration Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            path = os.path.join(save_dir, f"calib_{count:03d}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            print(f"Saved {path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done — {count} images saved to {save_dir}")
    return save_dir


def compute_calibration(image_dir='data/raw/calibration'):
    """Compute K, D from saved images and write to models/camera_calibration.npz."""
    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

    obj_pts, img_pts = [], []
    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    if not images:
        print(f"No images found in {image_dir}")
        return

    img_shape = None
    for path in images:
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners2)
            print(f"  OK: {os.path.basename(path)}")
        else:
            print(f"  SKIP (corners not found): {os.path.basename(path)}")

    if len(obj_pts) < 10:
        print(f"Need at least 10 valid images, got {len(obj_pts)}")
        return

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_shape, None, None)
    print(f"\nRMS reprojection error: {ret:.4f}  (< 0.5 is good)")
    print(f"K =\n{K}")
    print(f"D = {D.ravel()}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez(SAVE_PATH, K=K, D=D)
    print(f"\nCalibration saved to {SAVE_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['capture', 'compute', 'both'], default='both')
    parser.add_argument('--dir',  default='data/raw/calibration')
    parser.add_argument('--camera', type=int, default=1)
    args = parser.parse_args()

    if args.mode in ('capture', 'both'):
        capture_calibration_images(camera_id=args.camera, save_dir=args.dir)
    if args.mode in ('compute', 'both'):
        compute_calibration(image_dir=args.dir)
