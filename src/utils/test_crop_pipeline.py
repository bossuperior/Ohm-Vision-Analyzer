"""
Test script: Run pose detection on test images → crop body → save results
Usage: python test_crop_pipeline.py [--img-dir PATH] [--out-dir PATH]
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

from config.configs import (POSE_BACKEND, POSE_MODEL, POSE_CONF, POSE_IOU,
                             CLS_RESISTOR)
from src.inference.model_engine import ModelEngine
from src.utils.crop_from_dataset import crop_body_for_classifier

IMG_DIR = 'data/processed/yolo-pose/images/test'
OUT_DIR = 'debug/test_crops'


def run(img_dir: str, out_dir: str) -> None:
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ModelEngine(POSE_BACKEND, POSE_MODEL,
                        conf=POSE_CONF, iou=POSE_IOU)

    images = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    saved = skipped = 0

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f'[skip] cannot read {img_path.name}')
            continue

        results = model.predict(frame)

        # วาด keypoints บนภาพต้นฉบับ (สำหรับ debug)
        annotated = frame.copy()

        found_any = False
        for idx, cls_id in enumerate(results.class_ids):
            if int(cls_id) != CLS_RESISTOR or idx >= len(results.keypoints):
                continue

            kps     = results.keypoints[idx]
            visible = [kp for kp in kps if kp[2] >= 0.5]
            if len(visible) < 2:
                print(f'  [{img_path.name}] r{idx}: visible kps={len(visible)} → skip')
                skipped += 1
                continue

            # วาด keypoints บน annotated frame
            for kp in visible:
                cv2.circle(annotated, (int(kp[0]), int(kp[1])), 4, (0, 255, 0), -1)
            p0, p1 = np.array(visible[0][:2]), np.array(visible[-1][:2])
            cv2.line(annotated, tuple(p0.astype(int)), tuple(p1.astype(int)),
                     (0, 200, 255), 2)

            crop = crop_body_for_classifier(frame, p0, p1)
            if crop is None:
                print(f'  [{img_path.name}] r{idx}: crop=None')
                skipped += 1
                continue

            stem = img_path.stem
            cv2.imwrite(str(out_dir / f'{stem}_r{idx}_crop.jpg'), crop)
            found_any = True
            saved += 1
            print(f'  [{img_path.name}] r{idx}: crop {crop.shape[1]}×{crop.shape[0]}px → saved')

        if found_any:
            cv2.imwrite(str(out_dir / f'{img_path.stem}_annot.jpg'), annotated)

    print(f'\nDone: saved={saved} crops, skipped={skipped}')
    print(f'Output: {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-dir', default=IMG_DIR)
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    run(args.img_dir, args.out_dir)
