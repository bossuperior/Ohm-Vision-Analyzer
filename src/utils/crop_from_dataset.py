"""
crop_from_dataset.py
────────────────────
อ่าน COCO JSON จาก Roboflow dataset แล้ว crop resistor body
ออกมาเก็บใน crops/unsorted/ สำหรับนำไปจัดเป็น classifier dataset

Usage:
    python -m src.utils.crop_from_dataset \
        --json  data/processed/yolo-pose/images/train/_annotations.coco.json \
        --imgs  data/processed/yolo-pose/images/train \
        --out   crops/unsorted

หลังรัน:
    เปิด crops/unsorted/ แล้วย้ายแต่ละ crop ไปไว้ใน
    data/classify/train/<class_name>/  ตามค่าความต้านทานจริง
"""

import json
import argparse
import cv2
import numpy as np
from pathlib import Path


# Class names ใน COCO ที่เป็น resistor body
_BODY_CLASSES = {'resistor_4b', 'resistor_5b'}


def _affine_crop(img: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray | None:
    """Affine-crop ตาม axis body (logic เดียวกับ BandReader._affine_crop)"""
    p0, p1 = p0.astype(float), p1.astype(float)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = float(np.hypot(dx, dy))
    if length < 10:
        return None

    cx, cy   = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    cos_a, sin_a = dx / length, dy / length
    perp_pad = int(np.clip(length * 0.20, 12, 30))

    out_w = int(length) + 60
    out_h = perp_pad * 2

    tx = cx - (out_w / 2) * cos_a + (out_h / 2) * sin_a
    ty = cy - (out_w / 2) * sin_a - (out_h / 2) * cos_a
    M  = np.float32([[cos_a, -sin_a, tx],
                     [sin_a,  cos_a, ty]])

    crop = cv2.warpAffine(img, M, (out_w, out_h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return crop if crop.size > 0 else None


def crop_from_coco(json_path: str, img_dir: str, out_dir: str) -> None:
    json_path = Path(json_path)
    img_dir   = Path(img_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    # Build lookup tables
    cat_id_to_name = {c['id']: c['name'].strip().lower()
                      for c in data['categories']}
    img_id_to_info = {img['id']: img for img in data['images']}

    saved = skipped = 0

    for ann in data['annotations']:
        cat_name = cat_id_to_name.get(ann['category_id'], '')
        if cat_name not in _BODY_CLASSES:
            continue

        img_info  = img_id_to_info[ann['image_id']]
        img_file  = img_dir / img_info['file_name']
        if not img_file.exists():
            print(f"[skip] image not found: {img_file}")
            skipped += 1
            continue

        # Keypoints: [x0,y0,v0, x1,y1,v1, ...]
        kpts = ann.get('keypoints', [])
        if len(kpts) < 6:
            skipped += 1
            continue

        # ใช้เฉพาะ kp0 และ kp1 (body endpoints)
        # ข้าม kp ที่ visibility = 0 (not labeled)
        kp_pairs = []
        for i in range(0, min(len(kpts), 6), 3):
            x, y, v = kpts[i], kpts[i+1], kpts[i+2]
            if v > 0:
                kp_pairs.append(np.array([x, y]))

        if len(kp_pairs) < 2:
            skipped += 1
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            skipped += 1
            continue

        crop = _affine_crop(img, kp_pairs[0], kp_pairs[-1])
        if crop is None:
            skipped += 1
            continue

        # ชื่อไฟล์: <image_stem>_ann<ann_id>_<class>.jpg
        stem     = Path(img_info['file_name']).stem
        out_name = f"{stem}_ann{ann['id']}_{cat_name}.jpg"
        cv2.imwrite(str(out_dir / out_name), crop)
        saved += 1

    print(f"Saved  : {saved} crops → {out_dir}")
    print(f"Skipped: {skipped} annotations")
    print(f"\nNext step: เปิด {out_dir} แล้วย้าย crop ไปใน data/classify/train/<class_name>/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='path to _annotations.coco.json')
    ap.add_argument('--imgs', required=True, help='folder containing images')
    ap.add_argument('--out',  default='crops/unsorted', help='output folder')
    args = ap.parse_args()
    crop_from_coco(args.json, args.imgs, args.out)


if __name__ == '__main__':
    OUT_DIR  = r"./data/crops"
    LABEL_DIR  = r"./data/labels/coco_keypoint"
    TRAIN_DIR = r"./data/processed/yolo-pose/images/train"
    # คง split เดิมที่ Roboflow แบ่งไว้แล้ว → crop แต่ละ split ออกแยกกัน
    SPLITS = [
        {
            'json': 'data/processed/yolo-pose/images/train/_annotations.coco.json',
            'imgs': 'data/processed/yolo-pose/images/train',
            'out':  'crops/train',
        },
        {
            'json': 'data/labels/coco_keypoint/valid/_annotations.coco.json',
            'imgs': 'data/labels/coco_keypoint/valid',
            'out':  'crops/val',
        },
        {
            'json': 'data/labels/coco_keypoint/test/_annotations.coco.json',
            'imgs': 'data/labels/coco_keypoint/test',
            'out':  'crops/test',
        },
    ]
    for s in SPLITS:
        print(f"\n── {s['out']} ──")
        crop_from_coco(s['json'], s['imgs'], s['out'])
