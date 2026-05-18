"""
augment_classify.py
───────────────────
Augment classifier dataset หลังจาก sort ด้วยมือแล้ว
รัน script นี้บน data/classify/train/ เพื่อเพิ่มจำนวนภาพต่อ class

Usage:
    python -m src.utils.augment_classify \
        --src data/classify/train \
        --n   8
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def _augments(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """คืน list ของ (suffix, augmented_image) — ไม่แตะ hue มากเพื่อกันสีแถบเพี้ยน"""
    out = []

    # 1. Flip horizontal (resistor อ่านได้สองทิศ)
    out.append(('flr', cv2.flip(img, 1)))

    # 2-3. Brightness ±25%
    for factor, tag in [(1.25, 'brt'), (0.75, 'drk')]:
        out.append((tag, np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)))

    # 4-5. Slight rotation ±12° (แถบอาจไม่ตั้งฉากพอดีใน crop)
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    for angle, tag in [(12, 'r12'), (-12, 'l12')]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        out.append((tag, rot))

    # 6. Gaussian noise (simulate sensor noise)
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    out.append(('nse', np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)))

    # 7. Mild contrast stretch (simulate different exposure)
    alpha = np.random.uniform(0.85, 1.15)
    beta  = np.random.uniform(-10, 10)
    out.append(('ctr', np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)))

    # 8. Flip + brightness (combination)
    out.append(('flb', np.clip(cv2.flip(img, 1).astype(np.float32) * 1.15, 0, 255).astype(np.uint8)))

    return out


def augment_dataset(src_dir: str, n: int = 8) -> None:
    src = Path(src_dir)
    if not src.exists():
        print(f"[error] ไม่พบ: {src}")
        return

    classes = [d for d in src.iterdir() if d.is_dir()]
    print(f"พบ {len(classes)} classes ใน {src}")

    total_new = 0
    for cls_dir in sorted(classes):
        originals = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
        # ข้ามไฟล์ที่ augment แล้ว (ชื่อมี _aug_)
        originals = [f for f in originals if '_aug_' not in f.stem]

        added = 0
        for img_path in originals:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            augs = _augments(img)[:n]
            for suffix, aug_img in augs:
                out_name = img_path.stem + f'_aug_{suffix}.jpg'
                cv2.imwrite(str(cls_dir / out_name), aug_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                added += 1

        print(f"  {cls_dir.name:20s}: {len(originals)} originals → +{added} augmented")
        total_new += added

    print(f"\nรวม augmented: +{total_new} ภาพ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/classify/train',
                    help='folder ที่มี class subfolders')
    ap.add_argument('--n',   type=int, default=8,
                    help='จำนวน augmentation ต่อภาพ (max 8)')
    args = ap.parse_args()
    augment_dataset(args.src, min(args.n, 8))


if __name__ == '__main__':
    main()
