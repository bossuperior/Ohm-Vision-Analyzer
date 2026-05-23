"""
Augment minority classes in crops directory to reach a target count.
Safe for resistor color-band crops: no horizontal flip, no cutout.
"""
import random
import cv2
import numpy as np
import albumentations as A
from pathlib import Path


TARGET = 100

# Classes to augment: folder name → current count (auto-detected at runtime)
MINORITY_CLASSES = ['4k7_5pct', '820R_1pct']

CROPS_DIR = Path('data/processed/crops')


def _build_pipeline() -> A.Compose:
    return A.Compose([
        # ── geometry (เบา — ภาพเล็ก + แถบสีต้องอ่านได้) ──────────────────
        A.Rotate(
            limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.08,
            rotate_limit=0,          # rotate แยกอยู่แล้วด้านบน
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),

        # ── แสงและสี ──────────────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.20,
            p=0.8,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8,       # เล็กน้อย — ป้องกันสีแถบเพี้ยน
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.6,
        ),
        A.RandomGamma(gamma_limit=(85, 115), p=0.4),

        # ── noise / blur (เบามาก — ภาพเล็ก) ──────────────────────────────
        A.GaussianBlur(blur_limit=(3, 3), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
    ])


def augment_class(cls_dir: Path, target: int) -> None:
    src_files = sorted(cls_dir.glob('*.jpg')) + sorted(cls_dir.glob('*.png'))
    current = len(src_files)
    needed = target - current

    if needed <= 0:
        print(f"[skip] {cls_dir.name}: {current} >= {target}, ไม่ต้อง augment")
        return

    print(f"[{cls_dir.name}] {current} → {target}  (สร้างเพิ่ม {needed} ใบ)")

    pipeline = _build_pipeline()
    generated = 0

    # วนซ้ำผ่าน source images จนครบ
    src_cycle = src_files * (needed // len(src_files) + 2)
    random.shuffle(src_cycle)

    for src_path in src_cycle:
        if generated >= needed:
            break

        img = cv2.imread(str(src_path))
        if img is None:
            continue

        aug = pipeline(image=img)['image']

        out_name = f"{src_path.stem}_aug{generated:03d}.jpg"
        cv2.imwrite(str(cls_dir / out_name), aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
        generated += 1

    total = len(list(cls_dir.glob('*.jpg'))) + len(list(cls_dir.glob('*.png')))
    print(f"  ✓ เสร็จ — รวม {total} ใบใน {cls_dir}")


def main() -> None:
    for cls_name in MINORITY_CLASSES:
        cls_dir = CROPS_DIR / cls_name
        if not cls_dir.exists():
            print(f"[warn] ไม่พบ folder: {cls_dir}")
            continue
        augment_class(cls_dir, TARGET)


if __name__ == '__main__':
    main()
