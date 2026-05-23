"""
Split + augment resistor crop dataset สำหรับ fair model comparison (Chapter 4)

Pipeline:
  data/processed/crops/<class>/*.jpg   (flat — ทุก class รวมกัน)
      ↓  stratified split 70 / 20 / 10
      ↓  train → augment จนครบ TARGET_TRAIN ต่อ class
      ↓  val / test → resize เท่านั้น (ไม่ augment)
  data/cls_dataset/
      ├── train/<class>/
      ├── val/<class>/
      └── test/<class>/

Run:
    python src/utils/cls_dataset_prepare.py
"""

import random
import shutil
import cv2
import albumentations as A
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.20
# TEST_RATIO  = 0.10 (เหลือจาก train + val)

TARGET_TRAIN = 150   # จำนวน train images ต่อ class หลัง augment

SOURCE_DIR = Path('data/processed/crops')
OUT_DIR    = Path('data/cls_dataset')

IMG_EXTS = {'.jpg', '.jpeg', '.png'}

# ── Transforms ────────────────────────────────────────────────────────────────
# val/test: resize เท่านั้น — ห้าม augment เพื่อ fair evaluation
_eval_tf = A.Compose([A.Resize(224, 224)])

# train: augmentation ที่ปลอดภัยสำหรับแถบสีต้านทาน
# ❌ ห้าม HorizontalFlip / VerticalFlip — แถบสีพลิกแล้วอ่านผิด
_train_tf = A.Compose([
    A.Resize(224, 224),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.20, p=0.8),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=20, p=0.6),
    A.RandomGamma(gamma_limit=(85, 115), p=0.4),
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),
    A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
])


# ── Helpers ───────────────────────────────────────────────────────────────────
def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMG_EXTS)


def _split(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Stratified split ต่อ class — reproducible ด้วย SEED"""
    files = files.copy()
    random.shuffle(files)
    n      = len(files)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val   = max(1, round(n * VAL_RATIO))
    train = files[:n_train]
    val   = files[n_train: n_train + n_val]
    test  = files[n_train + n_val:]
    return train, val, test


def _copy_resize(src: Path, dst: Path) -> None:
    img = cv2.imread(str(src))
    if img is None:
        return
    out = _eval_tf(image=img)['image']
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _augment_to_target(src_files: list[Path], out_dir: Path,
                        target: int) -> None:
    """
    คัดลอก source ก่อน แล้ว augment เพิ่มจนครบ target
    ไฟล์ original ใช้ชื่อเดิม, ไฟล์ augment ใช้ suffix _augNNN
    """
    # 1. copy + resize ไฟล์ต้นฉบับทั้งหมดก่อน
    for src in src_files:
        _copy_resize(src, out_dir / src.name)

    needed  = target - len(src_files)
    if needed <= 0:
        return

    # 2. วน augment จนครบ target
    pool    = src_files * (needed // len(src_files) + 2)
    random.shuffle(pool)
    counter = 0

    for src in pool:
        if counter >= needed:
            break
        img = cv2.imread(str(src))
        if img is None:
            continue
        aug = _train_tf(image=img)['image']
        out_name = f"{src.stem}_aug{counter:03d}.jpg"
        cv2.imwrite(str(out_dir / out_name), aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
        counter += 1


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    random.seed(SEED)

    # ล้าง output เก่า
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    class_dirs = sorted(d for d in SOURCE_DIR.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"[error] ไม่พบ class folders ใน {SOURCE_DIR}")
        return

    print(f"พบ {len(class_dirs)} classes  |  TARGET_TRAIN = {TARGET_TRAIN}/class")
    print(f"Split ratio: train={TRAIN_RATIO:.0%}  val={VAL_RATIO:.0%}  "
          f"test={1-TRAIN_RATIO-VAL_RATIO:.0%}\n")

    totals = {'train': 0, 'val': 0, 'test': 0}

    for cls_dir in class_dirs:
        files = _list_images(cls_dir)
        if not files:
            print(f"  [skip] {cls_dir.name} — ไม่มีภาพ")
            continue

        train_files, val_files, test_files = _split(files)

        # สร้าง output dirs
        for split in ('train', 'val', 'test'):
            (OUT_DIR / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

        # train → augment จนครบ TARGET_TRAIN
        _augment_to_target(train_files,
                           OUT_DIR / 'train' / cls_dir.name,
                           TARGET_TRAIN)

        # val / test → resize เท่านั้น
        for src in val_files:
            _copy_resize(src, OUT_DIR / 'val' / cls_dir.name / src.name)
        for src in test_files:
            _copy_resize(src, OUT_DIR / 'test' / cls_dir.name / src.name)

        n_train = len(list((OUT_DIR / 'train' / cls_dir.name).glob('*.jpg')))
        totals['train'] += n_train
        totals['val']   += len(val_files)
        totals['test']  += len(test_files)

        print(f"  {cls_dir.name:15s}  "
              f"src={len(files):3d}  "
              f"train={n_train:3d}  "
              f"val={len(val_files):3d}  "
              f"test={len(test_files):3d}")

    print(f"\nรวม → train={totals['train']}  val={totals['val']}  test={totals['test']}")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == '__main__':
    main()
