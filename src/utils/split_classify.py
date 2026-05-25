"""
Split data/processed/crops/<class>/ → data/classify/{train,val,test}/<class>/

Grouping rules:
  stem_annID.jpg       → original  (unit of split)
  stem_annID_flip.jpg  → follows its original's split
  stem_annID_aug*.jpg  → train only, regardless of which group

Split ratio: 70% train / 15% val / 15% test  (stratified per class)
"""

import re
import random
import shutil
from collections import defaultdict
from pathlib import Path

CROPS_DIR  = Path('data/processed/crops')
OUTPUT_DIR = Path('data/classify')
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42


def _base_stem(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r'_flip$', '', stem)
    stem = re.sub(r'_aug\d+$', '', stem)
    return stem


def _is_aug(path: Path) -> bool:
    return bool(re.search(r'_aug\d+', path.stem))


def _split_class(cls_dir: Path) -> dict:
    cls_name = cls_dir.name
    files = sorted(cls_dir.glob('*.jpg')) + sorted(cls_dir.glob('*.png'))
    if not files:
        print(f"[skip] {cls_name}: ไม่มีไฟล์")
        return {}

    # จัดกลุ่มไฟล์ตาม base_stem
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        groups[_base_stem(f)].append(f)

    # shuffle source groups ด้วย seed คงที่
    group_keys = sorted(groups.keys())
    rng = random.Random(SEED)
    rng.shuffle(group_keys)

    n       = len(group_keys)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_keys = set(group_keys[:n_train])
    val_keys   = set(group_keys[n_train:n_train + n_val])

    counts = {'train': 0, 'val': 0, 'test': 0}

    for key, group_files in groups.items():
        for f in group_files:
            if _is_aug(f):
                split = 'train'          # aug เข้า train เสมอ
            elif key in train_keys:
                split = 'train'
            elif key in val_keys:
                split = 'val'
            else:
                split = 'test'

            dest = OUTPUT_DIR / split / cls_name / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            counts[split] += 1

    return counts


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"ลบ {OUTPUT_DIR} เดิมแล้ว")

    total = {'train': 0, 'val': 0, 'test': 0}

    for cls_dir in sorted(CROPS_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        counts = _split_class(cls_dir)
        if counts:
            print(f"  [{cls_dir.name}] train={counts['train']}  val={counts['val']}  test={counts['test']}")
            for k in total:
                total[k] += counts[k]

    print(f"\nรวมทั้งหมด → train={total['train']}  val={total['val']}  test={total['test']}")
    print(f"Output → {OUTPUT_DIR}/{{train,val,test}}/<class>/")


if __name__ == '__main__':
    main()
