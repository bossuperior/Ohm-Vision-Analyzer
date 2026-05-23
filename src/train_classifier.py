"""
Train resistor classification model for fair comparison (Chapter 4).
Supported backbones: yolo_cls | shufflenet | mobilenet

Usage:
    python train_classifier.py --backbone yolo_cls  --data path/to/dataset
    python train_classifier.py --backbone shufflenet --data path/to/dataset
    python train_classifier.py --backbone mobilenet  --data path/to/dataset

Dataset layout (Roboflow ImageFolder export):
    dataset/
    ├── train/<class>/...
    ├── valid/<class>/...   (หรือ val/)
    └── test/<class>/...

Colab quick-start:
    !pip install albumentations scikit-learn ultralytics
    %run src/train_classifier.py --backbone yolo_cls  --data /content/drive/MyDrive/dataset
    %run src/train_classifier.py --backbone shufflenet --data /content/drive/MyDrive/dataset
    %run src/train_classifier.py --backbone mobilenet  --data /content/drive/MyDrive/dataset
"""

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Transform — normalize + tensor เท่านั้น ───────────────────────────────────
# dataset ถูก augment และ resize 224×224 ไปแล้วใน cls_dataset_prepare.py
# ไม่ augment ซ้ำที่นี่ เพื่อให้ทุกโมเดลเห็น input เดียวกัน (fair comparison)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_transform = A.Compose([
    A.Resize(224, 224),   # safety — ภาพควรเป็น 224×224 อยู่แล้ว
    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])


# ── Dataset wrapper (PyTorch models) ─────────────────────────────────────────
class AlbumDataset(Dataset):
    def __init__(self, root: str, transform: A.Compose):
        self._ds        = datasets.ImageFolder(root)
        self._transform = transform
        self.classes    = self._ds.classes

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        path, label = self._ds.samples[idx]
        img    = cv2.imread(path)
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self._transform(image=img)['image']
        return tensor, label


def _resolve_val_dir(data_root: Path) -> Path:
    """Roboflow ส่งออกมาเป็น valid/ แต่บางครั้งเป็น val/"""
    for name in ('valid', 'val'):
        p = data_root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"ไม่พบ val/valid ใน {data_root}")


# ── PyTorch model factory ─────────────────────────────────────────────────────
def _build_torch_model(backbone: str, num_classes: int) -> nn.Module:
    import torchvision.models as M

    if backbone == 'shufflenet':
        net    = M.shufflenet_v2_x1_0(weights=M.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        net.fc = nn.Linear(1024, num_classes)
        return net

    if backbone == 'mobilenet':
        net               = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.DEFAULT)
        net.classifier[3] = nn.Linear(1024, num_classes)
        return net

    raise ValueError(f"Unknown backbone: {backbone!r}")


# ── Training / eval loops ─────────────────────────────────────────────────────
def _train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    correct = total = 0
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        optimizer.zero_grad()
        out    = model(x)
        loss   = criterion(out, y)
        loss.backward()
        optimizer.step()
        correct += (out.detach().argmax(1) == y).sum().item()
        total   += len(y)
    return correct / total


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[float, list, list]:
    model.eval()
    correct = total = 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        preds  = model(x).argmax(1)
        correct    += (preds == y).sum().item()
        total      += len(y)
        all_preds  += preds.cpu().tolist()
        all_labels += y.cpu().tolist()
    return correct / total, all_preds, all_labels


def _cpu_inference_ms(model_cpu: nn.Module) -> float:
    """วัด inference speed บน CPU (fair comparison ทุกโมเดล)"""
    model_cpu.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    for _ in range(10):           # warm-up
        model_cpu(dummy)
    t0 = time.time()
    for _ in range(100):
        model_cpu(dummy)
    return (time.time() - t0) / 100 * 1000


def _write_summary(out_dir: Path, backbone: str,
                   test_acc: float, best_val_acc: float,
                   params_m: float, size_mb: float, ms_cpu: float,
                   stopped_epoch: int, max_epochs: int) -> None:
    summary_path = out_dir.parent / 'comparison.csv'
    fields = ['backbone', 'test_acc', 'best_val_acc',
              'stopped_epoch', 'max_epochs',
              'params_M', 'size_mb', 'ms_per_image_cpu']
    write_header = not summary_path.exists()
    with open(summary_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({
            'backbone':         backbone,
            'test_acc':         round(test_acc, 4),
            'best_val_acc':     round(best_val_acc, 4),
            'stopped_epoch':    stopped_epoch,
            'max_epochs':       max_epochs,
            'params_M':         round(params_m, 2),
            'size_mb':          round(size_mb, 2),
            'ms_per_image_cpu': round(ms_cpu, 2),
        })
    print(f"[saved] {summary_path}")


# ── YOLO-cls branch ───────────────────────────────────────────────────────────
def train_yolo_cls(args) -> None:
    from ultralytics import YOLO

    data_root = Path(args.data)
    val_dir   = _resolve_val_dir(data_root)
    out_dir   = Path(args.out) / 'yolo_cls'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = '0' if torch.cuda.is_available() else 'cpu'

    # สร้าง symlink val/ → valid/ ถ้า YOLO หา val ไม่เจอ
    val_link = data_root / 'val'
    if not val_link.exists() and val_dir.name == 'valid':
        val_link.symlink_to(val_dir.resolve())

    model = YOLO('yolov8n-cls.pt')   # YOLOv8 nano-cls pretrained

    model.train(
        data    = str(data_root),
        epochs  = args.epochs,
        imgsz   = 224,
        batch   = args.batch,
        seed    = SEED,
        device  = device,
        patience= args.patience,
        project = str(out_dir),
        name    = 'run',
        exist_ok= True,
        verbose = False,
    )

    # actual save dir from trainer (more reliable than reconstructing the path)
    save_dir = Path(model.trainer.save_dir)
    best_pt  = save_dir / 'weights' / 'best.pt'

    # ── test accuracy ────────────────────────────────────────────────────────
    test_results  = model.val(data=str(data_root), split='test',
                               imgsz=224, device=device, verbose=False)
    test_acc      = float(test_results.top1)

    val_results   = model.val(data=str(data_root), split='val',
                               imgsz=224, device=device, verbose=False)
    best_val_acc  = float(val_results.top1)

    # ── inference speed (CPU) ────────────────────────────────────────────────
    torch_model = model.model.cpu().eval()
    ms_cpu = _cpu_inference_ms(torch_model)

    # ── params / size ────────────────────────────────────────────────────────
    params_m = sum(p.numel() for p in torch_model.parameters()) / 1e6
    size_mb  = best_pt.stat().st_size / 1e6 if best_pt.exists() else 0.0

    # ── classification report (test set) ────────────────────────────────────
    # reload from disk: model.model was moved to CPU for speed test, making
    # subsequent model.predict() calls fail with tensor version counter errors
    model_report = YOLO(str(best_pt))
    test_dir    = data_root / 'test'
    class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    all_preds, all_labels = [], []
    for label_idx, cls in enumerate(class_names):
        for img_path in (test_dir / cls).glob('*'):
            r = model_report.predict(str(img_path), imgsz=224, verbose=False, device=device)
            all_preds.append(int(r[0].probs.top1))
            all_labels.append(label_idx)

    print(f"\n{'='*50}")
    print(f"[yolo_cls] Test Accuracy : {test_acc:.4f}")
    print(f"[yolo_cls] Inference (CPU): {ms_cpu:.2f} ms/image")
    print(f"\n{classification_report(all_labels, all_preds, target_names=class_names)}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # YOLO บันทึก epoch จริงที่หยุดไว้ใน results.csv
    results_csv = save_dir / 'results.csv'
    if results_csv.exists():
        import pandas as pd
        stopped_epoch = len(pd.read_csv(results_csv))
    else:
        stopped_epoch = args.epochs

    _write_summary(out_dir, 'yolo_cls', test_acc, best_val_acc,
                   params_m, size_mb, ms_cpu, stopped_epoch, args.epochs)


# ── PyTorch branch (shufflenet / mobilenet) ───────────────────────────────────
def train_torch(args) -> None:
    _seed_everything(SEED)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = Path(args.data)
    val_dir   = _resolve_val_dir(data_root)

    print(f"[device] {device}  |  backbone: {args.backbone}")

    train_ds = AlbumDataset(str(data_root / 'train'), _transform)
    val_ds   = AlbumDataset(str(val_dir),             _transform)
    test_ds  = AlbumDataset(str(data_root / 'test'),  _transform)

    num_classes = len(train_ds.classes)
    print(f"[data] {num_classes} classes — train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"[classes] {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model     = _build_torch_model(args.backbone, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    out_dir = Path(args.out) / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    no_improve   = 0
    stopped_epoch = args.epochs
    history      = []

    for epoch in range(1, args.epochs + 1):
        t0        = time.time()
        train_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _ = _evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        history.append({'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc})
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_acc:.4f}  val={val_acc:.4f}  "
              f"no_improve={no_improve}/{args.patience}  ({time.time()-t0:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            no_improve    = 0
            torch.save(model.state_dict(), out_dir / 'best.pt')
        else:
            no_improve += 1

        if no_improve >= args.patience:
            stopped_epoch = epoch
            print(f"[early stop] epoch {epoch} — val_acc ไม่ดีขึ้นใน {args.patience} epochs")
            break

    # ── test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device,
                                     weights_only=True))
    ms_cpu   = _cpu_inference_ms(model.cpu())
    model    = model.to(device)
    test_acc, preds, labels = _evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    print(f"[{args.backbone}] Test Accuracy : {test_acc:.4f}")
    print(f"[{args.backbone}] Inference (CPU): {ms_cpu:.2f} ms/image")
    print(f"\n{classification_report(labels, preds, target_names=train_ds.classes)}")
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    with open(out_dir / 'history.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_acc', 'val_acc'])
        w.writeheader()
        w.writerows(history)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    size_mb  = (out_dir / 'best.pt').stat().st_size / 1e6
    _write_summary(out_dir, args.backbone, test_acc, best_val_acc,
                   params_m, size_mb, ms_cpu, stopped_epoch, args.epochs)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', required=True,
                    choices=['yolo_cls', 'shufflenet', 'mobilenet'])
    ap.add_argument('--data',   required=True,  help='path to dataset root')
    ap.add_argument('--epochs',   type=int,   default=100)
    ap.add_argument('--patience', type=int,   default=15,
                    help='early stop หาก val_acc ไม่ดีขึ้นใน N epochs')
    ap.add_argument('--batch',    type=int,   default=32)
    ap.add_argument('--lr',       type=float, default=1e-3)
    ap.add_argument('--out',      default='results', help='output folder')
    args = ap.parse_args()

    if args.backbone == 'yolo_cls':
        train_yolo_cls(args)
    else:
        train_torch(args)


if __name__ == '__main__':
    main()
