import json
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path


# Class names ใน COCO ที่ต้องการ crop body
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
    perp_pad = int(np.clip(length * 0.35, 25, 40))

    out_w = int(length) + 24
    out_h = perp_pad * 2

    tx = cx - (out_w / 2) * cos_a + (out_h / 2) * sin_a
    ty = cy - (out_w / 2) * sin_a - (out_h / 2) * cos_a
    M  = np.float32([[cos_a, -sin_a, tx],
                     [sin_a,  cos_a, ty]])

    crop = cv2.warpAffine(img, M, (out_w, out_h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return crop if crop.size > 0 else None


_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)) # อัปเกรด ClipLimit เพิ่ม Contrast แถบสีให้ชัดขึ้น


def _to_landscape(img: np.ndarray) -> np.ndarray:
    """หมุน 90° CW ถ้าภาพออกมาแนวตั้ง — ใช้ทั้ง training และ inference ให้สอดคล้องกัน"""
    if img is not None and img.shape[0] > img.shape[1]:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _mask_body(crop: np.ndarray) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return None

    # 1. แปลงภาพและเบลอเพื่อลดสัญญาณรบกวน
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    # 2. คัดแยกพิกเซลที่ไม่ใช่ขอบดำ (ขอบดำเกิดจากตอนหมุนภาพ Affine)
    valid = blur > 8
    valid_vals = blur[valid]
    if len(valid_vals) < 50:
        return None

    # 3. Binary mask: เทียบกับสี background จาก margin บน/ล่างของ affine crop
    # (ดีกว่า Otsu เพราะจับ body ceramic สีอ่อนได้ด้วย ไม่ใช่แค่แถบสีเข้ม)
    marg = max(4, h // 7)
    bg_strip = np.concatenate([blur[:marg, :].ravel(), blur[-marg:, :].ravel()])
    bg_valid  = bg_strip[bg_strip > 8]
    if len(bg_valid) > 20:
        bg_val = float(np.median(bg_valid))
        diff   = np.abs(blur.astype(np.float32) - bg_val)
        binary = np.where((diff > 18) & valid, np.uint8(255), np.uint8(0))
    else:                                              # fallback: Otsu ถ้า margin ว่าง
        thr, _ = cv2.threshold(valid_vals.reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, binary = cv2.threshold(blur, int(thr), 255, cv2.THRESH_BINARY_INV)
        binary[~valid] = 0

    # 4. Opening 5×5 — ลบ noise เล็ก ≤4px (รูบอร์ด, ลวดบางมาก)
    k5 = np.ones((5, 5), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k5)
    if cv2.countNonZero(binary_clean) < 50:
        binary_clean = binary

    # 5. Contour → Y range เท่านั้น (reliable เพราะ body อยู่กลาง crop แนวตั้ง)
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_cx, img_cy = w / 2.0, h / 2.0
    # กรอง: กว้าง ≥ 8px และ สูง ≥ 10px (รูบอร์ด ~5px จะตกออก)
    min_ch = max(10, h // 6)
    valid_cs = [(c, cv2.boundingRect(c)) for c in contours
                if cv2.boundingRect(c)[2] >= 8 and cv2.boundingRect(c)[3] >= min_ch]
    if not valid_cs:
        # fallback: ลดเกณฑ์ถ้าไม่เจออะไรเลย
        valid_cs = [(c, cv2.boundingRect(c)) for c in contours
                    if cv2.boundingRect(c)[2] >= 8 and cv2.boundingRect(c)[3] >= 5]
    if not valid_cs:
        return None

    best_c, best_rect = min(valid_cs, key=lambda t: (
        np.hypot(t[1][0] + t[1][2] / 2 - img_cx,
                 t[1][1] + t[1][3] / 2 - img_cy)
        - cv2.contourArea(t[0]) * 0.05
    ))

    # รวม Y range จากทุก contour ที่ overlap แนวตั้งกับ best
    _, by, _, bh_c = best_rect
    ry0 = by; ry1 = by + bh_c
    for _, (cx2, cy2, cw2, ch2) in valid_cs:
        if not (cy2 + ch2 < ry0 - 4 or cy2 > ry1 + 4):
            ry0 = min(ry0, cy2);  ry1 = max(ry1, cy2 + ch2)
    y0 = max(0, ry0 - 2);  y1 = min(h, ry1 + 2)
    body_h = max(1, ry1 - ry0)

    # 6. Column scan ภายใน Y strip — แยก body ออกจาก leads
    # leads บาง ~5px ใน strip สูง 25px → coverage 20%; body เต็ม → 80-100%
    # ใช้ threshold 50% → leads ตกออกเสมอ, body ผ่านเสมอ
    strip     = binary_clean[ry0:ry1, :]
    col_cover = strip.sum(axis=0) / 255.0
    x0, x1 = 0, w
    for thr_frac in [0.50, 0.35, 0.20, 0.10]:
        is_body = col_cover >= body_h * thr_frac
        if not is_body.any():
            continue
        xs   = np.where(is_body)[0]
        # longest contiguous run (gap ≤ 15px รองรับช่องว่างระหว่าง band)
        gaps = np.where(np.diff(xs) > 15)[0]
        if len(gaps):
            starts = np.concatenate([[xs[0]], xs[gaps + 1]])
            ends   = np.concatenate([xs[gaps], [xs[-1]]])
            best   = int(np.argmax(ends - starts))
            rx0, rx1 = int(starts[best]), int(ends[best])
        else:
            rx0, rx1 = int(xs[0]), int(xs[-1])
        x0 = max(0, rx0 - 2);  x1 = min(w, rx1 + 2)
        break

    tight = crop[y0:y1, x0:x1]

    # 7. Portrait safety
    th, tw = tight.shape[:2]
    if tw > 0 and th > tw:
        tight = crop[y0:y1, :]

    return _to_landscape(tight)


def _inv_letterbox_kpts(kpts_flat: list, proc_w: int, proc_h: int,
                        orig_w: int, orig_h: int) -> list:
    scale = min(proc_w / orig_w, proc_h / orig_h)
    pad_x = (proc_w - orig_w * scale) / 2
    pad_y = (proc_h - orig_h * scale) / 2
    out = []
    for i in range(0, len(kpts_flat), 3):
        x = (kpts_flat[i]     - pad_x) / scale
        y = (kpts_flat[i + 1] - pad_y) / scale
        v =  kpts_flat[i + 2]
        out.extend([x, y, v])
    return out


_MIN_CROP_W   = 50   # pixels — crop เล็กกว่านี้ไม่มีรายละเอียดแถบสีเพียงพอ
_MAX_CROP_RATIO = 6.0  # w/h — กว้างกว่านี้ = keypoint อยู่ที่ปลายลวด ไม่ใช่ขอบ body


def _postprocess_crop(crop: np.ndarray) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None

    # หาพื้นที่ที่ไม่ใช่สีดำ (ส่วนที่เป็นเนื้องานของตัวต้านทาน)
    gray_orig = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    valid_mask = gray_orig > 5

    if valid_mask.sum() < 50:
        return None

    # 1. CLAHE (สกัดสีให้เด้งขึ้นโดยดึง Contrast ใน Channel L ของแกน LAB)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. ป้องกันภาพเบลอหลุดรอด (คำนวณ Blur Variance *เฉพาะจุดที่ไม่ใช่พื้นดำ*)
    gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray_enhanced, cv2.CV_64F)[valid_mask].var())
    
    # ลดเกณฑ์ลงเหลือ 15 เพื่อรับภาพที่พิกเซลแตกนิดหน่อยได้ (Data Realism)
    if lap_var < 15.0: 
        return None

    # 3. Unsharp Masking (สูตรเร่งความคมชัดขอบแถบสี)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # 4. ฟื้นฟูการถมดำ! (ป้องกันไม่ให้กระบวนการ Sharpen ไปสร้าง Noise ขาวๆ ตรงขอบภาพ)
    mask3 = np.stack([valid_mask] * 3, axis=-1)
    final_result = np.where(mask3, sharpened, 0)

    return final_result.astype(np.uint8)


def crop_from_coco(json_path: str, img_dir: str, out_dir: str,
                   raw_dir: str | None = None) -> None:
    json_path = Path(json_path)
    img_dir   = Path(img_dir)
    out_dir   = Path(out_dir)
    raw_dir   = Path(raw_dir) if raw_dir else None

    # clean previous output to avoid stale crops accumulating across runs
    for cls in _BODY_CLASSES:
        d = out_dir / cls
        if d.exists():
            shutil.rmtree(d)

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    cat_id_to_name = {c['id']: c['name'].strip().lower()
                      for c in data['categories']}
    img_id_to_info = {img['id']: img for img in data['images']}

    saved = skipped = raw_used = rejected_blur = rejected_small = rejected_body = 0

    for ann in data['annotations']:
        cat_name = cat_id_to_name.get(ann['category_id'], '')
        if cat_name not in _BODY_CLASSES:
            continue

        img_info = img_id_to_info[ann['image_id']]
        kpts = ann.get('keypoints', [])
        if len(kpts) < 6:
            skipped += 1
            continue

        # ── ลองใช้ภาพต้นฉบับจาก raw_dir ก่อน ──────────────────────────────
        src_img  = None
        src_kpts = kpts
        if raw_dir is not None:
            orig_name = (img_info.get('extra') or {}).get('name', '')
            if orig_name:
                raw_file = raw_dir / orig_name
                if raw_file.exists():
                    candidate = cv2.imread(str(raw_file))
                    if candidate is not None:
                        oh, ow = candidate.shape[:2]
                        ph, pw = img_info['height'], img_info['width']
                        src_kpts = _inv_letterbox_kpts(kpts, pw, ph, ow, oh)
                        src_img  = candidate
                        raw_used += 1

        # ── fallback: ใช้ภาพ processed ──────────────────────────────────────
        if src_img is None:
            proc_file = img_dir / img_info['file_name']
            if not proc_file.exists():
                print(f"[skip] image not found: {proc_file}")
                skipped += 1
                continue
            src_img = cv2.imread(str(proc_file))
            if src_img is None:
                skipped += 1
                continue

        # ── keypoints → kp_pairs ────────────────────────────────────────────
        kp_pairs = []
        for i in range(0, min(len(src_kpts), 6), 3):
            x, y, v = src_kpts[i], src_kpts[i + 1], src_kpts[i + 2]
            if v > 0:
                kp_pairs.append(np.array([x, y]))

        if len(kp_pairs) < 2:
            skipped += 1
            continue

        crop = _affine_crop(src_img, kp_pairs[0], kp_pairs[-1])
        if crop is None:
            skipped += 1
            continue
        crop = _mask_body(crop)
        if crop is None:
            rejected_body += 1
            continue
        if crop.shape[1] < _MIN_CROP_W:
            rejected_small += 1
            continue
        crop = _postprocess_crop(crop)
        if crop is None:
            rejected_blur += 1
            continue

        class_dir = out_dir / cat_name
        class_dir.mkdir(parents=True, exist_ok=True)
        stem     = Path(img_info['file_name']).stem
        out_name = f"{stem}_ann{ann['id']}.jpg"
        cv2.imwrite(str(class_dir / out_name), crop)
        saved += 1

    total_attempted = saved + skipped + rejected_blur + rejected_small + rejected_body
    src_note   = f", from_raw={raw_used}/{total_attempted}" if raw_dir else ""
    blur_note  = f", rejected_blur={rejected_blur}"   if rejected_blur  else ""
    small_note = f", rejected_small={rejected_small}" if rejected_small else ""
    body_note  = f", rejected_body={rejected_body}"   if rejected_body  else ""
    print(f"Saved  : {saved} crops{src_note}{blur_note}{small_note}{body_note} → {out_dir}/{{resistor_4b,resistor_5b}}/")
    print(f"Skipped: {skipped} annotations")
    print(f"\nNext step: ใน {out_dir}/<class>/ แยก crop ไปใน data/classify/train/<ohm_value>/")


def crop_body_for_classifier(img: np.ndarray,
                             p0: np.ndarray,
                             p1: np.ndarray) -> np.ndarray | None:
    """
    Full crop pipeline identical to training data generation.
    Input: BGR numpy array + two endpoint keypoints (x, y).
    Output: BGR crop ready for ClassificationEngine.predict(), or None.
    """
    crop = _affine_crop(img, p0, p1)
    if crop is None:
        return None
    crop = _mask_body(crop)
    if crop is None:
        return None
    h_c, w_c = crop.shape[:2]
    if w_c < _MIN_CROP_W:
        return None
    if h_c > 0 and w_c / h_c > _MAX_CROP_RATIO:  # keypoint อยู่ปลายลวด ไม่ใช่ body
        return None
    crop = _postprocess_crop(crop)
    if crop is None:
        return None
    return _to_landscape(crop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='path to _annotations.coco.json')
    ap.add_argument('--imgs', required=True, help='folder containing images')
    ap.add_argument('--out',  default='crops/unsorted', help='output folder')
    args = ap.parse_args()
    crop_from_coco(args.json, args.imgs, args.out)


if __name__ == '__main__':
    RAW_DIR = 'data/raw/dataset_capture'   # ภาพต้นฉบับ — None ถ้าไม่มี
    SPLITS = [
        {
            'json': 'data/labels/raw_data_crop/train/_annotations.coco.json',
            'imgs': 'data/labels/raw_data_crop/train',
            'out':  'data/processed/crops/train',
        },
        {
            'json': 'data/labels/raw_data_crop/valid/_annotations.coco.json',
            'imgs': 'data/labels/raw_data_crop/valid',
            'out':  'data/processed/crops/val',
        },
        {
            'json': 'data/labels/raw_data_crop/test/_annotations.coco.json',
            'imgs': 'data/labels/raw_data_crop/test',
            'out':  'data/processed/crops/test',
        },
    ]
    for s in SPLITS:
        print(f"\n── {s['out']} ──")
        crop_from_coco(s['json'], s['imgs'], s['out'], raw_dir=RAW_DIR)
