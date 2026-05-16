import json
import os

class CocoToYoloPoseConverter:
    def __init__(self, max_keypoints=4):
        self.max_keypoints = max_keypoints
        # Mapping COCO class names → YOLO class IDs
        self.target_classes = {
            'resistor':    0,
            'wire':        1,
        }

    def convert(self, json_path, output_dir):
        print(f"\n Processing: {json_path}")
        if not os.path.exists(json_path):
            print(f"  [skip] File not found: {json_path}")
            return
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Category ID Pairing
        coco_to_yolo_id = {}
        found_cats = []
        for cat in data.get('categories', []):
            cat_name = cat['name'].strip().lower()
            if cat_name in self.target_classes:
                coco_to_yolo_id[cat['id']] = self.target_classes[cat_name]
                found_cats.append(f"'{cat['name']}' → class {self.target_classes[cat_name]}")

        print(f" Categories mapped: {', '.join(found_cats) if found_cats else 'NONE (check category names!)'}")

        # 2. Store image width and height information
        image_info = {
            img['id']: (img['width'], img['height'], img['file_name'])
            for img in data.get('images', [])
        }

        os.makedirs(output_dir, exist_ok=True)

        # 3. Group Annotations by image
        annotations_by_image = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            annotations_by_image.setdefault(img_id, []).append(ann)
        files_created       = 0
        skipped_annotations = 0

        # 4. Loop through each image and convert annotations
        for img_id, (img_w, img_h, file_name) in image_info.items():
            if img_id not in annotations_by_image:
                continue

            txt_filename = os.path.splitext(os.path.basename(file_name))[0] + '.txt'
            txt_filepath = os.path.join(output_dir, txt_filename)

            yolo_lines = []

            for ann in annotations_by_image[img_id]:
                coco_cat_id = ann['category_id']

                if coco_cat_id not in coco_to_yolo_id:
                    skipped_annotations += 1
                    continue

                yolo_class_id = coco_to_yolo_id[coco_cat_id]

                # BBox (COCO: x_min, y_min, w, h → YOLO: cx, cy, w, h normalized)
                bx, by, bw, bh = [float(v) for v in ann['bbox']]
                x_center = (bx + bw / 2.0) / img_w
                y_center = (by + bh / 2.0) / img_h
                norm_bw  = bw / img_w
                norm_bh  = bh / img_h

                # Clamp bbox to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_bw  = max(0.0, min(1.0, norm_bw))
                norm_bh  = max(0.0, min(1.0, norm_bh))

                line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_bw:.6f} {norm_bh:.6f}"

                points_processed = 0
                kpts = ann.get('keypoints', [])

                for i in range(0, len(kpts), 3):
                    kx = float(kpts[i])
                    ky = float(kpts[i + 1])
                    kv = int(kpts[i + 2])
                    if kv == 0:
                        # Invisible: 0,0,0 YOLO convention
                        norm_kx, norm_ky = 0.0, 0.0
                    else:
                        norm_kx = max(0.0, min(1.0, kx / img_w))
                        norm_ky = max(0.0, min(1.0, ky / img_h))  

                    line += f" {norm_kx:.6f} {norm_ky:.6f} {kv}"
                    points_processed += 1

                # Zero-pad keypoints if fewer than max_keypoints
                while points_processed < self.max_keypoints:
                    line += " 0.000000 0.000000 0"
                    points_processed += 1

                yolo_lines.append(line)

            if yolo_lines:
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines) + '\n')
                files_created += 1

        print(f"  Created : {files_created} label files → {output_dir}")
        if skipped_annotations > 0:
            print(f"  Skipped : {skipped_annotations} annotations (unknown category)")


if __name__ == "__main__":
    IN_DIR  = r"./data"
    OUT_DIR = r"./data/processed/yolo-pose"

    datasets = {
        "train": {
            "json": os.path.join(OUT_DIR, "images", "train", "_annotations.coco.json"),
            "out":  os.path.join(OUT_DIR, "labels", "train"),
        },
        "val": {
            "json": os.path.join(IN_DIR, "labels", "coco_keypoint", "valid", "_annotations.coco.json"),
            "out":  os.path.join(OUT_DIR, "labels", "val"),
        },
        "test": {
            "json": os.path.join(IN_DIR, "labels", "coco_keypoint", "test", "_annotations.coco.json"),
            "out":  os.path.join(OUT_DIR, "labels", "test"),
        },
    }

    print("Starting COCO → YOLO Pose Conversion...")
    converter = CocoToYoloPoseConverter(max_keypoints=4)

    for split_name, paths in datasets.items():
        print(f"\n[{split_name.upper()}]")
        converter.convert(json_path=paths["json"], output_dir=paths["out"])

    print("\n All splits converted!")
