import json
import os

class CocoToYoloPoseConverter:
    def __init__(self, max_keypoints=4):
        self.max_keypoints = max_keypoints
        # Mapping COCO class names to YOLO IDs
        self.target_classes = {
            'board': 0,
            'resistor': 1,
            'wire': 2
        }

    def convert(self, json_path, output_dir):
        print(f"\n Processing file...: {json_path}")
        if not os.path.exists(json_path):
            print(f" File not found: {json_path} Skipping...")
            return
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Category ID Pairing
        coco_to_yolo_id = {}
        for cat in data.get('categories', []):
            cat_name = cat['name'].lower()
            if cat_name in self.target_classes:
                coco_to_yolo_id[cat['id']] = self.target_classes[cat_name]

        # 2. Store image width and height information
        image_info = {img['id']: (img['width'], img['height'], img['file_name']) for img in data.get('images', [])}

        os.makedirs(output_dir, exist_ok=True)
        
        # 3. Group Annotations by image
        annotations_by_image = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        files_created = 0
        skipped_annotations = 0

        # 4. Loop through each image and convert annotations
        for img_id, (img_w, img_h, file_name) in image_info.items():
            if img_id not in annotations_by_image:
                continue

            txt_filename = os.path.splitext(file_name)[0] + '.txt'
            txt_filepath = os.path.join(output_dir, txt_filename)

            yolo_lines = []
            
            for ann in annotations_by_image[img_id]:
                coco_cat_id = ann['category_id']
                
                # Skip classes that are not in target_classes
                if coco_cat_id not in coco_to_yolo_id:
                    skipped_annotations += 1
                    continue
                    
                yolo_class_id = coco_to_yolo_id[coco_cat_id]

                # Normalize 0.0 - 1.0 for bbox and keypoints
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2.0) / img_w
                y_center = (bbox[1] + bbox[3] / 2.0) / img_h
                w = bbox[2] / img_w
                h = bbox[3] / img_h

                line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

                # Convert Keypoints
                points_processed = 0
                if 'keypoints' in ann and len(ann['keypoints']) > 0:
                    kpts = ann['keypoints']
                    for i in range(0, len(kpts), 3):
                        kx, ky, kv = kpts[i], kpts[i+1], kpts[i+2]
                        # Normalize coordinates
                        norm_kx = kx / img_w if kx > 0 else 0.0
                        norm_ky = ky / img_h if ky > 0 else 0.0
                        line += f" {norm_kx:.6f} {norm_ky:.6f} {kv}"
                        points_processed += 1

                # --- ZERO-PADDING LOGIC ---
                while points_processed < self.max_keypoints:
                    line += " 0.000000 0.000000 0"
                    points_processed += 1

                yolo_lines.append(line)

            if yolo_lines:
                with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                    txt_file.write('\n'.join(yolo_lines))
                files_created += 1

        print(f" Conversion completed: {files_created} files -> Saved to {output_dir}")
        if skipped_annotations > 0:
            print(f" Filtered out classes: {skipped_annotations} annotations")

if __name__ == "__main__":
    IN_DIR = r"./data" 
    OUT_DIR = r"./data/processed/yolo_pose"
    
    datasets = {
        "train": {
            "json": os.path.join(IN_DIR,"augmented","coco_keypoint", "train_augmented", "_annotations.coco.json"),
            "out": os.path.join(OUT_DIR,"labels",  "train")
        },
        "val": {
            "json": os.path.join(IN_DIR,"labels","coco_keypoint", "valid", "_annotations.coco.json"),
            "out": os.path.join(OUT_DIR,"labels",  "val")
        },
        "test": {
            "json": os.path.join(IN_DIR,"labels","coco_keypoint", "test", "_annotations.coco.json"),
            "out": os.path.join(OUT_DIR, "labels", "test")
        }
    }
    
    print("Starting COCO to YOLO Pose Conversion...")
    converter = CocoToYoloPoseConverter(max_keypoints=4)
    
    for split_name, paths in datasets.items():
        converter.convert(json_path=paths["json"], output_dir=paths["out"])
        
    print("\n Conversion completed for all datasets!")