import json
import os

class CocoToYoloPoseConverter:
    def __init__(self, json_path, output_dir):
        self.json_path = json_path
        self.output_dir = output_dir
        self.max_keypoints = 4  # Enforce a maximum of 4 points (based on the breadboard corners)
        
        # Mapping COCO class names to YOLO IDs (starting at 0)
        # Notice we omit 'bodyresist' here to automatically filter it out.
        self.target_classes = {
            'board': 0,
            'resistor': 1,
            'wire': 2
        }

    def convert(self):
        print(f"Reading file: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Create a mapping table from COCO Category ID to YOLO ID
        coco_to_yolo_id = {}
        for cat in data['categories']:
            cat_name = cat['name'].lower()
            if cat_name in self.target_classes:
                coco_to_yolo_id[cat['id']] = self.target_classes[cat_name]

        # 2. Store image dimensions to calculate Normalized coordinates (0.0 - 1.0)
        image_info = {img['id']: (img['width'], img['height'], img['file_name']) for img in data['images']}

        # Create the output labels directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        files_created = 0
        skipped_annotations = 0

        # 3. Process image by image
        for img_id, (img_w, img_h, file_name) in image_info.items():
            if img_id not in annotations_by_image:
                continue

            # Change file extension from .jpg/.png to .txt
            txt_filename = os.path.splitext(file_name)[0] + '.txt'
            txt_filepath = os.path.join(self.output_dir, txt_filename)

            yolo_lines = []
            
            for ann in annotations_by_image[img_id]:
                coco_cat_id = ann['category_id']
                
                # If the class is not in our target list (e.g., bodyresist), skip it
                if coco_cat_id not in coco_to_yolo_id:
                    skipped_annotations += 1
                    continue
                    
                yolo_class_id = coco_to_yolo_id[coco_cat_id]

                # Convert Bounding Box [x_top_left, y_top_left, width, height]
                # to YOLO Format [x_center, y_center, width, height] Normalized (0.0 - 1.0)
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2.0) / img_w
                y_center = (bbox[1] + bbox[3] / 2.0) / img_h
                w = bbox[2] / img_w
                h = bbox[3] / img_h

                line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

                # Convert Keypoints
                if 'keypoints' in ann and len(ann['keypoints']) > 0:
                    kpts = ann['keypoints']
                    # Data arrives as [x1, y1, v1, x2, y2, v2, ...]
                    
                    points_processed = 0
                    for i in range(0, len(kpts), 3):
                        kx, ky, kv = kpts[i], kpts[i+1], kpts[i+2]
                        # Normalize coordinates
                        norm_kx = kx / img_w if kx > 0 else 0.0
                        norm_ky = ky / img_h if ky > 0 else 0.0
                        line += f" {norm_kx:.6f} {norm_ky:.6f} {kv}"
                        points_processed += 1

                    # --- ZERO-PADDING LOGIC ---
                    # If this object has fewer than 4 points (like a resistor), pad with 0.0 0.0 0
                    while points_processed < self.max_keypoints:
                        line += " 0.000000 0.000000 0"
                        points_processed += 1

                yolo_lines.append(line)

            # Save the .txt file
            if yolo_lines:
                with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                    txt_file.write('\n'.join(yolo_lines))
                files_created += 1

        print("\n" + "="*50)
        print("✅ COCO -> YOLO-Pose conversion complete!")
        print(f"📁 Total .txt files created: {files_created}")
        print(f"🗑️ Skipped annotations (e.g., bodyresist): {skipped_annotations}")
        print(f"📍 Results saved to: {self.output_dir}")
        print("="*50)

if __name__ == "__main__":
    # 1. Define your Master JSON file (Ensure the path is correct!)
    INPUT_MASTER_JSON = r"./data/labels/coco_keypoint/master_dataset_keypoints.json"
    
    # 2. Destination folder for the .txt files
    OUTPUT_YOLO_DIR = r"./data/yolo_pose_labels"
    
    converter = CocoToYoloPoseConverter(INPUT_MASTER_JSON, OUTPUT_YOLO_DIR)
    converter.convert()