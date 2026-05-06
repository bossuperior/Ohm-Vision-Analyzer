import json
import os
import cv2
import albumentations as A
import copy
import numpy as np

MIN_BRIGHTNESS = 30


class DatasetsAugmentor:
    def __init__(self, json_path, img_dir, output_img_dir, output_json_path):
        self.json_path       = json_path
        self.img_dir         = img_dir
        self.output_img_dir  = output_img_dir
        self.output_json_path = output_json_path

        os.makedirs(self.output_img_dir, exist_ok=True)

        # ── Augmentation pipeline
        self.transform = A.Compose([
            # 1. Lighting
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.2,
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),

            # 3. Color variation — NO hue shift (preserves resistor band colors)
            A.HueSaturationValue(
                hue_shift_limit=0,
                sat_shift_limit=20,
                val_shift_limit=30,
                p=0.5
            ),

            # 4. Blur / noise — simulate defocus and sensor noise seen in dataset
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),

            # 5. CLAHE — simulate high-contrast lighting on breadboard surface
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
        ],
        # Crucial: Instruct Albumentations to calculate and update BBoxes and Keypoints
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def run_augmentation(self, augment_times: int = 4):
        print(f"Loading Master JSON: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        new_coco               = copy.deepcopy(coco)
        new_coco['images']     = []
        new_coco['annotations'] = []
        new_img_id = 1
        new_ann_id = 1

        # Iterate through every original image in the dataset
        for img_info in coco['images']:
            orig_img_id  = img_info['id']
            img_filename = img_info['file_name']
            img_path     = os.path.join(self.img_dir, img_filename)

            image = cv2.imread(img_path)
            if image is None:
                print(f"  [skip] image not found: {img_path}")
                continue

            # Albumentations expects RGB format, while OpenCV reads in BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            anns = [a for a in coco['annotations'] if a['image_id'] == orig_img_id]

            # Save the Original Image to the new dataset
            orig_out_name = f"orig_{img_filename}"
            cv2.imwrite(os.path.join(self.output_img_dir, orig_out_name),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            img_info_copy              = copy.deepcopy(img_info)
            img_info_copy['id']        = new_img_id
            img_info_copy['file_name'] = orig_out_name
            new_coco['images'].append(img_info_copy)

            for ann in anns:
                ac              = copy.deepcopy(ann)
                ac['id']        = new_ann_id
                ac['image_id']  = new_img_id
                new_coco['annotations'].append(ac)
                new_ann_id += 1
            new_img_id += 1

            # Generate Augmented variations based on 'augment_times'
            for aug_i in range(augment_times):
                bboxes       = [a['bbox'] for a in anns]
                category_ids = [a['category_id'] for a in anns]

                # Flatten all keypoints (maintain order for index tracking)
                all_kpts_xy  = []   # [(x, y), ...] fed to albumentations
                all_kpts_vis = []   # [v, ...]       original visibility flags
                ann_kpt_slice = []  # [(start, end)] index range per annotation

                # Format Keypoints: COCO format is [x,y,v, x,y,v], Albumentations needs [(x,y), (x,y)]
                for ann in anns:
                    flat = ann.get('keypoints', [])
                    start = len(all_kpts_xy)
                    for j in range(0, len(flat), 3):
                        all_kpts_xy.append((flat[j], flat[j + 1]))
                        all_kpts_vis.append(flat[j + 2])
                    ann_kpt_slice.append((start, len(all_kpts_xy)))

                # Execute the transformation pipeline
                try:
                    t = self.transform(
                        image=image,
                        bboxes=bboxes,
                        category_ids=category_ids,
                        keypoints=all_kpts_xy
                    )
                except Exception as e:
                    print(f"  [skip aug {aug_i} '{img_filename}'] {e}")
                    continue

                aug_img      = t['image']
                aug_bboxes   = t['bboxes']
                aug_kpts_out = t['keypoints']   # same length as all_kpts_xy (remove_invisible=False)

                aug_H, aug_W = aug_img.shape[:2]

                # Skip if augmented image is too dark
                brightness = int(np.mean(aug_img))
                if brightness < MIN_BRIGHTNESS:
                    print(f"  [skip aug {aug_i} '{img_filename}'] too dark (brightness={brightness})")
                    continue

                # Save augmented image
                aug_filename = f"aug_{aug_i}_{img_filename}"
                cv2.imwrite(os.path.join(self.output_img_dir, aug_filename),
                            cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

                new_img_info              = copy.deepcopy(img_info)
                new_img_info['id']        = new_img_id
                new_img_info['file_name'] = aug_filename
                new_img_info['width']     = aug_W
                new_img_info['height']    = aug_H
                new_coco['images'].append(new_img_info)

                # Rebuild annotations
                for ann_idx, ann in enumerate(anns):
                    s, e = ann_kpt_slice[ann_idx]

                    pts_after = []
                    for k in range(s, e):
                        px, py = float(aug_kpts_out[k][0]), float(aug_kpts_out[k][1])
                        orig_v  = all_kpts_vis[k]
                        in_bounds = (0 <= px < aug_W) and (0 <= py < aug_H)
                        new_v     = orig_v if in_bounds else 0
                        pts_after.append((px, py, new_v))

                    # Flatten back to COCO [x, y, v, x, y, v, ...]
                    new_kpts = []
                    for px, py, v in pts_after:
                        new_kpts += [px, py, v]

                    new_ann              = copy.deepcopy(ann)
                    new_ann['id']        = new_ann_id
                    new_ann['image_id']  = new_img_id
                    new_ann['bbox']      = list(aug_bboxes[ann_idx]) \
                                           if ann_idx < len(aug_bboxes) else ann['bbox']
                    new_ann['keypoints'] = new_kpts
                    new_coco['annotations'].append(new_ann)
                    new_ann_id += 1

                new_img_id += 1

        # Export
        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_coco, f, indent=4)

        print("\n Augmentation complete!")
        print(f"  Images → {self.output_img_dir}")
        print(f"  JSON   → {self.output_json_path}")


if __name__ == "__main__":
    JSON_PATH     = r"./data/labels/coco_keypoint/train/_annotations.coco.json"
    IMG_DIR       = r"./data/labels/coco_keypoint/train"
    OUT_IMG_DIR   = r"./data/augmented/coco_keypoint/train_augmented"
    OUT_JSON_PATH = r"./data/augmented/coco_keypoint/train_augmented/_annotations.coco.json"

    augmentor = DatasetsAugmentor(JSON_PATH, IMG_DIR, OUT_IMG_DIR, OUT_JSON_PATH)
    augmentor.run_augmentation(augment_times=4)
