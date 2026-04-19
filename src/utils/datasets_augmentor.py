import json
import os
import cv2
import albumentations as A
import copy

class DatasetsAugmentor:
    def __init__(self, json_path, img_dir, output_img_dir, output_json_path):
        self.json_path = json_path
        self.img_dir = img_dir
        self.output_img_dir = output_img_dir
        self.output_json_path = output_json_path
        
        # Create output directories if they do not exist
        os.makedirs(self.output_img_dir, exist_ok=True)

        #  CORE LOGIC: Color-Safe Augmentation Pipeline
        self.transform = A.Compose([
            # 1. Solve Orientation Bias: Randomly rotate 90 degrees to ensure vertical variants
            A.RandomRotate90(p=0.5), 
            
            # 2. Spatial Transformation: Simulate camera tilts (up to 45 degrees for diagonals)
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45,interpolation=cv2.INTER_CUBIC, p=0.8),
            
            # 3. Lighting Adjustment: Modify only brightness/contrast; NEVER touch Hue.
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  
                contrast_limit=0.02,  
                p=0.4                  
            )  
        ], 
        # Crucial: Instruct Albumentations to calculate and update BBoxes and Keypoints
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def run_augmentation(self, augment_times=3):
        print(f"Loading Master JSON data from: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        # Initialize the structure for the new Augmented JSON
        new_coco = copy.deepcopy(coco)
        new_coco['images'] = []
        new_coco['annotations'] = []
        
        new_img_id = 1
        new_ann_id = 1

        # Iterate through every original image in the dataset
        for img_info in coco['images']:
            orig_img_id = img_info['id']
            img_filename = img_info['file_name']
            img_path = os.path.join(self.img_dir, img_filename)
            
            # Read image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Image not found - {img_path}")
                continue
            
            # Albumentations expects RGB format, while OpenCV reads in BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

            # Extract all annotations associated with the current image
            anns = [ann for ann in coco['annotations'] if ann['image_id'] == orig_img_id]
            
            # ---------------------------------------------------------
            # Step 1: Save the Original Image to the new dataset
            # ---------------------------------------------------------
            orig_out_name = f"orig_{img_filename}"
            cv2.imwrite(os.path.join(self.output_img_dir, orig_out_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            img_info_copy = copy.deepcopy(img_info)
            img_info_copy['id'] = new_img_id
            img_info_copy['file_name'] = orig_out_name
            new_coco['images'].append(img_info_copy)
            
            for ann in anns:
                ann_copy = copy.deepcopy(ann)
                ann_copy['id'] = new_ann_id
                ann_copy['image_id'] = new_img_id
                new_coco['annotations'].append(ann_copy)
                new_ann_id += 1
            new_img_id += 1

            # ---------------------------------------------------------
            # Step 2: Generate Augmented variations based on 'augment_times'
            # ---------------------------------------------------------
            for i in range(augment_times):
                bboxes = [ann['bbox'] for ann in anns]
                category_ids = [ann['category_id'] for ann in anns]
                
                # Format Keypoints: COCO format is [x,y,v, x,y,v], Albumentations needs [(x,y), (x,y)]
                keypoints = []
                for ann in anns:
                    kpts = ann.get('keypoints', [])
                    for j in range(0, len(kpts), 3):
                        keypoints.append((kpts[j], kpts[j+1]))

                # Execute the transformation pipeline
                transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids, keypoints=keypoints)
                
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_keypoints = transformed['keypoints']
                
                # Save the new Augmented image (Convert back to BGR for OpenCV)
                aug_filename = f"aug_{i}_{img_filename}"
                cv2.imwrite(os.path.join(self.output_img_dir, aug_filename), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                
                # Register the new image in the JSON structure
                new_img_info = copy.deepcopy(img_info)
                new_img_info['id'] = new_img_id
                new_img_info['file_name'] = aug_filename
                new_coco['images'].append(new_img_info)

                # Map the dynamically calculated bounding boxes and keypoints back to JSON annotations
                kpt_idx = 0
                for ann_idx, ann in enumerate(anns):
                    new_ann = copy.deepcopy(ann)
                    new_ann['id'] = new_ann_id
                    new_ann['image_id'] = new_img_id
                    new_ann['bbox'] = list(aug_bboxes[ann_idx])
                    
                    # Reconstruct the COCO keypoint format [x, y, v]
                    new_kpts = []
                    old_kpts = ann.get('keypoints', [])
                    for j in range(0, len(old_kpts), 3):
                        new_kpts.append(aug_keypoints[kpt_idx][0]) # New Transformed X
                        new_kpts.append(aug_keypoints[kpt_idx][1]) # New Transformed Y
                        new_kpts.append(old_kpts[j+2])             # Original Visibility flag
                        kpt_idx += 1
                    
                    new_ann['keypoints'] = new_kpts
                    new_coco['annotations'].append(new_ann)
                    new_ann_id += 1
                    
                new_img_id += 1

        # Export the final consolidated JSON file
        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_coco, f, indent=4)
            
        print("\n✅ Data Augmentation Completed Successfully!")
        print(f"📁 Augmented Images saved to: {self.output_img_dir}")
        print(f"📄 Master JSON saved to: {self.output_json_path}")

if __name__ == "__main__":
    JSON_PATH = r"./data/labels/coco_keypoint/train/_annotations.coco.json"
    IMG_DIR = r"./data/labels/coco_keypoint/train"
    
    # Destination paths 
    OUT_IMG_DIR = r"./data/augmented/coco_keypoint/train_augmented"
    OUT_JSON_PATH = r"./data/augmented/coco_keypoint/train_augmented/_annotations.coco.json"
    
    augmentor = DatasetsAugmentor(JSON_PATH, IMG_DIR, OUT_IMG_DIR, OUT_JSON_PATH)
    augmentor.run_augmentation(augment_times=4)