import json
import os

class CocoKeypointMerger:
    """
    A class to merge multiple COCO keypoint JSON annotation files into a single master file.
    It includes an automated post-processing step to spatially sort keypoints (Left-to-Right)
    for specific categories to eliminate human labeling inconsistencies.
    """
    
    def __init__(self, json_files, output_filename, categories_to_sort=None):
        """
        Initialize the merger with input files, output filename, and filtering lists.
        
        :param categories_to_sort: List of category names (e.g., ['resistor', 'wire']) 
                                   that should have their keypoints automatically sorted.
        """
        self.json_files = json_files
        self.output_filename = output_filename
        
        # Convert to lowercase for case-insensitive matching
        self.categories_to_sort = [cat.lower() for cat in (categories_to_sort or [])]
        
        self.merged_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        self.image_id_offset = 1
        self.annotation_id_offset = 1
        self.next_category_id = 1
        
        self.category_name_to_id = {}
        # We need a reverse mapping to check the category name during annotation processing
        self.global_id_to_category_name = {}

    def _process_categories(self, data):
        """Map categories and build reverse lookups for the sorting logic."""
        cat_id_mapping = {}
        for cat in data.get('categories', []):
            cat_name = cat['name']
            
            if cat_name not in self.category_name_to_id:
                self.category_name_to_id[cat_name] = self.next_category_id
                self.global_id_to_category_name[self.next_category_id] = cat_name
                
                new_cat = cat.copy()
                new_cat['id'] = self.next_category_id
                self.merged_data['categories'].append(new_cat)
                self.next_category_id += 1
                
            cat_id_mapping[cat['id']] = self.category_name_to_id[cat_name]
            
        return cat_id_mapping

    def _process_images(self, data):
        """Assign global IDs to images."""
        img_id_mapping = {}
        for img in data.get('images', []):
            old_img_id = img['id']
            new_img_id = self.image_id_offset
            img_id_mapping[old_img_id] = new_img_id
            
            new_img = img.copy()
            new_img['id'] = new_img_id
            self.merged_data['images'].append(new_img)
            
            self.image_id_offset += 1
            
        return img_id_mapping

    def _sort_keypoints(self, keypoints_flat_list):
        """
        Geometrically sorts a flat list of COCO keypoints [x1, y1, v1, x2, y2, v2, ...].
        Sorting priority: Primary by X-axis (Left-to-Right), Secondary by Y-axis (Top-to-Bottom).
        """
        # 1. Chunk the flat list into grouped tuples of (x, y, v)
        points = [keypoints_flat_list[i:i+3] for i in range(0, len(keypoints_flat_list), 3)]
        
        # 2. Sort the points based on X coordinate first, then Y coordinate
        # Note: If a point is invisible (0, 0, 0), it will naturally move to the front.
        points.sort(key=lambda p: (p[0], p[1]))
        
        # 3. Flatten the sorted tuples back into a single list
        sorted_flat_list = [val for point in points for val in point]
        
        return sorted_flat_list

    def _process_annotations(self, data, img_id_mapping, cat_id_mapping):
        """Process annotations and apply sorting if the category matches."""
        for ann in data.get('annotations', []):
            new_ann = ann.copy()
            
            global_cat_id = cat_id_mapping[ann['category_id']]
            category_name = self.global_id_to_category_name[global_cat_id]
            
            new_ann['id'] = self.annotation_id_offset
            new_ann['image_id'] = img_id_mapping[ann['image_id']]
            new_ann['category_id'] = global_cat_id
            
            # --- AUTO-SORTING LOGIC TRIGGER ---
            # Check if this annotation has keypoints AND belongs to our target categories
            if 'keypoints' in new_ann and category_name.lower() in self.categories_to_sort:
                new_ann['keypoints'] = self._sort_keypoints(new_ann['keypoints'])
            
            self.merged_data['annotations'].append(new_ann)
            self.annotation_id_offset += 1

    def save_output(self):
        """Save the merged and sorted dataset."""
        try:
            with open(self.output_filename, 'w', encoding='utf-8') as f:
                json.dump(self.merged_data, f, indent=4)
                
            print("\n" + "="*50)
            print("✅ Merge & Keypoint Sorting completed successfully!")
            print(f"📁 Saved output to  : {self.output_filename}")
            print(f"🖼️ Total images     : {len(self.merged_data['images'])}")
            print(f"🎯 Total annotations: {len(self.merged_data['annotations'])}")
            print(f"🧹 Auto-sorted categories: {self.categories_to_sort}")
            print("="*50)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save output - {e}")

    def merge(self):
        """Main execution pipeline."""
        for file_path in self.json_files:
            if not os.path.exists(file_path):
                print(f"⚠️ Warning: File not found -> {file_path}")
                continue
                
            print(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cat_id_mapping = self._process_categories(data)
            img_id_mapping = self._process_images(data)
            self._process_annotations(data, img_id_mapping, cat_id_mapping)

        self.save_output()


if __name__ == "__main__":
    # 1. Define the list of JSON files to merge (update paths as needed)
    input_json_files = [
       "./data/labels/coco_keypoint/coco_keypoint_labels_1.json",
        "./data/labels/coco_keypoint/coco_keypoint_labels_2.json",
        "./data/labels/coco_keypoint/coco_keypoint_labels_3.json",
        "./data/labels/coco_keypoint/coco_keypoint_labels_4.json"
    ]
    
    # 2. Get output filename (optional: can be customized or derived from input files)
    if input_json_files:
        output_dir = os.path.dirname(input_json_files[0])
        output_filename = os.path.join(output_dir, "master_dataset_keypoints.json")
    else:
        output_filename = "master_dataset_keypoints.json"
    
    # 3. DEFINING THE TARGET CATEGORIES FOR SORTING
    # IMPORTANT: Do not include 'breadboard' here to preserve its clockwise corner mapping!
    target_sort_categories = ['resistor', 'wire']
    
    # 4. Instantiate and run
    merger = CocoKeypointMerger(
        json_files=input_json_files, 
        output_filename=output_filename,
        categories_to_sort=target_sort_categories
    )
    merger.merge()