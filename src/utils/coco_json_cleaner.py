import json

input_json_path = r"./data/augmented/coco_keypoint/train_augmented/_annotations.coco.json" 
output_json_path = r"./data/augmented/coco_keypoint/train_augmented/_annotations_cleaned.coco.json"

bad_keywords = [
    "aug_2_img_264", 
    "aug_3_img_264"
]

print(" Scanning the JSON file for corrupted images...")
with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Step A: Identify the Image IDs of the corrupted files
bad_image_ids = []
for img in data.get('images', []):
    for keyword in bad_keywords:
        if keyword in img['file_name']:
            bad_image_ids.append(img['id'])

if not bad_image_ids:
    print("No matching corrupted files found, or they have already been removed.")
else:
    print(f"Found {len(bad_image_ids)} suspicious images (Image IDs: {bad_image_ids})")

    # Step B: Filter out the images that are in the bad list
    clean_images = [img for img in data['images'] if img['id'] not in bad_image_ids]
    
    # Step C: Filter out the annotations linked to the bad image IDs
    clean_annotations = [ann for ann in data['annotations'] if ann['image_id'] not in bad_image_ids]

    # Update the dataset with the cleaned data
    data['images'] = clean_images
    data['annotations'] = clean_annotations

    # Step D: Save the cleaned dataset to a new JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f) # No indent to minimize file size

    print("\n=========================================")
    print(f"Success! Corrupted data has been completely removed.")
    print(f"Images: Reduced from {len(clean_images) + len(bad_image_ids)} -> {len(clean_images)} images.")
    print(f"Annotations: Synchronized to match the remaining images.")
    print(f"Cleaned file saved to: {output_json_path}")
    print("=========================================")