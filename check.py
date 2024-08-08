import os
from collections import defaultdict
from datetime import datetime

# Paths
DATASET_SPLIT_DIR = 'datasets/CN_dataset_obj_detection_04_23/split_data/'
DUPLICATE_CHECK_RESULT = 'duplicate_check_result.txt'

def extract_image_key(image_path):
    return os.path.basename(image_path).rsplit('.', 1)[0]

def check_for_duplicates(data_dir):
    image_keys = defaultdict(list)
    splits = ['train', 'test', 'eval']
    
    # Iterate through each split directory
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    image_key = extract_image_key(img_path)
                    image_keys[image_key].append((split, img_path))
    
    duplicates = {key: paths for key, paths in image_keys.items() if len(paths) > 1}
    
    return duplicates

# Check for duplicates
duplicates = check_for_duplicates(DATASET_SPLIT_DIR)

# Print and save the results
with open(DUPLICATE_CHECK_RESULT, 'w') as f:
    if duplicates:
        print(f"Found {len(duplicates)} duplicates:")
        f.write(f"Found {len(duplicates)} duplicates:\n")
        for key, paths in duplicates.items():
            print(f"{key}:")
            f.write(f"{key}:\n")
            for split, path in paths:
                print(f"  - {split}: {path}")
                f.write(f"  - {split}: {path}\n")
    else:
        print("No duplicates found.")
        f.write("No duplicates found.\n")

print(f"***{datetime.now()}: Duplicate check complete. Results saved to {DUPLICATE_CHECK_RESULT}***")
