import os
from collections import defaultdict
from datasets import Dataset, Image
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image as PILImage
import shutil
import random
import json
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

data_dir = 'datasets/CN_dataset_obj_detection_04_23/dataset_obj_detection/'
data_dir_split = 'datasets/CN_dataset_obj_detection_04_23/split_data/'
data_save_dir = 'datasets/multilabel/'
limiter = 20  # Minimum number of images per class

# Delete split data directories if they exist
print(f"***{datetime.now()}: Delete dirs***")
if os.path.exists(data_dir_split):
    shutil.rmtree(data_dir_split)

def get_unique_images_and_labels(data_dir):
    image_files = defaultdict(list)
    label_paths = defaultdict(list)
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                image_key = os.path.basename(img_path).rsplit('.', 1)[0]
                image_files[image_key].append(img_path)
                label_paths[image_key].append(label)
    
    combined_files = []
    combined_labels = []
    duplicate_counts = defaultdict(int)
    
    for image_key, paths in image_files.items():
        combined_files.append(paths[0])  # Use the first path as the representative
        combined_labels.append(list(set(label_paths[image_key])))  # Combine and deduplicate labels
        duplicate_counts[len(paths)] += 1
    
    return combined_files, combined_labels, duplicate_counts

def split_data(image_files, labels, output_dir, train_ratio=0.8, test_ratio=0.1, eval_ratio=0.1):
    # Create output directories if they don't exist
    for split in ['train', 'test', 'eval']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    print(f"***{datetime.now()}: Make dirs***")
    
    data = list(zip(image_files, labels))
    random.shuffle(data)
    
    total = len(data)
    train_end = int(train_ratio * total)
    test_end = train_end + int(test_ratio * total)
    
    train_data = data[:train_end]
    test_data = data[train_end:test_end]
    eval_data = data[test_end:]
    
    split_data = {
        'train': train_data,
        'test': test_data,
        'eval': eval_data
    }
    
    # Save split data
    for split, split_data in split_data.items():
        for img_path, label in split_data:
            label_dir = os.path.join(output_dir, split, '_'.join(label))
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(img_path, label_dir)
    
    # Print split statistics
    print(f"Train split: {len(train_data)} images")
    print(f"Test split: {len(test_data)} images")
    print(f"Eval split: {len(eval_data)} images")

# Get unique images and labels
unique_files, unique_labels, duplicate_counts = get_unique_images_and_labels(data_dir)

# Print duplicate statistics
print("Duplicate statistics:")
for count, num_files in duplicate_counts.items():
    print(f"{num_files} images with {count} duplicates")

# Save unique image paths and labels to a JSON file
unique_data = {
    "files": unique_files,
    "labels": unique_labels
}
with open(os.path.join(data_save_dir, 'unique_images_labels.json'), 'w') as f:
    json.dump(unique_data, f)

# Split the data and save the splits
split_data(unique_files, unique_labels, data_dir_split)

print(f"***{datetime.now()}: Data processing complete***")
