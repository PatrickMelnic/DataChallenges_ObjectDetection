import os
from datasets import Dataset, Image
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image as PILImage
import shutil
import random
from collections import defaultdict
from datetime import datetime
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

data_dir = 'datasets/CN_dataset_obj_detection_04_23/dataset_obj_detection/'
data_dir_split ='datasets/CN_dataset_obj_detection_04_23/split_data/'
data_save_dir='datasets/multilabel/'
limiter = 20 #how many pictures in class at least

# Delete split data directories if they exist
print(f"***{datetime.now()}: Delete dirs***")
if os.path.exists(data_dir_split):
    shutil.rmtree(data_dir_split)

def split_data(data_dir, output_dir, train_ratio=0.8, test_ratio=0.1, eval_ratio=0.1):
    # Create output directories if they don't exist
    for split in ['train', 'test', 'eval']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        
    print(f"***{datetime.now()}: Make dirs***")
    
    # Collect image paths and labels
    label_paths = defaultdict(list)
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                label_paths[label].append(img_path)

    filtered_label_paths = {label: paths for label, paths in label_paths.items() if len(paths) >= limiter}
    
    # Print the number of images collected and save to text file
    total_images = sum(len(paths) for paths in filtered_label_paths.values())
    print(f"Total images collected: {total_images}")
    print(f"Label distribution: {json.dumps({label: len(paths) for label, paths in filtered_label_paths.items()}, indent=2)}")

    # Save the dictionary to a text file
    with open('label_distribution.txt', 'w') as f:
        f.write(f"Total images collected: {total_images}\n")
        f.write(json.dumps({label: len(paths) for label, paths in filtered_label_paths.items()}, indent=2))
    
    print(f"***{datetime.now()}: Collecting paths***")

    # Split and copy files
    split_counts = {'train': 0, 'test': 0, 'eval': 0}
    for label, paths in filtered_label_paths.items():
        random.shuffle(paths)
        total = len(paths)
        train_end = int(train_ratio * total)
        test_end = train_end + int(test_ratio * total)

        train_paths = paths[:train_end]
        test_paths = paths[train_end:test_end]
        eval_paths = paths[test_end:]

        # Ensure test and eval have at least one image
        if not test_paths:
            test_paths.append(train_paths.pop())
        if not eval_paths:
            eval_paths.append(train_paths.pop())

        split_counts['train'] += len(train_paths)
        split_counts['test'] += len(test_paths)
        split_counts['eval'] += len(eval_paths)

        # Helper function to copy files
        def copy_files(file_paths, split):
            for img_path in file_paths:
                split_dir = os.path.join(output_dir, split, label)
                os.makedirs(split_dir, exist_ok=True)
                shutil.copy(img_path, split_dir)

        copy_files(train_paths, 'train')
        copy_files(test_paths, 'test')
        copy_files(eval_paths, 'eval')

    # Print split statistics
    print(f"Train split: {split_counts['train']} images across {len(os.listdir(os.path.join(output_dir, 'train')))} subfolders")
    print(f"Test split: {split_counts['test']} images across {len(os.listdir(os.path.join(output_dir, 'test')))} subfolders")
    print(f"Eval split: {split_counts['eval']} images across {len(os.listdir(os.path.join(output_dir, 'eval')))} subfolders")

split_data(data_dir, data_dir_split)

print(f"***{datetime.now()}: Split data, you can now comment the above code***")







"""
# Function to get image files and labels
def get_image_files_and_labels(data_dir):
    image_files = []
    labels = []
    label_names = os.listdir(data_dir)
    for label in label_names:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                if img_path not in image_files:
                    image_files.append(img_path)
                    labels.append([label])
                else:
                    index = image_files.index(img_path)
                    labels[index].append(label)
    return image_files, labels



# Get image files and labels
image_files, labels = get_image_files_and_labels(data_dir)

print(f"Found {len(image_files)} images.")
print(f"Example labels: {labels[:5]}")

# Encode labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(labels)

# Load images as PIL objects
images = [PILImage.open(img_path) for img_path in image_files]

# Create HuggingFace Dataset object
dataset = Dataset.from_dict({
    'image_path': image_files,
    'image': images,
    'label': labels,
    'encoded_label': encoded_labels.tolist()
})

# Cast the image column to the Image type
dataset = dataset.cast_column('image', Image())

# Print first example to verify
print(f"First example: {dataset[0]}")

# Save the dataset to disk
dataset.save_to_disk(data_save_dir)  # Replace with your desired save path

# To load the dataset from disk later
loaded_dataset = Dataset.load_from_disk(data_save_dir)
print(f"Loaded dataset: {loaded_dataset[0]}")
"""
