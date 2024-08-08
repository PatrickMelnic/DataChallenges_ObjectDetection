import os
import json
import matplotlib.pyplot as plt
from datasets import load_from_disk
import numpy as np
from datetime import datetime
from PIL import Image

# Function to print current datetime with a message
def print_with_time(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# Customizable paths
DATASET_PATH = 'datasets/multilabel'
CLASS_MAPPING_FILE = 'class_index_mapping.json'
IMAGE_SAVE_DIR = 'visualizations/'

# Ensure the save directory exists
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

print_with_time("Starting the dataset loading process.")

# Load the dataset
dataset_dict = load_from_disk(DATASET_PATH)
train_dataset = dataset_dict["train"]
val_dataset = dataset_dict["eval"]
test_dataset = dataset_dict["test"]
print_with_time("Dataset loaded.")

# Load the class index mapping
with open(CLASS_MAPPING_FILE, 'r') as f:
    class_index_mapping = json.load(f)
print_with_time("Class index mapping loaded.")

num_labels = len(class_index_mapping)
print(f"Number of labels: {num_labels}")

# Function to count images per class
def count_images_per_class(dataset):
    class_counts = np.zeros(num_labels)
    for sample in dataset:
        class_counts += np.array(sample['encoded_label'])
    return class_counts

# Function to count the number of labels per image
def count_labels_per_image(dataset):
    labels_per_image = []
    for sample in dataset:
        labels_per_image.append(sum(sample['encoded_label']))
    return labels_per_image

# Function to check for duplicate images within and across datasets
def check_for_duplicates(datasets):
    image_paths = {}
    duplicate_paths = set()

    for split_name, dataset in datasets.items():
        print_with_time(f"Checking for duplicates in {split_name} split.")
        for sample in dataset:
            image_filename = os.path.basename(sample['image_path']).split('.')[0]
            if image_filename in image_paths:
                duplicate_paths.add(image_filename)
                print_with_time(f"Duplicate found: {image_filename} in {split_name} and {image_paths[image_filename]}")
            else:
                image_paths[image_filename] = split_name

    if duplicate_paths:
        print_with_time(f"Found {len(duplicate_paths)} duplicates:")
        for path in duplicate_paths:
            print(path)
    else:
        print_with_time("No duplicates found.")

# Combine datasets for duplicate check
datasets = {
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset
}

# Check for duplicates
check_for_duplicates(datasets)

# Debug function to print sample labels and images
def print_sample_info(dataset, split_name, num_samples=5):
    print_with_time(f"Printing sample info for {split_name} split:")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        img_path = sample['image_path']
        image = Image.open(img_path).convert("RGB")
        print(f"Image Path: {img_path}")
        print(f"Labels (encoded): {sample['encoded_label']}")
        print(f"Labels (human-readable): {sample['label']}")
        image.show(title=f"{split_name} Sample {i+1}")
        
# Print sample info to check if the dataset is multi-label
print_sample_info(train_dataset, "train")
print_sample_info(val_dataset, "val")
print_sample_info(test_dataset, "test")

print_with_time("Counting images per class in training dataset.")
train_class_counts = count_images_per_class(train_dataset)
print_with_time("Counting images per class in validation dataset.")
val_class_counts = count_images_per_class(val_dataset)
print_with_time("Counting images per class in test dataset.")
test_class_counts = count_images_per_class(test_dataset)

# Combine counts for visualization
total_class_counts = train_class_counts + val_class_counts + test_class_counts

print_with_time("Generating plot for number of images per class.")
# Plot number of images per class
plt.figure(figsize=(15, 7))
plt.bar(range(num_labels), total_class_counts, color='skyblue')
plt.xlabel('Class Index')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class')
plt.savefig(os.path.join(IMAGE_SAVE_DIR, 'images_per_class.png'))
print_with_time("Plot for number of images per class saved.")

print_with_time("Counting labels per image in training dataset.")
train_labels_per_image = count_labels_per_image(train_dataset)
print_with_time("Counting labels per image in validation dataset.")
val_labels_per_image = count_labels_per_image(val_dataset)
print_with_time("Counting labels per image in test dataset.")
test_labels_per_image = count_labels_per_image(test_dataset)

# Combine counts for visualization
total_labels_per_image = train_labels_per_image + val_labels_per_image + test_labels_per_image

# Debugging: Print the number of images that have 1, 2, 3, etc. labels
unique, counts = np.unique(total_labels_per_image, return_counts=True)
label_distribution = dict(zip(unique, counts))
print_with_time("Number of images with specific number of labels:")
for num_labels, count in label_distribution.items():
    print(f"Images with {num_labels} labels: {count}")

print_with_time("Generating histogram for number of labels per image.")
# Plot histogram of labels per image
plt.figure(figsize=(10, 7))
plt.hist(total_labels_per_image, bins=range(1, num_labels + 2), edgecolor='black')
plt.xlabel('Number of Labels')
plt.ylabel('Number of Images')
plt.title('Histogram of Number of Labels per Image')
plt.savefig(os.path.join(IMAGE_SAVE_DIR, 'labels_per_image_histogram.png'))
print_with_time("Histogram for number of labels per image saved.")

print('Visualizations saved in:', IMAGE_SAVE_DIR)

