"""
import os
from datasets import Dataset, DatasetDict, Image
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image as PILImage
import json


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


data_dir ='datasets/CN_dataset_obj_detection_04_23/split_data/'
data_save_dir='datasets/multilabel/'
class_mapping_file = 'class_index_mapping.json'

def get_image_files_and_labels(data_dir):
    image_files = []
    labels = []
    label_names = os.listdir(data_dir)
    for label in label_names:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                image_files.append(img_path)
                labels.append([label])  # Append raw label
    return image_files, labels

def create_hf_dataset(split_dir):
    image_files, labels = get_image_files_and_labels(split_dir)
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(labels)  # Binary encode labels

    # Store the class to index mapping
    class_index_mapping = {cls: idx for idx, cls in enumerate(mlb.classes_)}

    dataset = Dataset.from_dict({
        'image_path': image_files,  # File paths
        'label': labels,  # Raw labels
        'encoded_label': encoded_labels.tolist()  # Encoded labels
    })

    # Add the image column
    def load_image(example):
        example['image'] = PILImage.open(example['image_path'])
        return example

    dataset = dataset.map(load_image, num_proc=4)

    # Cast the image column to the Image type
    dataset = dataset.cast_column('image', Image(decode=True))  # Use lazy decoding

    return dataset, class_index_mapping

# Create datasets for train, test, and eval splits
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
eval_dir = os.path.join(data_dir, 'eval')

train_dataset, train_mapping = create_hf_dataset(train_dir)
test_dataset, test_mapping = create_hf_dataset(test_dir)
eval_dataset, eval_mapping = create_hf_dataset(eval_dir)

# Ensure that all mappings are the same and save only once
assert train_mapping == test_mapping == eval_mapping, "Class index mappings are not consistent across splits!"
class_index_mapping = train_mapping

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'eval': eval_dataset
})

# Save the DatasetDict to disk
dataset_dict.save_to_disk(data_save_dir)

# Save the class index mapping to a JSON file
with open(class_mapping_file, 'w') as f:
    json.dump(class_index_mapping, f)

# To load the datasets from disk later
loaded_dataset_dict = DatasetDict.load_from_disk(data_save_dir)
print(f"Loaded train dataset: {loaded_dataset_dict['train'][0]}")
print(f"Loaded test dataset: {loaded_dataset_dict['test'][0]}")
print(f"Loaded eval dataset: {loaded_dataset_dict['eval'][0]}")

# Load and print the class to index mapping
with open(class_mapping_file, 'r') as f:
    loaded_class_index_mapping = json.load(f)
print("Class to Index mapping:", loaded_class_index_mapping)

# Print the entire dataset structure
print(dataset_dict)
"""
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

import os
from collections import defaultdict
from datasets import Dataset, DatasetDict, Image
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image as PILImage
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

data_dir = 'datasets/CN_dataset_obj_detection_04_23/split_data/'
data_save_dir = 'datasets/multilabel/'
class_mapping_file = 'class_index_mapping.json'

def get_image_files_and_labels(data_dir):
    image_files = defaultdict(list)
    label_names = os.listdir(data_dir)
    for label in label_names:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                image_key = os.path.basename(img_path).rsplit('.', 1)[0]
                image_files[image_key].append((img_path, label))
    return image_files

def combine_labels(image_files):
    combined_files = []
    combined_labels = []
    for image_key, files_and_labels in image_files.items():
        paths, labels = zip(*files_and_labels)
        combined_files.append(paths[0])  # Use the first path as the representative
        combined_labels.append(list(set(labels)))  # Combine and deduplicate labels
    return combined_files, combined_labels

def create_hf_dataset(split_dir):
    image_files = get_image_files_and_labels(split_dir)
    combined_files, combined_labels = combine_labels(image_files)
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(combined_labels)  # Binary encode labels

    # Store the class to index mapping
    class_index_mapping = {cls: idx for idx, cls in enumerate(mlb.classes_)}

    dataset = Dataset.from_dict({
        'image_path': combined_files,  # File paths
        'label': combined_labels,  # Raw labels
        'encoded_label': encoded_labels.tolist()  # Encoded labels
    })

    # Add the image column
    def load_image(example):
        example['image'] = PILImage.open(example['image_path'])
        return example

    dataset = dataset.map(load_image, num_proc=4)

    # Cast the image column to the Image type
    dataset = dataset.cast_column('image', Image(decode=True))  # Use lazy decoding

    return dataset, class_index_mapping

# Create datasets for train, test, and eval splits
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
eval_dir = os.path.join(data_dir, 'eval')

train_dataset, train_mapping = create_hf_dataset(train_dir)
test_dataset, test_mapping = create_hf_dataset(test_dir)
eval_dataset, eval_mapping = create_hf_dataset(eval_dir)

# Ensure that all mappings are the same and save only once
assert train_mapping == test_mapping == eval_mapping, "Class index mappings are not consistent across splits!"
class_index_mapping = train_mapping

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'eval': eval_dataset
})

# Save the DatasetDict to disk
dataset_dict.save_to_disk(data_save_dir)

# Save the class index mapping to a JSON file
with open(class_mapping_file, 'w') as f:
    json.dump(class_index_mapping, f)

# To load the datasets from disk later
loaded_dataset_dict = DatasetDict.load_from_disk(data_save_dir)
print(f"Loaded train dataset: {loaded_dataset_dict['train'][0]}")
print(f"Loaded test dataset: {loaded_dataset_dict['test'][0]}")
print(f"Loaded eval dataset: {loaded_dataset_dict['eval'][0]}")

# Load and print the class to index mapping
with open(class_mapping_file, 'r') as f:
    loaded_class_index_mapping = json.load(f)
print("Class to Index mapping:", loaded_class_index_mapping)

# Print the entire dataset structure
print(dataset_dict)



