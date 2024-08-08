import os
import pandas as pd
from PIL import Image
from datasets import DatasetDict, Dataset
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Define dataset and CSV file paths
dataset_dir = os.path.abspath("datasets")
train_csv = os.path.join(dataset_dir, "CN_coin_descriptions", "CN_hpz_dataset_train.csv")
eval_csv = os.path.join(dataset_dir, "CN_coin_descriptions", "CN_hpz_dataset_eval.csv")

# Define where to save the processed dataset
save_path = os.path.join(dataset_dir, "obj_det", "dl")

# Ensure the save directory exists
os.makedirs(save_path, exist_ok=True)

print("**********loading dataset**********")

# Load CSV files with proper handling of quotes
train_df = pd.read_csv(train_csv, quotechar='"')
eval_df = pd.read_csv(eval_csv, quotechar='"')

# Strip whitespace from column names
train_df.columns = train_df.columns.str.strip()
eval_df.columns = eval_df.columns.str.strip()
print(train_df.columns)
print(eval_df.columns)

# Function to correct and normalize image paths
def correct_path(path):
    path = path.replace('\\', '/')  # Replace backslashes with forward slashes
    path_parts = path.split('/')
    # Convert the last part (before the filename) to lowercase
    path_parts[-2] = path_parts[-2].lower()
    # Join the path parts back together
    normalized_path = '/'.join(path_parts)
    return os.path.join(normalized_path)

# Apply path correction to image_path column
train_df['image_path'] = train_df['image_path'].apply(correct_path)
eval_df['image_path'] = eval_df['image_path'].apply(correct_path)

# Add filename column with relative path and remove the common prefix
common_prefix = os.path.join('CN_dataset_obj_detection_04_23', 'dataset_obj_detection')
train_df['filename'] = train_df['image_path'].apply(lambda x: os.path.relpath(x, dataset_dir).replace(common_prefix, ''))
eval_df['filename'] = eval_df['image_path'].apply(lambda x: os.path.relpath(x, dataset_dir).replace(common_prefix, ''))

# Ensure "image" and "label" columns in both datasets
train_df = train_df.rename(columns={train_df.columns[0]: "image_path", train_df.columns[1]: "label"})
eval_df = eval_df.rename(columns={eval_df.columns[0]: "image_path", eval_df.columns[1]: "label"})

# Print first 5 entries of the raw dataframes after renaming
print("First 5 entries of the raw train dataframe:")
print(train_df.head())

print("First 5 entries of the raw eval dataframe:")
print(eval_df.head())

# Load images and create dataset
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# Create a Huggingface dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Print first 5 entries of the datasets
print("First 5 entries of the train dataset:")
print(train_dataset[:5])

print("First 5 entries of the eval dataset:")
print(eval_dataset[:5])

# Create a DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "eval": eval_dataset
})

# Add image column by loading images and keep the filename
def add_image_and_filename(example):
    image = load_image(example['image_path'])
    filename = example['filename']
    return {"image": image, "filename": filename}

dataset_dict = dataset_dict.map(
    add_image_and_filename,
    remove_columns=["image_path"]
)

# Print first 5 entries after adding images and filenames
print("First 5 entries of the train dataset after adding images and filenames:")
print(dataset_dict["train"][:5])

print("First 5 entries of the eval dataset after adding images and filenames:")
print(dataset_dict["eval"][:5])

# Save dataset_dict for later use
dataset_dict.save_to_disk(save_path)

from datasets import load_from_disk
from PIL import Image

# Function to check if an object is a PIL image
def is_pil_image(obj):
    return isinstance(obj, Image.Image)

# Load the dataset from disk
dataset_dict = load_from_disk(save_path)

# Print the dataset_dict to see its structure
print(dataset_dict)

def print_labels_and_ids(dataset, split_name):
    print(f"\nLabels and IDs for the {split_name} split:")
    for i, example in enumerate(dataset):
        print(f"ID: {i}, Label: {example['label']}, Filename: {example['filename']}")
        if i >= 4:  # Print only the first 5 entries
            break

# Print labels and IDs for the train split
print_labels_and_ids(dataset_dict["train"], "train")

# Print labels and IDs for the eval split
print_labels_and_ids(dataset_dict["eval"], "eval")

# Print the number of samples in each split
print(f"\nNumber of samples in the train split: {len(dataset_dict['train'])}")
print(f"Number of samples in the eval split: {len(dataset_dict['eval'])}")

# Additional check to verify image type
def verify_image_type(dataset, split_name):
    print(f"\nVerifying image types for the {split_name} split:")
    for i, example in enumerate(dataset):
        image = example['image']
        label = example['label']
        filename = example['filename']
        print(f"ID: {i}, Image type: {'PIL Image' if is_pil_image(image) else type(image)}, Label: {label}, Filename: {filename}")
        if i >= 4:  # Check only the first 5 entries
            break

# Verify image types for the train split
verify_image_type(dataset_dict["train"], "train")

# Verify image types for the eval split
verify_image_type(dataset_dict["eval"], "eval")



"""
import os
import pandas as pd
from PIL import Image
from datasets import DatasetDict, Dataset
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Define dataset and CSV file paths
dataset_dir = os.path.abspath("datasets")
train_csv = os.path.join(dataset_dir, "CN_coin_descriptions", "CN_hpz_dataset_train.csv")
eval_csv = os.path.join(dataset_dir, "CN_coin_descriptions", "CN_hpz_dataset_eval.csv")

# Define where to save the processed dataset
save_path = os.path.join(dataset_dir, "obj_det")

# Ensure the save directory exists
os.makedirs(save_path, exist_ok=True)

print("**********loading dataset**********")

# Load CSV files with proper handling of quotes
train_df = pd.read_csv(train_csv, quotechar='"')
eval_df = pd.read_csv(eval_csv, quotechar='"')

# Strip whitespace from column names
train_df.columns = train_df.columns.str.strip()
eval_df.columns = eval_df.columns.str.strip()

# Function to correct and normalize image paths
def correct_path(path):
    path = path.replace('\\', '/')  # Replace backslashes with forward slashes
    path_parts = path.split('/')
    # Convert the last part (before the filename) to lowercase
    path_parts[-2] = path_parts[-2].lower()
    # Join the path parts back together
    normalized_path = '/'.join(path_parts)
    return os.path.join(dataset_dir, normalized_path)

# Apply path correction to image_path column
train_df['image_path'] = train_df['image_path'].apply(correct_path)
eval_df['image_path'] = eval_df['image_path'].apply(correct_path)

# Ensure "image" and "label" columns in both datasets
train_df = train_df.rename(columns={train_df.columns[0]: "image_path", train_df.columns[1]: "label"})
eval_df = eval_df.rename(columns={eval_df.columns[0]: "image_path", eval_df.columns[1]: "label"})

# Print first 5 entries of the raw dataframes after renaming
print("First 5 entries of the raw train dataframe:")
print(train_df.head())

print("First 5 entries of the raw eval dataframe:")
print(eval_df.head())

# Load images and create dataset
def load_image(image_path):
    print("image  : ", image_path)
    img=Image.open(image_path).convert("RGB")
    print(img)
    return img

# Create a Huggingface dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Print first 5 entries of the datasets
print("First 5 entries of the train dataset:")
print(train_dataset[:5])

print("First 5 entries of the eval dataset:")
print(eval_dataset[:5])

# Create a DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "eval": eval_dataset
})

# Add image column by loading images
dataset_dict = dataset_dict.map(
    lambda example: {"image": load_image(example['image_path'])},
    remove_columns=["image_path"]
)

# Print first 5 entries after adding images
print("First 5 entries of the train dataset after adding images:")
print(dataset_dict["train"][:1])

print("First 5 entries of the eval dataset after adding images:")
print(dataset_dict["eval"][:1])

# Save dataset_dict for later use
dataset_dict.save_to_disk(save_path)

from datasets import load_from_disk


#Load the dataset from disk
dataset_dict = load_from_disk(save_path)

print(dataset_dict)

from datasets import load_from_disk
from PIL import Image

# Function to check if an object is a PIL image
def is_pil_image(obj):
    return isinstance(obj, Image.Image)

# Print first 5 entries after adding images
print("First 5 entries of the train dataset after adding images:")
print(dataset_dict["train"][:5])

print("First 5 entries of the eval dataset after adding images:")
print(dataset_dict["eval"][:5])



def print_labels_and_ids(dataset, split_name):
    print(f"\nLabels and IDs for the {split_name} split:")
    for i, example in enumerate(dataset):
        print(f"ID: {i}, Label: {example['label']}")
        if i >= 4:  # Print only the first 5 entries
            break

# Print labels and IDs for the train split
print_labels_and_ids(dataset_dict["train"], "train")

# Print labels and IDs for the eval split
print_labels_and_ids(dataset_dict["eval"], "eval")

# Print the number of samples in each split
print(f"\nNumber of samples in the train split: {len(dataset_dict['train'])}")
print(f"Number of samples in the eval split: {len(dataset_dict['eval'])}")

# Additional check to verify image type
def verify_image_type(dataset, split_name):
    print(f"\nVerifying image types for the {split_name} split:")
    for i, example in enumerate(dataset):
        image = example['image']
        label = example['label']
        print(f"ID: {i}, Image type: {'PIL Image' if is_pil_image(image) else type(image)}, Label: {label}")
        if i >= 4:  # Check only the first 5 entries
            break

# Verify image types for the train split
verify_image_type(dataset_dict["train"], "train")

# Verify image types for the eval split
verify_image_type(dataset_dict["eval"], "eval")

"""

