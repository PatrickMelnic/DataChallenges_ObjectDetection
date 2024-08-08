import os
import random
from collections import defaultdict, Counter
from datasets import Dataset, DatasetDict, Image, load_from_disk
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image as PILImage
from datetime import datetime
import json
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_dir = 'datasets/CN_dataset_obj_detection_04_23/dataset_obj_detection/'
data_save_dir = 'datasets/multilabel2/'
transformed_data_save_dir = 'datasets/transformed_multilabel/'
limiter = 20  # Minimum number of images per class
BATCH_SIZE = 1000  # Incremental save batch size

def print_with_time(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def get_filtered_image_files(data_dir):
    label_paths = defaultdict(list)
    
    # Collect image paths and labels
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                label_paths[label].append(img_path)

    # Filter labels with at least 'limiter' images
    filtered_label_paths = {label: paths for label, paths in label_paths.items() if len(paths) >= limiter}
    return filtered_label_paths

def process_images(filtered_label_paths, data_dir):
    temp_dict = defaultdict(lambda: {
        'absolute_path': None,
        'image_path': None,
        'labels': set(),
        'image': None,
        'encoded_labels': None
    })

    # Ensure data_dir ends with a separator
    if not data_dir.endswith(os.path.sep):
        data_dir += os.path.sep

    # Fill the temp_dict
    for label, paths in filtered_label_paths.items():
        for img_path in paths:
            image_key = os.path.basename(img_path).rsplit('.', 1)[0]
            if temp_dict[image_key]['absolute_path'] is None:
                # Remove everything before data_dir to get the relative path
                start_index = img_path.find(data_dir)
                relative_path = img_path[start_index:]
                temp_dict[image_key]['absolute_path'] = relative_path
                temp_dict[image_key]['image_path'] = image_key
                with PILImage.open(img_path) as img:
                    temp_dict[image_key]['image'] = img.copy()
            temp_dict[image_key]['labels'].add(label)
    
    return temp_dict

def encode_labels(temp_dict, class_index_mapping):
    mlb = MultiLabelBinarizer(classes=list(class_index_mapping.keys()))
    labels_list = [list(item['labels']) for item in temp_dict.values()]
    encoded_labels = mlb.fit_transform(labels_list)

    for idx, key in enumerate(temp_dict.keys()):
        temp_dict[key]['encoded_labels'] = encoded_labels[idx].tolist()
    
    return temp_dict

def create_dataset_dict(temp_dict):
    data = list(temp_dict.values())
    
    dataset_dict = DatasetDict()

    # Splitting the data
    random.shuffle(data)
    total = len(data)
    train_end = int(0.8 * total)
    test_end = train_end + int(0.1 * total)
    
    train_data = data[:train_end]
    test_data = data[train_end:test_end]
    eval_data = data[test_end:]
    
    splits = {
        'train': train_data,
        'test': test_data,
        'eval': eval_data
    }

    for split, split_data in splits.items():
        dataset = Dataset.from_dict({
            'image_path': [item['absolute_path'] for item in split_data],
            'absolute_path': [item['absolute_path'] for item in split_data],
            'label': [list(item['labels']) for item in split_data],
            'encoded_label': [item['encoded_labels'] for item in split_data],
            'image': [item['image'] for item in split_data]
        })

        dataset = dataset.cast_column('image', Image())
        dataset_dict[split] = dataset

    return dataset_dict

def print_random_samples(dataset_dict, num_samples=5):
    for split in dataset_dict:
        print_with_time(f"Random samples from {split} dataset:")
        random_indices = random.sample(range(len(dataset_dict[split])), num_samples)
        for idx in random_indices:
            sample = dataset_dict[split][idx]
            print(f"Image path: {sample['image_path']}")
            print(f"Absolute Path: {sample['absolute_path']}")
            print(f"Labels: {sample['label']}")
            print(f"Encoded Labels: {sample['encoded_label']}")
            # Displaying the image is not feasible in this script, so we skip that part
            print("---")

def transform_dataset_to_hf_format(dataset_dict):
    print_with_time("Transforming dataset into the expected format.")
    
    all_image_files = []
    all_absolute_paths = []
    all_labels = []
    all_encoded_labels = []
    all_images = []
    print_with_time ("extracting data")
    for split in dataset_dict:
        dataset = dataset_dict[split]
        for i in range(len(dataset)):
            all_image_files.append(dataset['image_path'][i])
            all_absolute_paths.append(dataset['absolute_path'][i])
            all_labels.append(dataset['label'][i])
            all_encoded_labels.append(dataset['encoded_label'][i])
            all_images.append(dataset['image'][i])
    
    # Create HuggingFace Dataset object
    print_with_time("new dataset")
    new_dataset = Dataset.from_dict({
        'image_path': all_image_files,
        'absolute_path': all_absolute_paths,
        'label': all_labels,
        'encoded_label': all_encoded_labels,
        'image': all_images
    })

    # Cast the image column to the Image type
    print_with_time("casting images")
    new_dataset = new_dataset.cast_column('image', Image())

    return new_dataset

def save_incremental(dataset, save_dir, batch_size):
    print_with_time("Saving dataset incrementally.")
    os.makedirs(save_dir, exist_ok=True)
    total = len(dataset)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = dataset.select(range(start_idx, end_idx))
        batch.save_to_disk(os.path.join(save_dir, f'batch_{start_idx // batch_size}'))
        print_with_time(f"Saved batch {start_idx // batch_size} ({start_idx}-{end_idx})")

def visualize_data_distribution(dataset_dict, output_dir):
    print_with_time("Visualizing data distribution.")

    # Distribution of images per class
    label_counter = Counter()
    for split in dataset_dict:
        for labels in dataset_dict[split]['label']:
            label_counter.update(labels)
    
    # Sorting labels alphabetically for the plot
    sorted_labels = sorted(label_counter.items(), key=lambda x: x[0])
    labels, values = zip(*sorted_labels)  # Unpacking the labels and values
    
    plt.figure(figsize=(12, 8))
    plt.bar(labels, values)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Class')
    plt.xticks(rotation=90, fontsize=8)  # Smaller font size for labels
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'images_per_class.png'))
    plt.close()

    # Distribution of the number of labels per image
    labels_per_image = []
    for split in dataset_dict:
        for encoded_labels in dataset_dict[split]['encoded_label']:
            labels_per_image.append(sum(encoded_labels))
    
    plt.figure(figsize=(8, 6))
    plt.hist(labels_per_image, bins=range(1, max(labels_per_image) + 2))
    plt.xlabel('Number of Labels')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Number of Labels per Image')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'labels_per_image.png'))
    plt.close()



def main():
    print_with_time("Starting the data processing script.")

    # Step 1: Filter classes with >= 20 images
    print_with_time("Filtering classes with at least 20 images.")
    filtered_label_paths = get_filtered_image_files(data_dir)

    # Step 2: Process images and fill the temp_dict
    print_with_time("Processing images and filling the temporary dictionary.")
    temp_dict = process_images(filtered_label_paths)

    # Step 3: Load or create class index mapping
    print_with_time("Loading or creating class index mapping.")
    class_mapping_file = 'class_index_mapping.json'
    if os.path.exists(class_mapping_file):
        with open(class_mapping_file, 'r') as f:
            class_index_mapping = json.load(f)
    else:
        all_labels = set()
        for item in temp_dict.values():
            all_labels.update(item['labels'])
        class_index_mapping = {label: idx for idx, label in enumerate(sorted(all_labels))}
        with open(class_mapping_file, 'w') as f:
            json.dump(class_index_mapping, f)

    # Step 4: Encode labels in binary format
    print_with_time("Encoding labels in binary format.")
    temp_dict = encode_labels(temp_dict, class_index_mapping)

    # Step 5: Create DatasetDict and save to disk
    print_with_time("Creating DatasetDict and saving to disk.")
    dataset_dict = create_dataset_dict(temp_dict)
    dataset_dict.save_to_disk(data_save_dir)

    # Step 6: Load the dataset from disk and print the number of images in each split
    print_with_time("Loading the dataset from disk.")
    loaded_dataset_dict = load_from_disk(data_save_dir)
    for split in loaded_dataset_dict:
        print_with_time(f"Loaded {split} dataset: {len(loaded_dataset_dict[split])} images")

    # Print random samples from each split
    print_random_samples(loaded_dataset_dict, num_samples=5)

    # Visualize data distribution
    visualize_data_distribution(loaded_dataset_dict, data_save_dir)

    print_with_time("Data processing complete.")



if __name__ == "__main__":
    main()
