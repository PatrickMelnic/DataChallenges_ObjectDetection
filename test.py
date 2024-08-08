import os
from datasets import load_from_disk, DatasetDict

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

data_dir = 'datasets/multilabel/'

# Load the dataset from disk
loaded_dataset = load_from_disk(data_dir)
print(f"Loaded dataset: {loaded_dataset}")

# Print the first 5 entries of each split with all features
for split in ['train', 'test', 'eval']:
    if split in loaded_dataset:
        print(f"First 5 entries in {split} split:")
        for i in range(5):
            print(loaded_dataset[split][i])
    else:
        print(f"No {split} split found in the dataset.")

# Collect all image paths from each split
train_paths = set(loaded_dataset['train']['image_path']) if 'train' in loaded_dataset else set()
test_paths = set(loaded_dataset['test']['image_path']) if 'test' in loaded_dataset else set()
eval_paths = set(loaded_dataset['eval']['image_path']) if 'eval' in loaded_dataset else set()

# Check for overlaps
train_test_overlap = train_paths.intersection(test_paths)
train_eval_overlap = train_paths.intersection(eval_paths)
test_eval_overlap = test_paths.intersection(eval_paths)

# Print results
if train_test_overlap:
    print("Overlap between train and test:", train_test_overlap)
else:
    print("No overlap between train and test.")

if train_eval_overlap:
    print("Overlap between train and eval:", train_eval_overlap)
else:
    print("No overlap between train and eval.")

if test_eval_overlap:
    print("Overlap between test and eval:", test_eval_overlap)
else:
    print("No overlap between test and eval.")
