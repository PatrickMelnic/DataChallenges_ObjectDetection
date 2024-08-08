import os
import json
from datasets import load_from_disk
from transformers import ViTHybridForImageClassification, ViTHybridImageProcessor
from PIL import Image
import torch.nn.functional as F
import torch
from datetime import datetime
from sklearn.metrics import f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

# Set the GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Customizable Paths
dataset_path = "datasets/multilabel/"
model_path1 = "models/vithybrid/"
model_path = "google/vit-hybrid-base-bit-384"

# Load the test split of the dataset
dataset = load_from_disk(dataset_path)
test_dataset = dataset["test"]
print(f"Loaded dataset with {len(test_dataset)} samples.")

# Load the model and feature extractor
model = ViTHybridForImageClassification.from_pretrained(model_path1)
image_processor = ViTHybridImageProcessor.from_pretrained(model_path)

# Load and prepare label mapping
with open('class_index_mapping.json', 'r') as file:
    label_mapping = json.load(file)
id_to_label = {v: k for k, v in label_mapping.items()}

# Prepare MultiLabelBinarizer with all labels
mlb = MultiLabelBinarizer(classes=list(id_to_label.values()))
mlb.fit([list(id_to_label.values())])  # Fitting with all possible labels

# Initialize results dictionary
results = {
    "path": [],
    "image": [],
    "true labels": [],
    "predicted labels": [],
    "accuracy": []
}

# Set the probability threshold
probability_threshold = 0.01

for i, entry in enumerate(test_dataset):
    current_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
    print(f"Processing image {i+1}/{len(test_dataset)} at {current_time}")
    
    image_path = entry["absolute_path"]
    image = Image.open(image_path)
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)

    # Sort predictions by probability and get the top-n labels where n is the length of the true labels
    top_predictions = sorted(
        [(prob.item(), index) for index, prob in enumerate(probabilities[0])],
        key=lambda x: x[0], reverse=True
    )[:len(entry['label'])]  # Adjust to match the number of true labels

    # Convert top prediction indices to labels
    predicted_labels = [id_to_label[index] for _, index in top_predictions]

    y_pred = mlb.transform([predicted_labels])  # Transform to binary format

    # Ensure y_true is in the same format as y_pred
    y_true = mlb.transform([entry['label']])

    # Calculate metrics
    try:
        individual_f1_score = f1_score(y_true, y_pred, average='micro')
        hamming_score = 1 - hamming_loss(y_true, y_pred)
    except Exception as e:
        print(f"Error calculating metrics for image {i+1}: {e}")
        continue

    # Append to results
    results["path"].append(image_path)
    results["image"].append(entry["image_path"])
    results["true labels"].append(entry["label"])
    results["predicted labels"].append(predicted_labels)
    results["accuracy"].append({"F1 Score": individual_f1_score, "Hamming Score": hamming_score})

# Save results to a JSON file
with open('results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)
    print("Results saved to results.json.")

# Optionally print some of the results
for key in results:
    print(f"{key}: {results[key][:5]}")  # Adjust according to your preference to view more/less data
