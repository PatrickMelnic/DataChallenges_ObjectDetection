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
model_path1 = "models/vithybrid/"  # Path where the model is stored
model_path = "google/vit-hybrid-base-bit-384"  # Model identifier on HuggingFace

# Load the test split of the dataset
dataset = load_from_disk(dataset_path)
test_dataset = dataset["test"]
print(f"Loaded dataset with {len(test_dataset)} samples.")

# Load the model and feature extractor
model = ViTHybridForImageClassification.from_pretrained(model_path1)
image_processor = ViTHybridImageProcessor.from_pretrained(model_path)

# Load label mapping from file
with open('class_index_mapping.json', 'r') as file:
    label_mapping = json.load(file)

# Prepare MultiLabelBinarizer with all possible labels from the mapping
mlb = MultiLabelBinarizer()
mlb.fit([list(label_mapping.keys())])  # Fit with all labels from your class_index_mapping

# Initialize the results dictionary
results = {
    "path": [],
    "image": [],
    "true labels": [],
    "predicted labels": [],
    "accuracy": []
}

# Set the probability threshold
probability_threshold = 0.01  # Example threshold value

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

    # Get predicted labels above the threshold
    predicted_labels = [label_mapping[label] for label, prob in enumerate(probabilities[0]) if prob.item() >= probability_threshold]

    # Convert true and predicted labels to binary format for this image
    y_true = mlb.transform([entry['encoded_label']])  # Use 'encoded_label' if it's already in integer ID format
    y_pred = mlb.transform([predicted_labels])

    # Calculate F1 Score for this image
    individual_f1_score = f1_score(y_true, y_pred, average='micro')

    # Calculate Hamming Score for this image
    hamming_score = 1 - hamming_loss(y_true, y_pred)

    # Populate the results dictionary
    results["path"].append(image_path)
    results["image"].append(entry["image_path"])
    results["true labels"].append(entry["encoded_label"])
    results["predicted labels"].append(predicted_labels)
    results["accuracy"].append({"F1 Score": individual_f1_score, "Hamming Score": hamming_score})

# Save results to a JSON file
with open('results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)
    print("Results saved to results.json.")

# Optionally print some of the results
for key in results:
    print(f"{key}: {results[key][:5]}")  # Adjust according to your preference to view more/less data

    
# Print the results (optional)
for key in results:
    print(f"{key}: {results[key][:5]}")  # Remove or modify as needed



