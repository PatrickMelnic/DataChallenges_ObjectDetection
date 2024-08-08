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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
mlb.fit([list(id_to_label.values())])

# Initialize results dictionary
results = {
    "path": [],
    "image": [],
    "true labels": [],
    "predicted labels": [],
    "accuracy": []
}

# Initialize co-occurrence and agreement matrices
num_labels = len(mlb.classes_)
co_occurrence_matrix = np.zeros((num_labels, num_labels), dtype=int)
agreement_matrix = np.zeros((num_labels, num_labels), dtype=int)

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

    predicted_indices = [index for index, prob in enumerate(probabilities[0]) if prob.item() >= probability_threshold]
    predicted_labels = [id_to_label[index] for index in predicted_indices]

    y_pred = mlb.transform([predicted_labels])
    y_true = mlb.transform([entry['label']])

    # Update matrices
    for true_index in range(num_labels):
        for pred_index in range(num_labels):
            if y_true[0][true_index] and y_true[0][pred_index]:
                co_occurrence_matrix[true_index][pred_index] += 1
            if y_pred[0][pred_index] and y_true[0][true_index]:
                agreement_matrix[true_index][pred_index] += 1

    # Append to results
    results["path"].append(image_path)
    results["image"].append(entry["image_path"])
    results["true labels"].append(entry["label"])
    results["predicted labels"].append(predicted_labels)
    results["accuracy"].append({"F1 Score": f1_score(y_true, y_pred, average='micro'), "Hamming Score": 1 - hamming_loss(y_true, y_pred)})

# Generate heatmaps
plt.figure(figsize=(10, 8))
sns.heatmap(co_occurrence_matrix, annot=False, cmap="Blues")
plt.title('Label Co-occurrence Matrix')
plt.xlabel('Labels')
plt.ylabel('Labels')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(agreement_matrix, annot=False, cmap="Greens")
plt.title('Label Prediction Agreement Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
# Save the figure
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')  # Saves the heatmap as a PNG file
plt.close()  # Close the plot to free up memory

print("Heatmap saved as 'heatmap.png'")
# Save results to a JSON file
with open('results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)
    print("Results saved to results.json.")

# Optionally print some of the results
for key in results:
    print(f"{key}: {results[key][:5]}")  # Adjust according to your preference to view more/less data
