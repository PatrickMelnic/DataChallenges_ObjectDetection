import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPImageProcessor, CLIPModel
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image

##################################################################
### Adjustable parameters                                      ###

dataset_path = 'datasets/obj_det/'  # cached hf dataset, eval split
model_name = "openai/clip-vit-base-patch32"  # for the processor
model_path = 'clip_pretraining/checkpoint'   # finetuned model
output_plot_path = 'tsne_visualization.png'  # Path to save the t-SNE plot

##################################################################

# Load dataset
print("*****Load Dataset*****")
dataset = load_from_disk(dataset_path)
print(dataset)

# Load model and processor
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_name)
img_processor = CLIPImageProcessor.from_pretrained(model_name)

# Prepare the evaluation set
results = {
    #"images": [],
    "true_labels": [],
    "similarity": [],
    "rank": [],
    "precision_at_1": [],
    "precision_at_5": [],
    "precision_at_10": [],
    "mrr": []
}

# Helper function to calculate cosine similarity between two embeddings
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)[0][0]

# Helper function to calculate precision at K
def precision_at_k(ranks, k):
    return np.mean([1 if r < k else 0 for r in ranks])

# Make predictions
print("*****Make predictions*****")
image_features_list = []
text_features_list = []
true_labels = []
image_paths = []

# Collect all image and text features
for example in dataset['eval']:  # Ensure we are iterating over the eval split
    img = example['image']
    labels = [", ".join(label[2:-2].split("', '")) for label in example["label"]]  # Assuming 'label' contains the caption
    image_path = example['image_path']
    # Load and process the image
    image = img_processor(images=img, return_tensors="pt")['pixel_values']
    #image = processor(img=img, return_tensors="pt")['pixel_values']
    with torch.no_grad():
        image_features = model.get_image_features(image)
    image_features_list.append(image_features.numpy())
    
    text_inputs = processor(text=labels, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features_list.append(text_features.numpy())

    for label in labels:
        if not label in true_labels:
            true_labels.append(label)
    image_paths.append(image_path)
    #print(file_name)
    #results['images'].append(image_path)

# Calculate similarity and retrieval metrics
for i, (image_features, true_label, image_path) in enumerate(zip(image_features_list, true_labels, image_paths)):
    """
    Cosine Similarity:

    Measures the cosine of the angle between two vectors in the embedding space.
    Higher similarity indicates that the vectors (image and text embeddings) are close to each other, implying a better match.
    """
    similarities = [calculate_similarity(image_features, text_features) for text_features in text_features_list]
    """
    Precision at K (P@K):

    Measures the proportion of correct items in the top-K retrieved results.
    precision_at_1: Checks if the true caption is the top-most similar caption.
    precision_at_5: Checks if the true caption is among the top-5 similar captions.
    precision_at_10: Checks if the true caption is among the top-10 similar captions.
    Higher precision indicates better retrieval performance.
    """
    sorted_indices = np.argsort(similarities)[::-1]  # Indices of captions sorted by similarity in descending order
    """
    The position of the true caption when all captions are sorted by their similarity to the image embedding.
    A lower rank indicates a better match.
    """
    rank = np.where(sorted_indices == i)[0][0] + 1  # Rank of the true caption

    # Save results
    #results['image'].append(image_path)
    results['true_labels'].append(true_label)
    results['similarity'].append(similarities[i])
    results['rank'].append(rank)
    results['precision_at_1'].append(precision_at_k([rank], 1))
    results['precision_at_5'].append(precision_at_k([rank], 5))
    results['precision_at_10'].append(precision_at_k([rank], 10))
    results['mrr'].append(1 / rank)

    """
    Mean Reciprocal Rank (MRR):

    Measures the average of the reciprocal ranks of results.
    MRR is high when the true captions are ranked highly across all queries.
    Calculated as 1 / rank for each query.
    Higher MRR indicates better overall ranking performance.
    """

# Save results to an Excel file
print("*****Save to file*****")
for k in results.keys():
    print(f"{k}: {len(results[k])}")
df = pd.DataFrame(results)
df.to_excel('evaluation_results.xlsx', index=False)

print("Evaluation complete. Results saved to evaluation_results.xlsx")

print("*****Visualise*****")
### Visualization Code Starts Here ###
# Extract image and text features for visualization
image_features_list = []
text_features_list = []
labels = []
captions = []

for example in dataset['eval']:  # Ensure we are iterating over the eval split
    image = example['image']  # Use the filename instead of the image object
    label = example['label']  # Assuming 'label' contains the caption
    file_name = example['image_path']
    

    # Debugging prints
    #print(f"Image path (PIL Image): {image_path}")
    #print(f"Caption: {caption}")
    #print(f"Filename: {file_name}")

    # Extract the relative file path for labeling
    #path_label = file_name  # Directly use the relative filename for the label

    # Load and process the image
    image_tensor = processor(images=image, return_tensors="pt")['pixel_values']  # Directly use the PIL Image object
    with torch.no_grad():
        image_features = model.get_image_features(image_tensor).numpy()
    image_features_list.append(image_features[0])  # Get the numpy array directly

    text_inputs = processor(text=label, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs).numpy()
    text_features_list.append(text_features[0])  # Get the numpy array directly

    labels.append(label)

# Convert lists to numpy arrays
image_features_np = np.array(image_features_list)
text_features_np = np.array(text_features_list)

# Combine the image and text features for t-SNE
combined_features = np.concatenate([image_features_np, text_features_np])

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
tsne_results = tsne.fit_transform(combined_features)

# Split the t-SNE results back into image and text
tsne_image = tsne_results[:len(image_features_list)]
tsne_text = tsne_results[len(image_features_list):]

# Plotting
plt.figure(figsize=(24, 18))

for i in range(len(tsne_image)):
    plt.scatter(tsne_image[i, 0], tsne_image[i, 1], color='blue', label='Image' if i == 0 else "", alpha=0.5)
    plt.text(tsne_image[i, 0], tsne_image[i, 1], labels[i], fontsize=9)

for i in range(len(tsne_text)):
    plt.scatter(tsne_text[i, 0], tsne_text[i, 1], color='red', label='Text' if i == 0 else "", alpha=0.5)
    plt.text(tsne_text[i, 0], tsne_text[i, 1], labels[i], fontsize=9)

plt.legend()
plt.title("t-SNE visualization of Image and Text embeddings")
plt.savefig(output_plot_path)  # Save the plot as an image file
plt.close()

print(f"t-SNE plot saved to {output_plot_path}")
