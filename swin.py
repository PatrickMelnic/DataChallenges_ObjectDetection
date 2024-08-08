import os
import pandas as pd
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from transformers import SwinForImageClassification, SwinConfig
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

# Set the GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Customizable paths and parameters
DATASET_PATH = 'datasets/multilabel'
CLASS_MAPPING_FILE = 'class_index_mapping.json'
MODEL_SAVE_PATH = 'models/swin_multilabel.pth'
MODEL_SAVE_DIR='models/swin/'
NUM_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 5e-5



# Load the dataset
dataset_dict = load_from_disk(DATASET_PATH)
train_dataset = dataset_dict["train"]
val_dataset = dataset_dict["eval"]
test_dataset = dataset_dict["test"]
print(dataset_dict)

# Load the class index mapping
with open(CLASS_MAPPING_FILE, 'r') as f:
    class_index_mapping = json.load(f)

num_labels = len(class_index_mapping)
print(f"Number of labels: {num_labels}")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])
"""
# Custom Dataset Class with transformations
class MultiLabelDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img_path = sample['image_path']
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor(sample['encoded_label'], dtype=torch.float)
        return image, labels
"""

class MultiLabelDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']  # Directly using the loaded PIL image object

        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor(sample['encoded_label'], dtype=torch.float)
        return image, labels


# Create datasets and dataloaders
train_dataset = MultiLabelDataset(train_dataset, transform=transform)
val_dataset = MultiLabelDataset(val_dataset, transform=transform)
test_dataset = MultiLabelDataset(test_dataset, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load and modify the model
config = SwinConfig.from_pretrained('microsoft/swin-base-patch4-window7-224', num_labels=num_labels)
model = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224', config=config, ignore_mismatched_sizes=True)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.config.hidden_size, num_labels),
    torch.nn.Sigmoid()
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define training and validation functions
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(data_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    acc = accuracy_score(all_labels, all_preds > 0.5)
    f1 = f1_score(all_labels, all_preds > 0.5, average='samples')
    return total_loss / len(data_loader), acc, f1

# Training loop
for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val F1: {val_f1}")

# Save model using PyTorch
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# Save model using Hugging Face
model.save_pretrained(MODEL_SAVE_DIR)

# Evaluation on test set
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss}, Test Acc: {test_acc}, Test F1: {test_f1}")


