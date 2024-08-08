import torch
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer, CLIPModel, CLIPProcessor
from datasets import load_from_disk
import os
#import wandb
import shutil
import os
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
import csv
from evaluate import load

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Define paths
#dataset_path = os.path.abspath("datasets/") #images
saved_dataset_path = "datasets/obj_det/hpz/"     #cached dataset
model_name = "openai/clip-vit-base-patch32"
output_dir = "tmp/"
final_model_save_path = "clip_pretraining/checkpoint"
#final_model_save_path = "Modelle/"

# Training parameters
num_train_epochs = 20
learning_rate = 5e-05
batch_size = 8
logging_steps = 50
save_steps = 50
save_total_limit = 2

# Create directories if they don't exist
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_model_save_path, exist_ok=True)

# Initialize Weights & Biases
"""
wandb.init(
    project="Test",
    config={
        "learning_rate": learning_rate,
        "architecture": "CLIP",
        "dataset": "Corpus Nummorum",
        "epochs": num_train_epochs,
    }
)
"""

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained(model_name)
clip_model.to('cuda')
processor = CLIPProcessor.from_pretrained(model_name)

def clip_loss(image, text_features):
    image_features = clip_model.get_image_features(image)
    input_normed = F.normalize(image_features.unsqueeze(1), dim=2)
    embed_normed = F.normalize(text_features.unsqueeze(0), dim=2)
    dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
    return dists.mean()

# Load the dataset from disk
dataset_dict = load_from_disk(saved_dataset_path)

# Function to preprocess the data
def preprocess_function(examples):
    prompts = ["A coin with " + ", ".join(label[2:-2].split("', '").split('", "')) for label in examples["label"]]
    inputs = processor(text=prompts, images=examples["image"], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    #inputs = processor(text=examples["caption"], images=examples["image"], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    return inputs

# Apply the preprocessing function to the datasets
dataset_dict = dataset_dict.map(preprocess_function, batched=True)

# Custom collate function
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    pixel_values = torch.stack([torch.tensor(item['pixel_values']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    return {
        'input_ids': input_ids,
        'pixel_values': pixel_values,
        'attention_mask': attention_mask,
    }

# Define a custom Trainer class to override the compute_loss method
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        image = inputs['pixel_values'].to('cuda')
        text = inputs['input_ids'].to('cuda')
        text_features = model.get_text_features(text)
        loss = clip_loss(image, text_features)
        return (loss, outputs) if return_outputs else loss

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,  # Output directory
    per_device_train_batch_size=batch_size,  # Batch size per device during training
    num_train_epochs=num_train_epochs,  # Total number of training epochs
    save_steps=save_steps,  # Number of update steps before saving checkpoint
    logging_steps=logging_steps,  # Number of update steps before logging
    save_total_limit=save_total_limit,  # Limit the total amount of checkpoints on disk
    remove_unused_columns=False,  # Remove unused columns from the dataset
    push_to_hub=False,  # Do not push the model to the hub
    #report_to="wandb",  # Report metrics to wandb if enabled
    learning_rate=learning_rate,
)

# Initialize the Trainer
trainer = CustomTrainer(
    model=clip_model,  # The instantiated 🤗 Transformers model to be trained
    args=training_args,  # Training arguments, defined above
    data_collator=collate_fn,  # The data collator that will be used for batching
    train_dataset=dataset_dict["train"],  # Training dataset
)

# Start training
trainer.train()

# Save the final model
trainer.save_model(final_model_save_path)
