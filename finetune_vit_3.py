#####################################################################################################
#### these parameters need to be adjusted to run the script on your own ####

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


dataset_path = "datasets/multilabel"  # Path to the dataset
model_name = "google/vit-base-patch16-224"  # Specify which ViT model to use
class_mapping_file="class_index_mapping.json"

num_train_epochs = 10
learning_rate = 5e-05
batch_size = 16
logging_steps = 500
eval_steps = 500
save_steps = 500
save_total_limit = 2
output_dir = "models/vit_models3/"  # Temporary output directory for the best 2 models
final_model_save_path = "models/final/vit3/"  # Final model save path

#####################################################################################################

import sys
print(sys.executable)
from datasets import load_from_disk
import torch
from PIL import Image
from transformers import TrainingArguments, Trainer, ViTModel, ViTImageProcessor
from evaluate import load
import numpy as np
import json
import torch.nn as nn
from dataclasses import dataclass


device = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories if they do not exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_model_save_path, exist_ok=True)
print("~~~loading dataset~~~")

# Load the dataset
dataset_dict = load_from_disk(dataset_path)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["eval"]
test_dataset = dataset_dict["test"]
print(dataset_dict)

# Load the class index mapping
with open(class_mapping_file, 'r') as f:
    class_index_mapping = json.load(f)

num_labels = len(class_index_mapping)
print(f"Number of labels: {num_labels}")

# Custom output class
from transformers.utils import ModelOutput
from typing import Optional, Tuple

@dataclass
class CustomSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# Custom model class
class CustomViTForImageClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # Take the output of the [CLS] token

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.float())

        return CustomSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

model = CustomViTForImageClassification(model_name=model_name, num_labels=num_labels)

# Initialize the image processor
image_processor = ViTImageProcessor.from_pretrained(model_name)

print("~~~preprocessing data~~~")

def transform(examples):
    # Convert all images to RGB format and preprocess using our image processor
    inputs = image_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
    # Labels
    inputs["labels"] = examples["encoded_label"]
    return inputs

# Use the with_transform() method to apply the transform to the dataset on the fly during training
train_dataset = train_dataset.with_transform(transform)
eval_dataset = eval_dataset.with_transform(transform)
test_dataset = test_dataset.with_transform(transform)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.float),
    }
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Ensure logits and labels are tensors
    logits = torch.tensor(logits)
    labels = torch.tensor(labels).int()
    
    # Apply sigmoid to logits and then threshold at 0.5
    preds = (logits.sigmoid() > 0.5).int()
    
    # Convert predictions and labels to the correct format for sklearn
    preds = preds.numpy()
    labels = labels.numpy()

    # Debug prints
    debug_info = {
        "predictions_shape": preds.shape,
        "predictions_dtype": str(preds.dtype),
        "predictions_example": preds[:5].tolist(),
        "references_shape": labels.shape,
        "references_dtype": str(labels.dtype),
        "references_example": labels[:5].tolist()
    }

    # Write debug information to a JSON file
    with open("debug_info.json", "w") as f:
        json.dump(debug_info, f, indent=4)

    # Flatten predictions and labels for metric calculation
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    # Evaluate metrics using sklearn
    accuracy = accuracy_score(labels_flat, preds_flat)
    precision = precision_score(labels_flat, preds_flat, average='micro')
    recall = recall_score(labels_flat, preds_flat, average='micro')
    f1 = f1_score(labels_flat, preds_flat, average='micro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Example debug information check
with open("debug_info.json", "r") as f:
    debug_info = json.load(f)
print(json.dumps(debug_info, indent=4))

# Example debug information check
with open("debug_info.json", "r") as f:
    debug_info = json.load(f)
print(json.dumps(debug_info, indent=4))
# Example debug information check
with open("debug_info.json", "r") as f:
    debug_info = json.load(f)
print(json.dumps(debug_info, indent=4))


# Example debug information check
with open("debug_info.json", "r") as f:
    debug_info = json.load(f)
print(json.dumps(debug_info, indent=4))



print("~~~train model~~")

training_args = TrainingArguments(
    output_dir=output_dir,  # output directory
    per_device_train_batch_size=batch_size,  # batch size per device during training
    evaluation_strategy="steps",  # evaluation strategy to adopt during training
    num_train_epochs=num_train_epochs,  # total number of training epochs
    save_steps=save_steps,  # number of update steps before saving checkpoint
    eval_steps=eval_steps,  # number of update steps before evaluating
    logging_steps=logging_steps,  # number of update steps before logging
    save_total_limit=2,  # limit the total amount of checkpoints on disk
    remove_unused_columns=False,  # remove unused columns from the dataset
    push_to_hub=False,  # do not push the model to the hub
    load_best_model_at_end=True,  # load the best model at the end of training
    learning_rate=learning_rate,
    run_name='vit_g5_10'
)

from transformers import Trainer

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    data_collator=collate_fn,  # the data collator that will be used for batching
    compute_metrics=compute_metrics,  # the metrics function that will be used for evaluation
    train_dataset=train_dataset,  # training dataset
    eval_dataset=eval_dataset,  # evaluation dataset
)

# Start training
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print(test_results)
model.config.to_json_file("config.json")
# Save final model
trainer.save_model(final_model_save_path)
trainer.save_pretrained("vit/1/")
