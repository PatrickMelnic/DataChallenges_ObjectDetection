import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

import torch
import torch.nn as nn
from torchvision.ops import MLP
import torchvision.transforms as T

from pathlib import Path
from PIL import Image

import datasets
from transformers import CLIPVisionModel
from transformers.optimization import get_cosine_schedule_with_warmup

from timm import list_models, create_model

from accelerate import Accelerator, notebook_launcher
from tqdm import tqdm

import evaluate

from safetensors.torch import load_model, load_file

#dataset = datasets.load_dataset('fuliucansheng/pascal_voc','voc2007_main')
#class_names = [
#    "Aeroplane","Bicycle","Bird","Boat","Bottle",
#    "Bus","Car","Cat","Chair","Cow","Diningtable",
#    "Dog","Horse","Motorbike","Person",
#    "Potted plant","Sheep","Sofa","Train","Tv/monitor"
#]
#print(dataset['train']['classes'][:10])
#sys.exit()
dataset = datasets.load_from_disk(os.path.join("datasets", "obj_det", "dl"))
#print(dataset['train'][:10])

class_names = os.listdir(os.path.join("datasets", "CN_dataset_obj_detection_04_23", "dataset_obj_detection"))
class_names = ['hades', 'poseidon', 'zeus']
#print([[class_names.index(label) for label in labels] for labels in dataset['train']['label'][:10]])
#sys.exit()
class_number = len(class_names)

label2id = {c:idx for idx,c in enumerate(class_names)}
id2label = {idx:c for idx,c in enumerate(class_names)}

def show_samples(ds,rows,cols):
    samples = ds.shuffle().select(np.arange(rows*cols)) # selecting random images
    fig = plt.figure(figsize=(cols*4,rows*4))
    # plotting
    for i in range(rows*cols):
        img = samples[i]['image']
        labels = samples[i]['classes']
        # getting string labels and combining them with a comma
        labels = ','.join([id2label[lb] for lb in labels])
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(img)
        plt.title(labels)
        plt.axis('off')
            
#show_samples(dataset['train'],rows=5,cols=5)

img_size = (224,224)

train_tfms = T.Compose([
    T.Resize(img_size),
    T.ToTensor(),
    T.Normalize(
        mean = (0.5,0.5,0.5),
         std = (0.5,0.5,0.5)
    )
])

valid_tfms = T.Compose([
    T.Resize(img_size),
    T.ToTensor(),
    T.Normalize(
        mean = (0.5,0.5,0.5),
         std = (0.5,0.5,0.5)
    )
])

batch_show= True

def train_transforms(batch):
    # convert all images in batch to RGB to avoid grayscale or transparent images
    #batch['image'] = [x.convert('RGB') for x in batch['image']]
    # apply torchvision.transforms per sample in the batch
    inputs = [train_tfms(x) for x in batch['image']]
    batch['pixel_values'] = inputs
    
    # one-hot encoding the labels
    #labels = torch.tensor(nn.utils.rnn.pad_sequence(batch['classes']))
    #classes = [torch.tensor([class_names.index(label) for label in labels], dtype=torch.int64) for labels in batch['label']]
    classes = [torch.tensor([class_names.index(label[1:-1]) for label in labels[1:-1].split(',')], dtype=torch.int64) for labels in batch['label']]
    labels = torch.swapaxes(nn.utils.rnn.pad_sequence(classes), 0, 1)

    #if batch_show:
    #    print(f"batch", classes)
    #    print(f"Batch size: {len(labels)}, Shape of labels: {labels.shape}")
    #    batch_show = False
    batch['labels'] = nn.functional.one_hot(labels,num_classes=class_number).sum(dim=1).clamp(max=1)
    
    return batch

def valid_transforms(batch):
    # convert all images in batch to RGB to avoid grayscale or transparent images
    #batch['image'] = [x.convert('RGB') for x in batch['image']]
    # apply torchvision.transforms per sample in the batch
    inputs = [valid_tfms(x) for x in batch['image']]
    batch['pixel_values'] = inputs
    
    # one-hot encoding the labels
    #labels = torch.tensor(nn.utils.rnn.pad_sequence(batch['classes']))
    #classes = [torch.tensor([class_names.index(label) for label in labels], dtype=torch.int64) for labels in batch['label']]
    classes = [torch.tensor([class_names.index(label[1:-1]) for label in labels[1:-1].split(',')], dtype=torch.int64) for labels in batch['label']]
    labels = torch.swapaxes(nn.utils.rnn.pad_sequence(classes), 0, 1)
    batch['labels'] = nn.functional.one_hot(labels,num_classes=class_number).sum(dim=1).clamp(max=1)
    
    return batch

train_dataset = dataset['train'].with_transform(train_transforms)
test_dataset = dataset['eval'].with_transform(valid_transforms)

len(train_dataset), len(test_dataset)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch]).float()
    }

def param_count(model):
    params = [(p.numel(),p.requires_grad) for p in model.parameters()]
    trainable = sum([count for count,trainable in params if trainable])
    total = sum([count for count,_ in params])
    frac = (trainable / total) * 100
    return total, trainable, frac

def hamming_score(references, predictions):
    score = 0.0
    for i, labels in enumerate(references):
        score += torch.mean(1-torch.abs(labels-predictions[i]))
    return score/len(references)

def train(model_name,batch_size=32,epochs=20,lr=2e-4):
    """
    contains all of our training loops.
    1. define Accelerator instance
    2. define dataloaders, model, optimizer, loss function, scheduler
    3. write training, testing loop.
    """
    
    accelerator = Accelerator() # create instance
    
    # define dataloaders
    
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size, # the batch_size will be per-device
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size*2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # timm model
    model = create_model(
        model_name,
        pretrained = True,
        num_classes = class_number
    ).to(accelerator.device) # device placement: accelerator.device
    #model.to(accelerator.device)
    
    total, trainable, frac = param_count(model)
    accelerator.print(f"{total = :,} | {trainable = :,} | {frac:.2f}%")
    
    # loss, optimizer, scheduler
    
    loss_fn = nn.BCEWithLogitsLoss()
        
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.02)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps = int(0.1 * len(train_dl)),
        num_training_steps=len(train_dl)
    )
    
    model, optimizer, scheduler, train_dl, test_dl = accelerator.prepare(
        model, optimizer, scheduler, train_dl, test_dl
    )

    loss_array, roc_auc_array = [], []
    
    # loops for number of epochs
    for epoch in tqdm(range(1,epochs+1)):
        
        model.train() # set model to train
        
        train_metric = evaluate.load('roc_auc','multilabel') # load metric
        
        running_loss = 0.
        
        for batch in tqdm(train_dl):
            
            logits = model(batch['pixel_values'])
            
            loss = loss_fn(logits,batch['labels'])
            accelerator.backward(loss) # backpropagation
            optimizer.step() # update weights
            scheduler.step() # update LR
            optimizer.zero_grad() # set grad values to zero
            
            running_loss += loss.item() # keep track of loss
            
            # prepare for metrics
            logits, labels = accelerator.gather_for_metrics(
                (logits, batch['labels'])
            )
            train_metric.add_batch(references=labels, prediction_scores=logits)
            
        # loss and metric over 1 epoch
        train_loss = running_loss / len(train_dl)
        train_roc_auc = train_metric.compute(average='micro')['roc_auc']
        loss_array.append(train_loss)
        roc_auc_array.append(train_roc_auc)
        
        accelerator.print(f"\n{epoch = }")
        accelerator.print(f"{train_loss = :.3f} | {train_roc_auc = :.3f}")

        # save model
        accelerator.save_model(model, f'./{model_name}-pascal')
        
    
    # testing loop after all epochs are over
    fig = plt.figure()
    ax0, ax1 = fig.subplots(ncols=2)
    ax0.plot(np.arange(1,epochs+1),np.array(loss_array))
    ax0.set_title("Loss over epochs")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax1.plot(np.arange(1,epochs+1),np.array(roc_auc_array))
    ax1.set_title('ROC AUC over epochs')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("ROC AUC")
    ax1.yaxis.tick_right()
    plt.savefig("multilabel_vit_development.png")
    plt.clf()

    hamm_scores = []
    test_metric = evaluate.load('roc_auc','multilabel')
    
    for batch in tqdm(test_dl):
            
        with torch.no_grad():
            logits = model(batch['pixel_values'])

        logits, labels = accelerator.gather_for_metrics(
            (logits, batch['labels'])
        )
        test_metric.add_batch(references=labels, prediction_scores=logits)
        hamm_score = hamming_score(labels, logits).item()#.cpu().numpy()
        hamm_scores.append(hamm_score)
        #accelerator.print(f"\nHamming score of batch: {hamm_score}")

    test_roc_auc = test_metric.compute(average='micro')['roc_auc']

    accelerator.print(f"\n\nTEST AUROC: {test_roc_auc:.3f}")
    hscores = np.array(hamm_scores)
    plt.bar(np.arange(1,len(test_dl)+1), hscores)
    plt.xlabel("batch ID")
    plt.ylabel("Hamming score")
    plt.title(f"Different hamming scores with mean score {np.mean(hscores)} and test ROC AUC of {test_roc_auc}")
    plt.savefig("test_res")

model_name = 'swin_s3_base_224'
clip = CLIPVisionModel.from_pretrained(os.path.join("clip_pretraining", "checkpoint"))
#print(clip)
#sys.exit()

class MultilabelFromViT(nn.Module):
    def __init__(self, vit_model, num_classes):
        super().__init__()
        self.vit = vit_model
        self.mlp = MLP(in_channels=768, hidden_channels=[num_classes])

    def forward(self, x):
        x = self.__vit.features(x)
        return torch.sigmoid(self.__mlp(x))
    
#model = MultilabelFromViT(clip, class_number)  
train(model_name)

model = create_model(
    model_name,
    num_classes=class_number
)
load_model(model, f'./{model_name}-pascal/model.safetensors')

def extract_vit_from_clip(safetensors_path : str):
    pass

def show_predictions(rows=2,cols=4):
    model.eval()
    samples = test_dataset.shuffle().select(np.arange(rows*cols))
    fig = plt.figure(figsize=(cols*4,rows*4))
    
    for i in range(rows*cols):
        
        img = samples[i]['image']
        inputs = samples[i]['pixel_values'].unsqueeze(0)
        labels = samples[i]['label']
        #labels = ','.join([id2label[lb] for lb in labels])
        
        with torch.no_grad():
            logits = model(inputs)

        # apply sigmoid activation to convert logits to probabilities
        # getting labels with confidence threshold of 0.5
        predictions = logits.sigmoid() > 0.5
        
        # converting one-hot encoded predictions back to list of labels
        predictions = predictions.float().numpy().flatten() # convert boolean predictions to float
        pred_labels = np.where(predictions==1)[0] # find indices where prediction is 1
        pred_labels = ','.join([id2label[label] for label in pred_labels]) # converting integer labels to string
        
        label = f"labels: {labels}\npredicted: {pred_labels}"
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
        plt.savefig('vit_multilabel.png')
            
show_predictions(rows=5,cols=5)
