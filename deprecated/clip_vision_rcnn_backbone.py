import os
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from transformers import TrainingArguments, Trainer, CLIPModel, CLIPProcessor, CLIPVisionModel

save_model_path = "detection_clip"

def create_model(num_classes, clip_path=os.path.join("clip_pretraining", "checkpoint")):
    # Load the pretrained SqueezeNet1_0 backbone.
    backbone = CLIPVisionModel.from_pretrained(clip_path).features
    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    backbone.out_channels = 512
    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    #print(model)
    return model



def train(model_name,batch_size=16,epochs=1,lr=2e-4):
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
    # loops for number of epochs
    for epoch in range(1,epochs+1):
        
        model.train() # set model to train
        
        train_metric = evaluate.load('roc_auc','multilabel') # load metric
        
        running_loss = 0.
        
        for batch in train_dl:
            
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
        
        accelerator.print(f"\n{epoch = }")
        accelerator.print(f"{train_loss = :.3f} | {train_roc_auc = :.3f}")

        # save model
        accelerator.save_model(model, f'./{model_name}-pascal')
        
    
    # testing loop after all epochs are over
    
    test_metric = evaluate.load('roc_auc','multilabel')
    
    for batch in test_dl:
            
        with torch.no_grad():
            logits = model(batch['pixel_values'])

        logits, labels = accelerator.gather_for_metrics(
            (logits, batch['labels'])
        )
        test_metric.add_batch(references=labels, prediction_scores=logits)

    test_roc_auc = test_metric.compute(average='micro')['roc_auc']

    accelerator.print(f"\n\nTEST AUROC: {test_roc_auc:.3f}")
