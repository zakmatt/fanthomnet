import json
import os
import ultralytics
import wandb
import yaml

from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

import params

BASE_MODEL = 'yolov8x' # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
EPOCHS = 20
BATCH_SIZE = 16
BASE_MODEL_WEIGHTS = f'{BASE_MODEL}.pt'
OPTIMIZER = 'auto' # SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto
LR = 1e-3
LR_FACTOR = 0.01
WEIGHT_DECAY = 5e-4

if __name__ == "__main__":
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="train_yolov8")
    artifact = run.use_artifact(f"{params.PROCESSED_YOLO_DATA_AT}:latest", type='dataset')
    artifact_dir = artifact.download()
    model = YOLO(BASE_MODEL_WEIGHTS)
    for k, v in model.model.model.named_parameters():
        if not k.startswith("22"):
            v.requires_grad = False
    add_wandb_callback(model, enable_model_checkpointing=True)
    model.train(
        data = os.path.join(artifact_dir, 'data.yaml'),
        task = 'detect',
        imgsz = (714, 368),
    
        epochs = EPOCHS,
        batch = 16, #BATCH_SIZE,
        optimizer = OPTIMIZER,
        lr0 = LR,
        lrf = LR_FACTOR,
        weight_decay = WEIGHT_DECAY,
        patience = 10,
    
        name = f'{BASE_MODEL}_init_train',
        seed = 42,
        
        save_period = 1,
        val = True,
        amp = True,    
        exist_ok = True,
        resume = False,
        device = 0,
        verbose = False,
    )
    wandb.finish()