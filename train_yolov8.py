import json
import os
import ultralytics
import wandb
import yaml

from types import SimpleNamespace
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

import params


BASE_MODEL = "yolov8x" # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
DETECTOR_MODE = "detector"
FULL_MODE = "full"

config = SimpleNamespace(
    base_model = BASE_MODEL,
    epochs = 30,
    batch_size = 24,
    base_model_weights = f"{BASE_MODEL}.pt",
    optimizer = "auto", # SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto
    lr = 1e-4,
    lr_factor = 0.01,
    weight_decay = 5e-4,
    mixed_precision = True,
    training_mode= DETECTOR_MODE,
    seed = 24,
    save_model = False
)

def freeze_layers(trainer):
    model = trainer.model
    num_freeze = 22
    print(f"Freezing {num_freeze} layers")

    for k, v in model.model.named_parameters():
        v.requires_grad =  True  # train all layers
        if int(k.split(".")[0]) < num_freeze:
            print(f"freezing {k}")
            v.requires_grad =  False
    print(f"{num_freeze} layers are freezed." )

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua): return True
    else: return False

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=config.epochs, help='number of training epochs')
    argparser.add_argument('--learning_rate', type=float, default=config.learning_rate, help='learning rate')
    argparser.add_argument('--pretrained', type=t_or_f, default=config.pretrained, help='Use pretrained model')
    argparser.add_argument('--mixed_precision', type=t_or_f, default=config.mixed_precision, help='use fp16')
    argparser.add_argument('--training_mode', type=str, default=config.training_mode, help='Training mode - full or just the last layer')
    argparser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    argparser.add_argument('--save_model', type=t_or_f, default=config.save_model, help='save model after training')

    args = argparser.parse_args()
    vars(config).update(vars(args))


if __name__ == "__main__":
    parse_args()
    gyorun = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="train_yolov8")
    artifact = run.use_artifact(f"{params.PROCESSED_YOLO_DATA_AT}:latest", type='dataset')
    artifact_dir = artifact.download()

    artifact_local_path = "/".join(artifact_dir.split("/")[-2:])
    with open(os.path.join(artifact_local_path, "data.yaml"), "r") as f:
        dict_file = yaml.safe_load(f)
    new_path = os.path.join("..", artifact_local_path)
    dict_file["path"] = new_path
    with open(os.path.join(artifact_local_path, "data.yaml"), "w") as f:
        yaml.dump(dict_file, f)

    model = YOLO(config.base_model_weights)
    if config.training_mode == DETECTOR_MODE:
        model.add_callback( "on_train_start", freeze_layer)
    add_wandb_callback(model, enable_model_checkpointing=False)
    model.train(
        data = os.path.join(artifact_local_path, 'data.yaml'),
        task = "detect",
        imgsz = (714, 368),
    
        epochs = config.epochs,
        batch = config.batch_size,
        optimizer = config.optimizer,
        lr0 = config.lr,
        lrf = config.lr_factor,
        weight_decay = config.weight_decay,
        patience = 10,
    
        name = f'{config.base_model}_init_train',
        seed = config.seed,
        
        save_period = 10,
        val = True,
        amp = config.mixed_precision,
        exist_ok = True,
        resume = False,
        device = 0,
        verbose = False,
    )
    wandb.finish()
    if config.save_model:
        model.save_model()