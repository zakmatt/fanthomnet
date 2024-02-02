import albumentations
import argparse
import functools
import json
import numpy as np
import os
import tokenizers
import torch
import wandb

from glob import glob
from pathlib import Path
from PIL import Image, ImageDraw
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    EvalPrediction,
    Trainer,
    TrainingArguments
)
from types import SimpleNamespace
from typing import Dict, List, Tuple

import params

DATASET_FOLDER = "training_dataset"
TRAIN_DATA = "train_data.json"
VAL_DATA = "val_data.json"
IMAGES_DIR = "images"
IMAGE_SIZE = 480
SEED = 42

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua): return True
    else: return False

config = SimpleNamespace(
    framework="huggingface",
    #img_size=IMAGE_SIZE,
    batch_size=8,
    augment=True,
    epochs=10,
    learning_rate=1e-5,
    weight_decay=1e-4,
    pretrained=True,
    mixed_precision=True,
    model_name="yolos",
    training_mode="bbox_classifier",
    seed=SEED,
    log_preds=False,
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=config.epochs, help='number of training epochs')
    argparser.add_argument('--learning_rate', type=float, default=config.learning_rate, help='learning rate')
    argparser.add_argument('--model_name', type=str, default=config.model_name, help='architecture')
    argparser.add_argument('--augment', type=t_or_f, default=config.augment, help='Use image augmentation')
    argparser.add_argument('--training_mode', type=t_or_f, default=config.training_mode, help='Training mode - full or just the last layer')
    argparser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    argparser.add_argument('--log_preds', type=t_or_f, default=config.log_preds, help='log model predictions')
    argparser.add_argument('--pretrained', type=t_or_f, default=config.pretrained, help='Use pretrained model')
    argparser.add_argument('--mixed_precision', type=t_or_f, default=config.mixed_precision, help='use fp16')
    args = argparser.parse_args()
    vars(config).update(vars(args))


def download_data():
    """Download the training & validation datatsets"""

    dataset_artifact = wandb.use_artifact(f'{params.FINAL_DATA_AT}:latest')
    dataset_path = Path(dataset_artifact.download())
    return dataset_path


def process_data(data: dict, data_path: str):
    """Convert data format

    Convert data format from json with multiple keys to a list of [image_info, image annotations]
    :param data: input data
    :type data: json
    :param data_path: path to images
    :type data_path: str
    :return: converted data into a list
    :rtype: [dict, [dict]]
    """

    processed_data = []
    for image_data in tqdm(data["images"]):
        image_path = os.path.join(data_path, image_data["file_name"])
        if not os.path.isfile(image_path):
            continue
        
        height = image_data["height"]
        width = image_data["width"]
        curr_image_id = image_data["id"]
        curr_annos = []
        for anno in data["annotations"]:
            if anno["image_id"] != curr_image_id:
                continue
            
            x, y, w, h = anno["bbox"]
            if w == 0 or h == 0:
                continue
            if x < 0:
                x = 0
            if y < 0:
                y = 0
    
            if x >= width:
                continue
            if x + w > width:
                w = width - x
            
            if y >= height:
                continue
            if y + h > height:
                h = height - y
            
            anno["bbox"] = [x, y, w, h]
            curr_annos.append(anno)

        if len(curr_annos) == 0:
            continue
    
        curr_image_data = image_data.copy()
        curr_image_data["image_path"] = image_path
        processed_data.append((curr_image_data, curr_annos))

    return processed_data


class SeaWorldDataset(Dataset):
    """
    """

    def __init__(self, data: List[Tuple[Dict, List[Dict]]], image_processor, is_augment: str = True):
        self._data = data
        self._image_processor = image_processor

        if is_augment:
            self._transform = albumentations.Compose(
                [
                    albumentations.ShiftScaleRotate(
                        shift_limit=(0, 0.3),
                        scale_limit=0.2,
                        rotate_limit=(-20, 20)
                    ),
                    albumentations.HorizontalFlip(p=.5),
                    albumentations.VerticalFlip(p=.5),
                    albumentations.RandomBrightnessContrast(p=.5),
                    albumentations.LongestMaxSize(image_processor.size["longest_edge"], always_apply=True),
                ],
                bbox_params=albumentations.BboxParams(format="coco", label_fields=["categories"]),
            )
        else:
            self._transform = albumentations.Compose(
                [
                    albumentations.LongestMaxSize(image_processor.size["longest_edge"], always_apply=True),
                ],
                bbox_params=albumentations.BboxParams(format="coco", label_fields=["categories"]),
            )

    def __len__(self):
        return len(self._data)

    @staticmethod
    def _format_processed_annotations(bboxes, categories, image_id):
        """
        """

        processed_annotations = []
        for bbox, category_id in zip(bboxes, categories):
            processed_annotations.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [int(arg + 0.5) for arg in bbox],
                    "isCrowd": 0,
                    "area": int(bbox[2] * bbox[3] + 0.5)
                }
            )

        return processed_annotations

    def __getitem__(self, index) -> Tuple[np.ndarray, List[Dict], int]:
        """
        """

        image_data, annotations = self._data[index]
        image_id = image_data["id"]
        bboxes, categories = [], []
        for annotation in annotations:
            bbox = annotation["bbox"]
            category_id = annotation["category_id"]
            bboxes.append(bbox)
            categories.append(category_id)

        image = np.array(Image.open(image_data["image_path"]).convert("RGB"))
        out = self._transform(
            image=image,
            bboxes=bboxes,
            categories=categories
        )
        annotations = self._format_processed_annotations(
            out["bboxes"], out["categories"], image_id
        )

        processed_image = self._image_processor.preprocess(
            images=image, annotations={"image_id": image_id, "annotations": annotations}, return_tensors="pt"
        )
        for key, val in processed_image.items():
            processed_image[key] = val[0]

        return processed_image


def collate_fn(batch, is_yolo=False):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    if not is_yolo:
        batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


def get_training_params(config, label2id, id2label):
    def create_model(checkpoint):
        return AutoModelForObjectDetection.from_pretrained(
            checkpoint,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    
    is_yolo = False
    if config.model_name == "yolos":
        checkpoint = "hustvl/yolos-small"
        is_yolo = True
        model = create_model(checkpoint)
        #if config.training_mode == "bbox_classifier":
        for param in model.vit.parameters():
            param.requires_grad = False
    else:
        if config.model_name == "DETA":
            checkpoint = "jozhang97/deta-swin-large"
        elif config.model_name == "CondDETR":
            checkpoint = "microsoft/conditional-detr-resnet-50"

        model = create_model(checkpoint)
        #if config.training_mode == "bbox_classifier":
        for param in model.model.parameters():
            param.requires_grad = False

    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return model, image_processor, is_yolo


class DetectionTrainer(Trainer):
    def compute_metrics(self, eval_pred: EvalPrediction):
        """Compute detection metrics"""

        _, scores, pred_boxes, last_hidden_state, encoder_last_hidden_state = eval_pred.predictions
        
        # scores shape: (number of samples, number of detected anchors, num_classes + 1) last class is the no-object class
        # pred_boxes shape: (number of samples, number of detected anchors, 4)
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/detr-resnet50/README.md
        predictions = []
        for score, box in zip(scores, pred_boxes):
            # Extract the bounding boxes, labels, and scores from the model's output
            pred_scores = torch.from_numpy(score[:, :-1])  # Exclude the no-object class
            pred_boxes = torch.from_numpy(box)
            pred_labels = torch.argmax(pred_scores, dim=-1)
    
            # Get the scores corresponding to the predicted labels
            pred_scores_for_labels = torch.gather(pred_scores, 1, pred_labels.unsqueeze(-1)).squeeze(-1)
            predictions.append(
                {
                    "boxes": pred_boxes,
                    "scores": pred_scores_for_labels,
                    "labels": pred_labels,
                }
            )
        eval_data = self.eval_dataset
        target = [
            {
                "boxes": self.eval_dataset[i]["labels"]["boxes"],
                "labels": self.eval_dataset[i]["labels"]["class_labels"],
            }
            for i in range(len(self.eval_dataset))
        ]
        map.update(preds=predictions, target=target)
        results = map.compute()
        results = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in results.items()}
        return results


if __name__ == "__main__":
    parse_args()
    #run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="training", config=config)
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="training_final_model", config=training_args)
    data_path = download_data()
    with open(os.path.join(data_path, DATASET_FOLDER, TRAIN_DATA), "r") as f:
        train_data = json.load(f)
    with open(os.path.join(data_path, DATASET_FOLDER, VAL_DATA), "r") as f:
        val_data = json.load(f)
    id2label = {cat["id"]: cat["name"] for cat in val_data["categories"]}
    label2id = {name: cat_id for cat_id, name in id2label.items()}
    images_path = os.path.join(data_path, DATASET_FOLDER, IMAGES_DIR)
    val_data = process_data(val_data, images_path)
    train_data = process_data(train_data, images_path)

    model, image_processor, is_yolo = get_training_params(
        config, label2id, id2label
    )
    train_dataset = SeaWorldDataset(train_data, image_processor, is_augment=True)
    val_dataset = SeaWorldDataset(val_data, image_processor, is_augment=False)

    
    run_name = f"{config.model_name}_{config.training_mode}_tuned"
    training_args = TrainingArguments(
        output_dir=run_name,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        logging_steps=10,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        save_total_limit=2,
        fp16=config.mixed_precision,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        run_name=run_name
    )

    trainer = DetectionTrainer(
        model=model,
        args=training_args,
        data_collator=lambda batch: collate_fn(batch, False),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor
    )
    trainer.train()
    wandb.finish()