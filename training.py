"""
-------------------------------------------
Baseball Detection — Training Pipeline
-------------------------------------------
Faster R-CNN is a CNN-based object detection model that predicts both a class label and bounding box coordinates for objects in an image.

Inputs (produced by main.py pre-processing pipeline):
    frames/              ← extracted .jpg frames per video
    annotations.csv      ← parsed CVAT bounding boxes

Outputs:
    baseball_model.pt    ← trained model weights
    training_log.csv     ← loss per epoch
"""

import os
import csv
import ast
import torch
import torchvision
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

ANNOTATIONS_CSV = "annotations.csv"
FRAMES_DIR      = "frames"
MODEL_OUT       = "baseball_model.pt"
LOG_OUT         = "training_log.csv"

NUM_EPOCHS      = 5     
BATCH_SIZE      = 2
LEARNING_RATE   = 0.005
VAL_SPLIT       = 0.2
NUM_CLASSES     = 2

class BaseballDataset(Dataset):
    def __init__(self, records, frames_dir, transform=None):
        self.records = records
        self.frames_dir = frames_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]

        video_name = os.path.splitext(os.path.basename(row["video"]))[0]
        frame_num  = int(row["frame"])
        frame_file = f"frame_{frame_num:04d}.jpg"
        frame_path = os.path.join(self.frames_dir, video_name, frame_file)

        image = Image.open(frame_path).convert("RGB")

        box = [
            float(row["xtl"]),
            float(row["ytl"]),
            float(row["xbr"]),
            float(row["ybr"]),
        ]
        boxes  = torch.tensor([box], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))

def build_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


def compute_iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def evaluate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    iou_scores = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                gt_boxes   = target["boxes"].cpu()
                pred_boxes = output["boxes"].cpu()
                scores     = output["scores"].cpu()

                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    iou_scores.append(0.0)
                    continue

                best_idx = scores.argmax()
                best_box = pred_boxes[best_idx]
                gt_box   = gt_boxes[0]

                iou = compute_iou(best_box.tolist(), gt_box.tolist())
                iou_scores.append(iou)

    mean_iou       = np.mean(iou_scores) if iou_scores else 0.0
    detection_rate = np.mean([s >= iou_threshold for s in iou_scores]) if iou_scores else 0.0

    return mean_iou, detection_rate

def main():
    print("\nBaseball Detection Training Pipeline")
    print("=" * 60)

    #Load annotations
    if not os.path.exists(ANNOTATIONS_CSV):
        print(f"[ERROR] '{ANNOTATIONS_CSV}' not found.")
        print("Run main.py first to generate annotations.")
        return

    df = pd.read_csv(ANNOTATIONS_CSV)

    #Keep ball in frame
    df = df[df["outside"] == 0].reset_index(drop=True)
    print(f"\nLabeled frames available for training: {len(df)}")

    if len(df) < 4:
        print("[ERROR] Not enough labeled frames to train. Need at least 4.")
        return

    records = df.to_dict("records")

    #train and validate split
    n_val   = max(1, int(len(records) * VAL_SPLIT))
    n_train = len(records) - n_val

    np.random.seed(42)
    idx = np.random.permutation(len(records))
    train_records = [records[i] for i in idx[:n_train]]
    val_records   = [records[i] for i in idx[n_train:]]

    print(f"Train frames: {len(train_records)}  |  Val frames: {len(val_records)}")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = BaseballDataset(train_records, FRAMES_DIR, transform)
    val_dataset   = BaseballDataset(val_records,   FRAMES_DIR, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=1,
                              shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading pre-trained Faster R-CNN (MobileNetV3 backbone)...")
    model = build_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE,
                                momentum=0.9, weight_decay=0.0005)

    #Trainiing loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...\n")
    log_rows = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        mean_iou, det_rate = evaluate(model, val_loader, device)

        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  |  "
              f"Loss: {train_loss:.4f}  |  "
              f"Mean IoU: {mean_iou:.3f}  |  "
              f"Detection Rate: {det_rate*100:.1f}%")

        log_rows.append({
            "epoch":          epoch,
            "train_loss":     round(train_loss, 4),
            "mean_iou":       round(mean_iou, 4),
            "detection_rate": round(det_rate, 4),
        })

    #Save weights
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nModel weights saved to '{MODEL_OUT}'")

    #Save training
    with open(LOG_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "mean_iou", "detection_rate"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Training log saved to '{LOG_OUT}'")

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Final Mean IoU:       {log_rows[-1]['mean_iou']}")
    print(f"  Final Detection Rate: {log_rows[-1]['detection_rate']*100:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()