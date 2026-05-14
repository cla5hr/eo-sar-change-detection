# optuna_search.py

import os
import yaml
import optuna
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import ChangeDetectionDataset
from model import DualEncoderFusionNet
from losses import CombinedLoss
from utils import compute_metrics


# =========================
# Reproducibility
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Train
# =========================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# Validation
# =========================

@torch.no_grad()
def validate(model, loader, criterion, device, threshold):
    model.eval()

    total_loss = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        total_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()

        metrics = compute_metrics(outputs, masks, threshold)

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_tn += metrics["tn"]

    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)

    f1 = (2 * precision * recall) / (precision + recall + 1e-7)

    iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)

    return {
        "loss": total_loss / len(loader),
        "f1": f1,
        "iou": iou,
        "precision": precision,
        "recall": recall
    }


# =========================
# Objective
# =========================

def objective(trial):

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Hyperparameters
    # -------------------------

    alpha = trial.suggest_float("alpha", 0.2, 0.8)
    beta = 1.0 - alpha

    focal_weight = trial.suggest_float("focal_weight", 0.2, 0.8)
    tversky_weight = 1.0 - focal_weight

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)

    threshold = trial.suggest_float("threshold", 0.01, 0.4)

    # -------------------------
    # Dataset
    # -------------------------

    train_dataset = ChangeDetectionDataset(
        root_dir=cfg["data"]["root_dir"],
        split="train",
        image_size=cfg["data"]["image_size"]
    )

    val_dataset = ChangeDetectionDataset(
        root_dir=cfg["data"]["root_dir"],
        split="val",
        image_size=cfg["data"]["image_size"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"]
    )

    # -------------------------
    # Model
    # -------------------------

    model = DualEncoderFusionNet(
        encoder_name=cfg["model"]["encoder_name"],
        pretrained=cfg["model"]["pretrained"]
    ).to(device)

    # -------------------------
    # Loss
    # -------------------------

    criterion = CombinedLoss(
        focal_weight=focal_weight,
        tversky_weight=tversky_weight,
        alpha=alpha,
        beta=beta
    )

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"]
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    # -------------------------
    # Train loop
    # -------------------------

    best_f1 = 0

    for epoch in range(10):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            threshold
        )

        scheduler.step()

        best_f1 = max(best_f1, val_metrics["f1"])

        print(
            f"\nTrial {trial.number} | "
            f"Epoch {epoch+1}/10 | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f}"
        )

        # -------------------------
        # Pruning
        # -------------------------

        trial.report(val_metrics["f1"], epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_f1


# =========================
# Main
# =========================

if __name__ == "__main__":

    set_seed(42)

    study = optuna.create_study(
        direction="maximize",
        study_name="change_detection",
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(
        objective,
        n_trials=30
    )

    print("\n======================")
    print("BEST TRIAL")
    print("======================")

    print(f"F1: {study.best_value:.4f}")

    print("\nBest Params:")

    for k, v in study.best_params.items():
        print(f"{k}: {v}")