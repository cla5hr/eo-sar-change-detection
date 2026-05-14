import torch
import yaml
import argparse
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import load_checkpoint

from dataset import ChangeDetectionDataset
from model import DualEncoderFusionNet
from losses import CombinedLoss
from utils import set_seed, compute_metrics, save_checkpoint


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        total_loss += loss.item()

        metrics = compute_metrics(outputs.detach(), masks)
        for k in ['tp', 'fp', 'fn', 'tn']:
            all_metrics[k] += metrics[k]

    avg_loss = total_loss / len(loader)
    tp, fp, fn = all_metrics['tp'], all_metrics['fp'], all_metrics['fn']
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)

    return avg_loss, {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    for images, masks in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)
        total_loss += loss.item()

        metrics = compute_metrics(outputs, masks)
        for k in ['tp', 'fp', 'fn', 'tn']:
            all_metrics[k] += metrics[k]

    avg_loss = total_loss / len(loader)
    tp, fp, fn = all_metrics['tp'], all_metrics['fp'], all_metrics['fn']
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)

    return avg_loss, {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    train_dataset = ChangeDetectionDataset(
        root_dir=cfg['data']['root_dir'],
        split="train",
        image_size=cfg['data']['image_size']
    )
    val_dataset = ChangeDetectionDataset(
        root_dir=cfg['data']['root_dir'],
        split="val",
        image_size=cfg['data']['image_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )

    # Model
    model = DualEncoderFusionNet(
        encoder_name=cfg['model']['encoder_name'],
        pretrained=cfg['model']['pretrained']
    ).to(device)

    # Loss + Optimizer + Scheduler
    criterion = CombinedLoss(
        focal_weight=cfg['training']['focal_weight'],
        tversky_weight=cfg['training']['tversky_weight']
    )
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['epochs'],
        eta_min=1e-6
    )
    # After model creation, load previous checkpoint
    # if os.path.exists('./checkpoints/best_model.pth'):
    #     load_checkpoint(model, './checkpoints/best_model.pth', optimizer)
    #     print("Resumed from checkpoint")
    best_f1 = 0.0
    save_dir = cfg['checkpoint']['save_dir']

    print(f"\nStarting training for {cfg['training']['epochs']} epochs\n")

    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{cfg['training']['epochs']} | "
              f"LR: {current_lr:.7f} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f} | "
              f"Val Precision: {val_metrics['precision']:.4f} | "
              f"Val Recall: {val_metrics['recall']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(save_dir, 'best_model.pth')
            )
            print(f"  → New best F1: {best_f1:.4f}")

    # Save final model
    save_checkpoint(
        model, optimizer, cfg['training']['epochs'], val_metrics,
        os.path.join(save_dir, 'final_model.pth')
    )
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(args.config)