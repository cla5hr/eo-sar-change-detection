import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(preds, targets, threshold=0.2):
    # preds: (B, 1, H, W) logits
    # targets: (B, H, W) long
    probs = torch.sigmoid(preds.squeeze(1))
    binary = (probs > threshold).long()
    targets = targets.long()

    # Flatten
    pred_flat   = binary.view(-1)
    target_flat = targets.view(-1)

    tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)

    return {
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'iou':       iou,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def save_checkpoint(model, optimizer, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, path, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint['metrics']


if __name__ == "__main__":
    logits  = torch.randn(2, 1, 256, 256)
    targets = torch.randint(0, 2, (2, 256, 256))
    metrics = compute_metrics(logits, targets)
    print(metrics)