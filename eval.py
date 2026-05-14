import torch
import yaml
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChangeDetectionDataset
from model import DualEncoderFusionNet
from utils import compute_metrics, load_checkpoint


def evaluate(model, loader, device, threshold=0.5):

    model.eval()

    all_metrics = {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'tn': 0
    }

    with torch.no_grad():

        for images, masks in tqdm(
            loader,
            desc="Evaluating"
        ):

            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)

            # Zero out predictions in no-data regions
            no_data = (
                images[:, :3, :, :].sum(
                    dim=1,
                    keepdim=True
                ) == 0
            )

            outputs[no_data] = -10

            metrics = compute_metrics(
                outputs,
                masks,
                threshold
            )

            for k in ['tp', 'fp', 'fn', 'tn']:

                all_metrics[k] += metrics[k]

    tp = all_metrics['tp']
    fp = all_metrics['fp']
    fn = all_metrics['fn']
    tn = all_metrics['tn']

    precision = tp / (tp + fp + 1e-8)

    recall = tp / (tp + fn + 1e-8)

    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )

    iou = tp / (tp + fp + fn + 1e-8)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'confusion_matrix': np.array([
            [tn, fp],
            [fn, tp]
        ])
    }


def visualize_predictions(
    model,
    dataset,
    device,
    threshold=0.1,
    save_dir="./visualizations"
):

    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():

        for idx in tqdm(
            range(len(dataset)),
            desc="Saving predictions"
        ):

            image, mask = dataset[idx]

            output = model(
                image.unsqueeze(0).to(device)
            )

            # Zero out no-data regions
            no_data = (
                image[:3].sum(dim=0) == 0
            )

            output_np = output.squeeze()

            output_np[no_data] = -10

            probs = torch.sigmoid(output_np)

            pred = (
                probs > threshold
            ).cpu().numpy()

            image_np = image.numpy()

            eo = np.transpose(
                image_np[:3],
                (1, 2, 0)
            )

            sar = image_np[3]

            # De-normalize EO image
            mean = np.array([
                0.485,
                0.456,
                0.406
            ])

            std = np.array([
                0.229,
                0.224,
                0.225
            ])

            eo = np.clip(
                eo * std + mean,
                0,
                1
            )

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(16, 4)
            )

            axes[0].imshow(eo)
            axes[0].set_title(
                "EO (Pre-event)"
            )
            axes[0].axis('off')

            axes[1].imshow(
                sar,
                cmap='gray'
            )
            axes[1].set_title(
                "SAR (Post-event)"
            )
            axes[1].axis('off')

            axes[2].imshow(
                mask.numpy(),
                cmap='gray'
            )
            axes[2].set_title(
                "Ground Truth"
            )
            axes[2].axis('off')

            axes[3].imshow(
                pred,
                cmap='gray'
            )
            axes[3].set_title(
                f"Prediction (t={threshold})"
            )
            axes[3].axis('off')

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    save_dir,
                    f"sample_{idx:03d}.png"
                ),
                dpi=150
            )

            plt.close()

    print(
        f"\nSaved all visualizations to: "
        f"{save_dir}"
    )


def main(
    config_path,
    weights_path,
    data_path=None,
    split="test"
):

    with open(config_path) as f:

        cfg = yaml.safe_load(f)

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    root_dir = (
        data_path
        if data_path
        else cfg['data']['root_dir']
    )

    dataset = ChangeDetectionDataset(
        root_dir=root_dir,
        split=split,
        image_size=cfg['data']['image_size']
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers']
    )

    model = DualEncoderFusionNet(
        encoder_name=cfg['model']['encoder_name'],
        pretrained=False
    ).to(device)

    load_checkpoint(
        model,
        weights_path
    )

    print(
        f"\nThreshold sweep on "
        f"{split} set:\n"
    )

    print(
        f"{'Threshold':<12}"
        f"{'F1':<10}"
        f"{'IoU':<10}"
        f"{'Precision':<12}"
        f"{'Recall':<10}"
    )

    best_f1 = 0
    best_threshold = 0.1

    thresholds = [
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
        0.03
    ]

    for threshold in thresholds:

        metrics = evaluate(
            model,
            loader,
            device,
            threshold
        )

        print(
            f"{threshold:<12.3f}"
            f"{metrics['f1']:<10.4f}"
            f"{metrics['iou']:<10.4f}"
            f"{metrics['precision']:<12.4f}"
            f"{metrics['recall']:<10.4f}"
        )

        if metrics['f1'] > best_f1:

            best_f1 = metrics['f1']
            best_threshold = threshold

    print(
        f"\nBest threshold: "
        f"{best_threshold}"
        f" | Best F1: "
        f"{best_f1:.4f}"
    )

    # Final evaluation
    print(
        f"\n--- Final Results "
        f"({split}) ---"
    )

    metrics = evaluate(
        model,
        loader,
        device,
        best_threshold
    )

    print(
        f"IoU:       "
        f"{metrics['iou']:.4f}"
    )

    print(
        f"F1:        "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(f"\nConfusion Matrix:")

    print(
        f"TN: "
        f"{metrics['confusion_matrix'][0,0]:,}"
        f" | FP: "
        f"{metrics['confusion_matrix'][0,1]:,}"
    )

    print(
        f"FN: "
        f"{metrics['confusion_matrix'][1,0]:,}"
        f" | TP: "
        f"{metrics['confusion_matrix'][1,1]:,}"
    )

    print("\nGenerating visualizations...")

    visualize_predictions(
        model,
        dataset,
        device,
        threshold=best_threshold,
        save_dir="./visualizations"
    )

    print("\nDone!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml'
    )

    parser.add_argument(
        '--weights',
        type=str,
        default='./checkpoints/best_model.pth'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default=None
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test'
    )

    args = parser.parse_args()

    main(
        args.config,
        args.weights,
        args.data_path,
        args.split
    )