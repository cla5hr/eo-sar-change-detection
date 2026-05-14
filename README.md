# EO-SAR Change Detection using Deep Learning

This project focuses on binary change detection using co-registered EO and SAR satellite image pairs.

## Features
- Dual encoder EO-SAR fusion network
- U-Net style decoder
- Binary change segmentation
- Threshold optimization
- Visualization generation
- EO-SAR modality fusion

---

## Folder Structure

```text
data/
├── train/
├── val/
└── test/
```

Dataset should contain:
- pre-event EO images
- post-event SAR images
- masks

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python eval.py --weights ./checkpoints/best_model.pth
```

---

## Model Weights

Model weights are available here:

https://huggingface.co/cla5hr/eo-sar-change-detection/resolve/main/best_model.pth

---

## Results

Final Test Metrics:

| Metric | Value |
|---|---|
| IoU | 0.0381 |
| F1-score | 0.0734 |
| Precision | 0.0437 |
| Recall | 0.2271 |

---

## Notes

- Dataset is not included due to size restrictions.
- Black image boundary regions were handled using noise injection preprocessing.
- EO and SAR modalities use separate normalization strategies.
