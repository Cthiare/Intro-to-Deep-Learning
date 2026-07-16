# ECGR 4106/5106 — Homework 5

Vision Transformers and Swin Transformers on CIFAR-100.

## Files

- `ECGR4106_HW5_ViT_Swin.ipynb` — notebook for both problems
- `HW5_Report.pdf` — report

## Problem 1 — ViT from scratch vs ResNet-18

Batch 64, 10 epochs, Adam (lr 0.001).

| Model | Params | GFLOPs | Test acc (%) |
|---|---|---|---|
| ViT-base (p4/d256/L4/h4) | 3,214,692 | 0.412 | 25.83 |
| ViT-p8 (p8/d256/L4/h4) | 3,239,268 | 0.109 | 13.13 |
| ViT-d512 (p4/d512/L4/h4) | 12,720,740 | 1.642 | 7.97 |
| ViT-L8 (p4/d256/L8/h4) | 6,373,732 | 0.822 | 19.92 |
| ViT-h8 (p4/d256/L4/h8) | 3,214,692 | 0.412 | 32.08 |
| ResNet-18 | 11,220,132 | 1.113 | **57.36** |

## Problem 2 — Swin fine-tuning vs from scratch

Batch 32, 5 epochs, Adam. Pretrained models use a frozen backbone (lr 2e-5), scratch uses lr 0.001.

| Model | Params | Trainable | Test acc (%) |
|---|---|---|---|
| Swin-Tiny (pretrained, head only) | 27,596,254 | 76,900 | 66.07 |
| Swin-Small (pretrained, head only) | 48,914,158 | 76,900 | **69.57** |
| Swin-Tiny (from scratch) | 27,596,254 | 27,596,254 | 1.00 |

## Running

Open the notebook in Google Colab, set the runtime to GPU, and run all cells. Results are saved to `hw5_results/` as they finish, so a dropped session doesn't restart training.
