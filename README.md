# ECGR 4106/5106 — Homework 1

CNN architecture comparison on CIFAR-10 using PyTorch.

## Contents

- `ECGR4106_HW1_CIFAR10.ipynb` — full notebook (all training runs and plots)

## Overview

- **Problem 1:** Modified AlexNet adapted for 32x32 inputs, plus dropout experiments (p = 0.3, 0.5)
- **Problem 2:** Adapted VGG-11 (closest parameter count to the AlexNet), same dropout comparison
- **Problem 3:** ResNet-18 from scratch vs. ResNet-11 baseline, dropout after global average pooling, and a final comparison across all four architectures

## Setup

Trained on Google Colab (T4 GPU). SGD (lr 0.01, momentum 0.9), batch size 128, seed 42, fixed 45k/5k/10k train/val/test split. 30 epochs for Problems 1–2, 50 epochs with cosine annealing for Problem 3.

## Results (test accuracy, best variant)

| Model | Accuracy |
|---|---|
| Modified AlexNet | 86.03% |
| VGG-11 | 84.12% |
| ResNet-11 | 91.08% |
| ResNet-18 | 91.88% |

Full analysis is in the homework report (PDF).
