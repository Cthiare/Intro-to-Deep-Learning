# ECGR 4106/5106 — Homework 2

Character-level next-character prediction with recurrent models (RNN, LSTM, GRU) using PyTorch.

## Contents

- `HW2_Recurrent_Models.ipynb` — full notebook (all training runs, tables, and plots)

## Overview

- **Problem 1:** RNN vs. LSTM vs. GRU on the supplied passage, trained at sequence lengths 10, 20, and 30. Compares training loss, validation accuracy, training time, and model/computational complexity.
- **Problem 2:** LSTM vs. GRU on the tiny Shakespeare dataset.
  - Sequence lengths 20 and 30 — compare loss, accuracy, time, and complexity.
  - Hyperparameter study — hidden size, number of recurrent layers, and an added fully-connected head; analyzes the effect on accuracy, runtime, and perplexity, with sample generated text.
  - Sequence length 50 — accuracy and model-complexity results.

## Setup

Trained on Google Colab (T4 GPU). Architecture: character embedding -> recurrent layer(s) -> linear head on the last time step. Adam optimizer, CrossEntropyLoss, seed 42, 80/20 train/validation split. Problem 1: the 2,386-character passage, full-batch, 300 epochs. Problem 2: the full tiny Shakespeare dataset (1,115,394 characters), batch size 256, 10 epochs (seq 20/30 and seq 50) and 8 epochs (hyperparameter sweep).

## Results (tiny Shakespeare, validation)

| Model / config | Val accuracy | Perplexity |
|---|---|---|
| GRU (seq 30) | 55.55% | 4.34 |
| LSTM (seq 30) | 57.89% | 3.96 |
| LSTM + extra FC head | 58.08% | 3.91 |
| LSTM, 2 recurrent layers (best) | 58.69% | 3.83 |

On Problem 1, GRU generalized best (52.95% validation accuracy at sequence length 20).

Full analysis is in the homework report (PDF).
