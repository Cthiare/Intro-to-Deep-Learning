# ECGR 4106/5106 — Homework 4

Transformers for character prediction and machine translation, in PyTorch.

## Contents

- `Homework_4_Transformers.ipynb` — full notebook (all four problems, training, tables, and plots)
- `vast_english_french.txt` — translation dataset (555 English-French sentence pairs)

## Overview

- **Problem 1:** Character-level Transformer on the lecture passage at sequence lengths 10, 20, and 30, compared against the HW2 RNN/LSTM/GRU baselines.
- **Problem 2:** Character-level Transformer on Tiny Shakespeare — a 2-block/2-head starting point at sequence lengths 20 and 30, a blocks × heads sweep, and a sequence-length-50 run.
- **Problem 3:** Transformer encoder-decoder with cross-attention for English → French translation, with a blocks × heads sweep and a comparison against the HW3 GRU models.
- **Problem 4:** The same translation task reversed (French → English), plus a comparison of the two directions.

## Setup

Trained on Google Colab (GPU), seed 42. Problems 1 and 2 use an encoder-style Transformer that reads a fixed window of characters and predicts the next character, matching how the HW2 recurrent models worked. Problems 3 and 4 use a full Transformer encoder-decoder for word-level translation on the same 80/20 split as HW3. All models use sinusoidal positional encoding, Adam, and cross-entropy loss. The block and head values (blocks ∈ {1, 2, 4}, heads ∈ {2, 4}) give six configurations per sweep.

Problem 2's Transformer was trained on a 300,000-character slice of Tiny Shakespeare to keep the runs inside one session, while the HW2 LSTM/GRU baselines used the full 1,115,394 characters. Problems 1, 3, and 4 use the same data as their baselines.

## Results

Character prediction (Tiny Shakespeare, sequence length 30):

| Model | Val accuracy | Perplexity | Params |
|---|---|---|---|
| Transformer 2blk-4h (best) | 37.89% | 8.12 | 1,611,326 |
| LSTM (HW2) | 57.89% | 3.96 | 559,681 |
| GRU (HW2) | 55.55% | 4.34 | 428,097 |

Translation (validation BLEU-4, best configuration 2blk-2h):

| Model | EN → FR | FR → EN |
|---|---|---|
| Transformer | 0.1898 | 0.1776 |
| GRU + attention (HW3) | 0.1514 | 0.1088 |
| GRU baseline (HW3) | 0.1500 | 0.1328 |

The Transformer beat both GRU models on translation in both directions, but trailed the recurrent models on character prediction. By validation loss, French → English was the easier direction to optimize (3.950 vs 4.197 averaged across configurations).

Full analysis is in the homework report (PDF).
