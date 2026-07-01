# ECGR 4106/5106 — Homework 3

Sequence-to-sequence English/French machine translation with GRU encoder-decoder models, in PyTorch.

## Contents

- `HW3_Seq2Seq_Translation.ipynb` — full notebook (all three problems, training, plots, and attention maps)
- `vast_english_french.txt` — dataset (555 English-French sentence pairs)

## Overview

- **Problem 1:** Baseline GRU encoder-decoder, English to French. Train/validation loss curves, exact-match sequence accuracy, corpus BLEU-4, and qualitative samples.
- **Problem 2:** Same model with Bahdanau-style attention. Loss curves, both metrics, comparison to Problem 1, and attention-map visualizations.
- **Problem 3:** Reversed direction (French to English) with both models, loss curves, both metrics, qualitative samples, and a synthesis comparing the two directions.

## Setup

Trained on Google Colab (T4 GPU), seed 42. Word-level translation. Each model is an embedding + GRU encoder with a GRU decoder (baseline) or a Bahdanau-attention decoder. Adam optimizer (lr 1e-3), cross-entropy (NLL) loss, teacher-forcing ratio 0.5, hidden size 256, dropout 0.1 in the attention decoder, 50 epochs, batch size 1. The dataset is split 80/20 (444 train / 111 validation) once and reused across all problems. BLEU-4 is corpus-level with NLTK method-4 smoothing; sequence accuracy is exact word-for-word match.

## Results (validation)

| Model | Direction | BLEU-4 | Seq accuracy | Final val loss |
|---|---|---|---|---|
| Baseline | EN → FR | 0.150 | 0.00% | 7.075 |
| Attention | EN → FR | 0.151 | 0.00% | 5.387 |
| Baseline | FR → EN | 0.133 | 0.00% | 6.040 |
| Attention | FR → EN | 0.109 | 0.00% | 4.867 |

Attention lowered validation loss in both directions, and French-to-English reached the lowest validation loss overall. On only 555 pairs the models overfit and exact-match stays at 0%, so the qualitative translations are the clearest evidence the models learned to translate.

Full analysis is in the homework report (PDF).
