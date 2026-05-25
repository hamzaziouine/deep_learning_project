# Ablation Study — Sentiment Classifier

Test set: 13,306 held-out reviews. Same preprocessing, same train/val/test splits, same seed=42.

## Comparison table

| Model | Params | Train config | Test Acc | Macro F1 | Best epoch |
|---|---|---|---|---|---|
| Random baseline | — | — | 0.3374 | — | — |
| Majority class | — | predict POSITIVE | 0.4509 | — | — |
| **DistilBERT-base-uncased** | 67M | bs=16, accum=2, max_len=256, lr=2e-5, fp16, 4 ep | **0.7959** | **0.7617** | 2 |
| **RoBERTa-base** | 125M | bs=8, accum=4, max_len=256, lr=2e-5, fp16, 4 ep | **0.8083** | **0.7686** | 1 (early-stop) |

**Delta RoBERTa vs DistilBERT: +1.24 pp accuracy, +0.0069 macro-F1.**

## Per-class F1

| Class | DistilBERT | RoBERTa | Delta |
|---|---|---|---|
| NEGATIVE | 0.8082 | (see `docs/evaluation_roberta.md`) | — |
| NEUTRAL | 0.5765 | (see `docs/evaluation_roberta.md`) | — |
| POSITIVE | 0.9004 | (see `docs/evaluation_roberta.md`) | — |

## Observations

1. RoBERTa's better contextual representations help on the boundary cases between NEUTRAL and the polar classes — but the gain is bounded by inherent label ambiguity in 3-star reviews (human inter-annotator agreement ~60-70% in literature).
2. RoBERTa converged in 1 epoch (early-stopping fired after epoch 3 with patience=2). DistilBERT peaked at epoch 2. RoBERTa's pre-training is stronger, so less fine-tuning is needed.
3. Both models hit a wall around 80-81% on this 3-class formulation. Per literature: DistilBERT ceiling ~81-82%, RoBERTa-base ceiling ~83-85%, DeBERTa-v3 ~85-87%.
4. The 5 percentage points to 85% target are not free — they require either (a) class-weighted loss (highest single-line ROI, +3-5pp expected on macro-F1), (b) star-rating prefix preprocessing (+1-2 NEUTRAL F1), (c) DeBERTa-v3 architecture, or (d) ensemble of seeds.

## Files

- DistilBERT eval: `docs/evaluation_distilbert.md` + `report/figures/*_distilbert.png`
- RoBERTa eval: `docs/evaluation_roberta.md` + `report/figures/*_roberta.png`
- Models on disk (gitignored): `models/sentiment_distilbert/`, `models/sentiment_roberta/`
