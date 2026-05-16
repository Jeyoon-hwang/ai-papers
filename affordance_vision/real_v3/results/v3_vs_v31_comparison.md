# v3 → v3.1 Direct Comparison

**Goal of v3.1 run:** Improve FFAL macro-AUROC above DINOv2 0.8632 baseline
via stronger architecture, data augmentation, and training.

**What actually happened:** DeepHead architecture lifts every backbone
(except CLIP). DINOv2 + DeepHead becomes the new accuracy SOTA at **0.8786**.
FFAL's FI regularizer reliably improves output stability (outFIS) but does
*not* improve raw discrimination on this 1,020-sample dataset.

## Headline table

| Model    | v3 AUROC | v3.1 AUROC | Δ AUROC | v3 outFIS | v3.1 outFIS | Δ outFIS |
|---|---|---|---|---|---|---|
| resnet50 | 0.8234 | **0.8382** | +0.0148 | 0.9406 | 0.9458 | +0.0052 |
| vit_b16  | 0.8230 | 0.8234     | +0.0004 | 0.9689 | 0.9550 | −0.0139 |
| clip_b32 | 0.8543 | 0.8352     | −0.0191 | 0.9626 | 0.9510 | −0.0116 |
| **dinov2** | 0.8632 | **0.8786** | **+0.0154** | 0.9669 | 0.9529 | −0.0140 |
| ffal     | 0.8483 | 0.8328     | −0.0155 | 0.9610 | **0.9761** | **+0.0151** |

## λ_FI sweep on FFAL

| λ_FI | Macro AUROC | outFIS  | Interpretation |
|------|------------|---------|---|
| 0.0  | 0.8786     | 0.9529  | DINOv2 + DeepHead, no FI — accuracy SOTA |
| 0.2  | 0.8194     | 0.9735  | mild FI, accuracy already drops below v3 baseline |
| 1.0  | 0.8328     | 0.9761  | strong FI, best outFIS, accuracy still lags DINOv2 |

There is **no λ_FI in this sweep** at which FFAL's macro-AUROC exceeds
plain DINOv2 + DeepHead. The trade-off is real.

## Efficiency: wall-clock

| Run                         | Wall-clock        | Lines of new code | Outcome              |
|---|---|---|---|
| v3 baseline                 | ~16 min           | —                 | DINOv2 0.8632        |
| v3.1 attempt #1 (heavy aug) | aborted >30 min   | +320              | DINOv2 0.7687 (worse)|
| v3.1 attempt #2 (two-phase) | aborted ~10 min   | +280              | regressed            |
| **v3.1 final (DeepHead + cached FI)** | **~3 min** | +290 | DINOv2 0.8786 SOTA |

The successful v3.1 pipeline is **5× faster than v3** and **+1.54 pp more
accurate** on macro-AUROC for the best backbone.

## What worked vs what didn't

**Worked (kept in v3.1):**
1. **DeepHead** (in→256→128→64→K with BN+ReLU+Dropout). +1.5 pp DINOv2
   AUROC. Same arch helped ResNet50 (+1.5 pp).
2. **Cached features + cached augmented views.** Cuts wall-clock from
   tens of minutes to seconds for the head training; same eval semantics
   as v3.
3. **AdamW(lr=1e-3) + cosine** with patience=15. lr=1e-2 was too noisy.

**Did not work (kept as negative results):**
1. **Heavy on-the-fly augmentation** in the BCE branch. Destabilized
   DINOv2 (0.8632 → 0.7687).
2. **Two-phase clean → joint FI fine-tune.** Joint phase regressed
   validation AUROC.
3. **Naïve "more epochs, higher lr"** as suggested in the task brief.
   With 1,020 samples, lr=1e-2 + 200 epochs is unstable; the model
   memorizes and oscillates.

## Bottom line

- New macro-AUROC SOTA on this dataset: **DINOv2 + DeepHead = 0.8786**.
- New outFIS SOTA: **FFAL (DINOv2 + DeepHead + FI λ=1.0) = 0.9761**.
- These two SOTAs are different models because the FI regularizer is a
  stability/accuracy trade-off on this scale of data — we report it as
  such instead of cherry-picking.

_Data source files:_
- `results/baselines_real.json`        (v3)
- `results/baselines_enhanced.json`    (v3.1 final, λ_FI=1.0)
- `results/baselines_enhanced_lambda0.2.json` (λ_FI=0.2 sweep point)
- `results/baselines_enhanced_lambda1.0.json` (backup of λ_FI=1.0 results)
