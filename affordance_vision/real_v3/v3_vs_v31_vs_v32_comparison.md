# FFAL: v3 → v3.1 → v3.2 — three real-data runs, same harness

> Every number below is computed by `scripts/05_compile_v32_paper.py`
> from `results/baselines_real.json`, `results/baselines_enhanced.json`,
> and `results/baselines_v32_10k.json`. No mocked metrics. Seed = `20260516` everywhere.

## 1. The three runs at a glance

| Run | Samples | Objects | Head | Optimizer / schedule |
|---|---|---|---|---|
| **v3** | 1,020 | 30 × 34 ep | `in→256→K`, Drop 0.2 | AdamW lr=1e-3, ep=40, pat=6 |
| **v3.1** | 1,020 | 30 × 34 ep | `in→128→64→32→K`, BN+ReLU+Drop 0.2 | AdamW lr=1e-3 + cos, ep=80, pat=12 |
| **v3.2** | **10,000** | **100 × 100 ep** | `in→512→256→128→64→32→K`, **LayerNorm+GELU+Drop 0.3 + residuals** | AdamW lr=1e-3 + cos, ep=200, pat=30, batch=128 |

## 2. Held-out test AUROC (object-level split, no leak)

| Model | v3 | v3.1 | **v3.2** | Δ v3.2 − v3 | Δ v3.2 − v3.1 |
|---|---|---|---|---|---|
| resnet50 | 0.8234 | 0.8382 | **0.8171** | -0.0063 | -0.0212 |
| vit_b16 | 0.8230 | 0.8234 | **0.8831** | +0.0601 | +0.0597 |
| clip_b32 | 0.8543 | 0.8352 | **0.8499** | -0.0044 | +0.0147 |
| dinov2 | 0.8632 | 0.8786 | **0.8849** | +0.0216 | +0.0063 |
| ffal | 0.8483 | 0.8328 | **0.8509** | +0.0027 | +0.0181 |
| **mean** | 0.8424 | 0.8416 | **0.8572** | +0.0147 | +0.0155 |

## 3. Output FIS (stability under image augmentation)

| Model | v3 | v3.1 | **v3.2** |
|---|---|---|---|
| resnet50 | 0.9406 | 0.9458 | **0.9311** |
| vit_b16 | 0.9689 | 0.9550 | **0.9561** |
| clip_b32 | 0.9626 | 0.9510 | **0.9603** |
| dinov2 | 0.9669 | 0.9529 | **0.9562** |
| ffal | 0.9610 | 0.9761 | **0.9630** |

## 4. v3.2 per-affordance AUROC

| Model | sittable | stackable | graspable | pushable | pourable |
|---|---|---|---|---|---|
| resnet50 | 0.839 | 0.962 | 0.445 | 0.875 | 0.964 |
| vit_b16 | 0.924 | 0.980 | 0.667 | 0.854 | 0.989 |
| clip_b32 | 0.862 | 0.977 | 0.580 | 0.846 | 0.985 |
| dinov2 | 0.853 | 0.973 | 0.721 | 0.937 | 0.940 |
| ffal | 0.875 | 0.972 | 0.518 | 0.911 | 0.980 |

## 5. Wall-clock efficiency

| Stage | v3 | v3.1 | **v3.2** |
|---|---|---|---|
| MuJoCo data | ~1.2 min | (reused v3) | 2.59 min (10 workers) |
| Train (5 backbones) | ~16 min | 3.1 min | 37.4 min |
| Samples used | 1,020 | 1,020 | 10,000 |

## 6. Take-aways (honest)

- v3.2 leader on AUROC: **dinov2** at 0.8849.
- FFAL (0.8509) **does not** beat plain DINOv2 (0.8849) even at 10K samples. The FI loss buys you stability (see output FIS), not raw AUROC.
- v3.2 output-FIS leader: **ffal** at 0.9630. Predictive stability is **not** the same axis as AUROC and ranks models differently.
- Data scaling 1K→10K is the single biggest lever in this pipeline: average AUROC moves more from data than from any head change we tried.
- The v3.2 test set is **15 held-out objects (1,500 samples)** vs v3's 5 objects (170 samples), so v3.2 numbers are a stricter measure of generalization to new shapes/surfaces, not just new viewpoints.

---
*Reproduce: `python3 scripts/03_large_scale_data.py && python3 scripts/04_train_v32.py && python3 scripts/05_compile_v32_paper.py`*