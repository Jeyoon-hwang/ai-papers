# FFAL v3 — Functional-Feature Affordance Learning on Real MuJoCo Physics
**Replicable, end-to-end real-data version.** No hand-typed metrics.

_Compiled 2026-05-16T21:33:44._

**Author:** 황제영 (Cheonjae @ Mac mini)

**Repository:** `ai-papers-repo/affordance_vision/real_v3/`

---

## Abstract

We collect a 1,020-sample affordance dataset entirely from real MuJoCo rigid-body simulations (no mock data, no hand-typed labels) and benchmark five vision backbones on the same train/val/test object split. For our ablation backbone (DINOv2 ViT-B/14 + a Functional-Invariance regularizer, denoted *FFAL*), we report test macro-AUROC = **0.8483**, output-stability FIS = **0.9610**, mean latency = **11.9 ms / image** on Apple-Silicon MPS. Crucially, **FFAL does not beat the strongest baseline (DINOv2 + plain head, 0.8632)** on macro-AUROC under this regime — and we report that openly, replacing the inflated numbers of v1/v2. We also patch the four logical holes the v2 paper acknowledged but had not actually fixed.

## 1. The four logical holes (v2 → v3 status)

| # | v2 admission | v3 status |
|---|---|---|
| 1 | "Zero-cost" annotation is marketing | Re-framed as **Automated physics-based supervision**. Wall-clock measured: 1.2 min for 1,020 samples on this Mac mini (see §2.4). |
| 2 | Baseline numbers were guessed | Every backbone re-run on the same images. Numbers in §3 come from `baselines_real.json`. |
| 3 | LLM context claim ungrounded | Real FFAL outputs feed a deterministic rule grader (§4). Top-1 = 0.5471, top-3 = 0.3588, κ = 0.1985. |
| 4 | FIS = 92% was latent-only | We report **both** latent FIS and output FIS for **all** five models. See §3. |

## 2. Method

### 2.1 Dataset (real, not mock)

30 procedurally-generated MuJoCo objects (5 shapes × 4 surface types × randomized size/density/friction). For each object we run 34 episodes × 5 affordance probes:

- **sittable**: drop a 70 kg surrogate, measure tilt; label=1 iff tilt<12° and top area>0.015 m²
- **stackable**: drop a 0.04 m cube, measure lateral drift
- **graspable**: simulate parallel-jaw closure under 80 N, measure object slip
- **pushable**: apply 10 N for 1 s, measure displacement
- **pourable**: spawn 30 particles, count fraction retained

Total: **1020 samples**, **1020 unique 224×224 renders** with per-episode camera/light jitter.

Class balance (positive fraction):

| Affordance | Positive | Negative | Pos rate |
|---|---|---|---|
| sittable | 350 | 670 | 34.31% |
| stackable | 405 | 615 | 39.71% |
| graspable | 408 | 612 | 40.00% |
| pushable | 578 | 442 | 56.67% |
| pourable | 102 | 918 | 10.00% |

### 2.2 Splits (no object leakage)

Split by `obj_id`: **714** train / **136** val / **170** test (21 / 4 / 5 objects). Test object IDs: `obj_09`, `obj_11`, `obj_12`, `obj_23`, `obj_26`.

### 2.3 Models compared

| Backbone | Source | Trainable params | Feature dim |
|---|---|---|---|
| ResNet50 | torchvision IMAGENET1K_V2 | head only (~525 K) | 2048 |
| ViT-B/16 | torchvision SWAG_E2E_V1 | head only | 768 |
| CLIP ViT-B/32 | open_clip OpenAI | head only | 512 |
| DINOv2 ViT-B/14 | timm `vit_base_patch14_dinov2.lvd142m` | head only | 768 |
| **FFAL (ours)** | DINOv2 + same head + **FI regularizer** (λ=0.5, 4 augmented views per sample) | head only | 768 |

### 2.4 Honest cost of "automated" supervision

Replacing the prior "Zero-cost" framing:

- **One-time engineering**: ~10 person-hours to design probes and MJCF templates (this entire `real_v3/` directory).
- **Per-run wall-clock** (Mac mini, MPS):
  - Data collection (1,020 samples + 1,020 renders): **1.2 min**.
  - Train 4 baseline heads + extract features: ~4 min total.
  - Train FFAL head (15 epochs, FI reg, MPS): ~12 min.
  - Stability + LLM grounding: ~1 min.
- **Marginal cost per new object archetype**: ~2.4 s of physics + 0.1 s rendering. Break-even vs. paid manual annotation kicks in around N ≈ 200 objects under standard industry rates; we run at N = 30 for proof of concept and document this honestly.

## 3. Results

### 3.1 Head-line numbers (real forward passes, real eval)

| Model | Macro AUROC | Macro F1 | Latent FIS | Output FIS | Latency (ms/img) |
|---|---|---|---|---|---|
| resnet50 | 0.8234 | 0.5726 | 0.7934 | 0.9406 | 4.9 |
| vit_b16 | 0.8230 | 0.5995 | 0.9006 | 0.9689 | 41.1 |
| clip_b32 | 0.8543 | 0.6391 | 0.9454 | 0.9626 | 11.9 |
| dinov2 | 0.8632 | 0.5779 | 0.8823 | 0.9669 | 12.0 |
| ffal | 0.8483 | 0.5564 | 0.8882 | 0.9610 | 11.9 |

### 3.2 What this means (honest interpretation)

- **Best macro-AUROC: `dinov2` at 0.8632**. FFAL is competitive but **does not** beat plain DINOv2 on this dataset and at this training budget. We retract any prior claim of a +10 pp advantage.
- **Output FIS gap closes the v2 "math illusion" hole**: every model reports both metrics. Output FIS ranges from 0.941 (ResNet50) to 0.969 (ViT-B/16). Latent FIS is *not* a reliable proxy for downstream output stability — confirmed empirically.
- **Per-affordance AUROC** (next table) shows where each backbone wins/loses. `pourable` is near-perfect for all models because hollow-top objects are visually obvious; `graspable` is the hardest (best 0.747 by ResNet50). Reporting per-affordance honestly prevents the v2 macro-score from hiding weak categories.

### 3.3 Per-affordance AUROC

| Model | sittable | stackable | graspable | pushable | pourable |
|---|---|---|---|---|---|
| resnet50 | 0.934 | 0.749 | 0.747 | 0.687 | 1.000 |
| vit_b16 | 0.896 | 0.781 | 0.563 | 0.880 | 0.995 |
| clip_b32 | 0.924 | 0.803 | 0.633 | 0.914 | 0.997 |
| dinov2 | 0.947 | 0.799 | 0.691 | 0.883 | 0.997 |
| ffal | 0.930 | 0.807 | 0.704 | 0.804 | 0.997 |

## 4. Multi-modal LLM context — grounded, not asserted

We sampled **170** test images, formed a deterministic prompt around each FFAL prediction, and asked a *rule grader* (same rule applied to gold labels) to compare predicted top-3 "plausible uses" against the gold top-3.

- Top-1 precision: **0.5471**
- Top-3 precision: **0.3588**
- Top-3 Jaccard:   **0.3118**
- Cohen's κ:        **0.1985**

These numbers are modest because (a) our test set has only 5 held-out objects, and (b) the rule grader is intentionally strict (it expects exact use-string matches). The point is that the metric is **now real**: rerunning `03_llm_context.py` reproduces it exactly. Future work: replace the rule grader with a calibrated LLM judge over a larger held-out set; the hook is in place (`--call-llm` switch).

## 5. Limitations

- Only 30 object archetypes; macro-AUROC variance across the 5-object test set is non-trivial. 
- All affordance labels come from a single physics engine (MuJoCo). Sim-to-real transfer is *not* evaluated in v3; v2 reported a 87.6% transfer figure that we cannot currently reproduce without a real-image testbed and we therefore **drop it** from the abstract.
- FI regularizer was trained for 15 epochs at batch=16 because of MPS memory; a larger budget might invert the FFAL-vs-DINOv2 ranking. We do not claim it would.
- The LLM context grader is rule-based; agreement with a calibrated language model is future work.

## 6. Reproducibility

Every metric in this manuscript is produced by:

```bash
cd ai-papers-repo/affordance_vision/real_v3
python3 scripts/01_collect_mujoco.py            # 1.2 min
python3 scripts/02_train_and_baselines.py        # ~16 min on Mac mini MPS
python3 scripts/03_llm_context.py                # <1 min
python3 scripts/04_compile_paper.py              # regenerates this PDF
```

Seed = 20260516, pinned for numpy, random, and torch. Pipeline tested on torch 2.8.0, MuJoCo 3.3.7, macOS Darwin 25.4 (arm64).

## 7. Appendix A — Per-metric provenance

| Metric in paper | Source file | Field |
|---|---|---|
| All AUROC, F1, FIS, latency | `results/baselines_real.json` | `results.<model>.*` |
| LLM-context numbers | `results/llm_context_real.json` | `top1_precision`, etc. |
| Class balance | `data/affordances_real.jsonl` | per-row `label_*` |
| Object catalog | `data/object_catalog.json` | 30 entries |
| Training history | `results/baselines_real.json` | `results.<model>.train_history` |
