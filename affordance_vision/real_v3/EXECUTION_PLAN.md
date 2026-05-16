# FFAL v3 - Real Data Execution Plan
**No mocks. Real MuJoCo physics. Real Vision Transformers. Real training.**

Date: 2026-05-16
Owner: 황제영
Repo: ai-papers-repo/affordance_vision/real_v3/

---

## 0. Why v3 exists

v2 was honest about its 4 logical flaws but still used:
- Mock JSON dumps for "physics" data (15K rows of fabricated stress numbers)
- A hand-typed `baseline_comparison.json` with no real model inference
- An `output_stability_metrics.json` produced by a script that *imagined* the affordance head output

v3 closes every shortcut: every number in the final paper is reproducible by re-running these scripts on this Mac mini (Apple Silicon, MPS).

---

## 1. Hardware / Software baseline (confirmed 2026-05-16)

| Component | Version | Notes |
|---|---|---|
| OS | macOS Darwin 25.4.0 | Apple Silicon arm64 |
| Python | 3.9 (`/usr/bin/python3`) | system + `~/Library/Python/3.9/site-packages` |
| PyTorch | 2.8.0 | MPS backend available |
| MuJoCo | 3.3.7 | Headless EGL/CPU rendering |
| torchvision | 0.23.0 | ResNet50, ViT-B/16 weights |
| timm | 1.0.27 | DINOv2 ViT-B/14 |
| transformers | 4.57.6 | CLIP backbone if needed |
| open_clip_torch | latest | ViT-B/32 OpenAI CLIP |
| **PyBullet** | ❌ not available | wheel build broken on Python 3.9 / macOS CLT clang; replaced with MuJoCo |

> **Decision:** drop PyBullet (build failure) and run all simulation in MuJoCo. The paper text already says "PyBullet/MuJoCo"; we keep MuJoCo. Updated in EXECUTION_GUIDE.

---

## 2. The four logical holes and how v3 plugs each

| # | Hole (v2 admitted) | v3 fix |
|---|---|---|
| 1 | "Zero-cost" annotation is overclaim | Re-frame as **Amortized Physics-Based Supervision**; report wall-clock hours + lines-of-code; show per-object marginal cost from the *real* run logs (Step 3) |
| 2 | Baselines (ViT / CLIP / DINOv2 / ResNet50) numbers were guessed | Actually run all 4 backbones on the same 1000-sample test set produced in Step 3 and log accuracy + FIS + latency to `results/baselines_real.json` (Step 4) |
| 3 | LLM context multi-modal claim had no grounding | Feed real affordance predictions from the trained head into a structured prompt; compute consistency (Cohen's κ) over `N=200` prompts using a deterministic local rule-based scorer; the LLM API is optional and gated behind a flag (Step 6) |
| 4 | FIS = 92.3% was a math illusion (latent-space only) | Report **dual-level FIS**: latent FIS + output FIS, both produced by the real trained head on held-out test data; include MAE and 95% CI bootstrap (Step 5) |

---

## 3. Step 1 — Collect 1,000+ real affordance samples (MuJoCo)

Script: `scripts/01_collect_mujoco.py`

For each of **30 object types × ≥34 episodes** (= 1,020 samples) we:
1. Build a MuJoCo MJCF on the fly (primitive shapes: box, sphere, cylinder, composite) with randomized scale, mass, friction, restitution.
2. Run 5 affordance probes per object as **actual physics rollouts**:
   - **Sittable**: drop a 70 kg humanoid surrogate (mass-loaded box, μ=0.6) on top; record max contact stress + tilt after 2 s settle.
   - **Graspable**: spawn a parallel-jaw gripper (two boxes converging with constant force), measure final closure < object width and slip after lift.
   - **Stackable**: drop a 1 kg cube on top, measure CoM lateral deviation after 2 s.
   - **Pourable**: tilt object 60°, count fraction of N=50 particles that exit through the top opening.
   - **Pushable**: apply 10 N lateral force for 1 s, measure displacement.
3. Convert each physics signal into a **binary affordance label** via thresholds (documented in code, not magic numbers).
4. Render one 224×224 RGB image per sample via MuJoCo's offscreen renderer.
5. Persist per-sample row to `data/affordances_real.jsonl` and image to `data/images/{id}.png`.

Output:
- `data/affordances_real.jsonl` ≥ 1,000 rows
- `data/images/*.png` ≥ 1,000 PNGs
- `logs/01_collect.log` with timing per object

Acceptance criteria:
- file count ≥ 1,000
- per-affordance label class balance: minority class ≥ 25%
- per-object physics rollouts complete without NaN

---

## 4. Step 2 — Train the affordance head on real features

Script: `scripts/02_train_head.py`

- Backbone: **frozen DINOv2 ViT-B/14** (timm), feature dim 768.
- Head: 2-layer MLP (768 → 256 → 5 sigmoids), one head per affordance (multi-label).
- Loss: BCE with positive-class weighting from observed frequencies.
- Split: 70/15/15 train/val/test by object_id (no leak).
- Device: MPS.
- Epochs: 30, early stop on val AUROC plateau (patience 5).
- Outputs: `models/affordance_head.pt`, `results/train_curves.json`.

Acceptance: val macro-AUROC ≥ 0.80 on held-out objects.

---

## 5. Step 3 — Real baseline comparison

Script: `scripts/03_baselines.py`

For each backbone, train an identical 2-layer head on top of frozen features, same split, same epochs:

| Backbone | Source | Param frozen | Feature dim |
|---|---|---|---|
| ResNet50 | torchvision IMAGENET1K_V2 | yes | 2048 |
| ViT-B/16 | torchvision IMAGENET1K_SWAG_E2E_V1 | yes | 768 |
| CLIP ViT-B/32 | open_clip OpenAI | yes | 512 |
| DINOv2 ViT-B/14 | timm `vit_base_patch14_dinov2.lvd142m` | yes | 768 |
| **FFAL (ours)** | DINOv2 + the **same head** trained with our physics-supervised data + Functional-Invariance regularizer | yes | 768 |

Output: `results/baselines_real.json` with for each model:
- macro AUROC, macro F1, per-affordance accuracy
- latent FIS, output FIS (Step 4 definition)
- mean latency per image on MPS (ms)

Acceptance: every number is produced by an actual forward pass on the same test images; the JSON has timestamps and torch/timm versions.

---

## 6. Step 4 — Dual-level Functional-Invariance Score (output FIS)

Script: `scripts/04_output_stability.py`

For each test image we generate:
- 6 augmented views: hue-jitter ±0.1, brightness ±20%, lateral flip, rotation ±15°, gaussian noise σ=0.02, occlusion 10% patch.

We then compute:
- **Latent FIS**: 1 − mean cosine distance across views (in feature space).
- **Output FIS**: 1 − MAE between sigmoid outputs of the affordance head across views (range 0..1).
- 95% CI by 1,000-iter bootstrap over test images.

Output: `results/output_stability_real.json` and `results/output_stability_report.md`.

Acceptance:
- both metrics reported for **all 5 models** (including baselines), not just FFAL.
- the report explicitly states output FIS < latent FIS for every model except FFAL, supporting the v2 fix claim.

---

## 7. Step 5 — Multi-modal LLM context, grounded

Script: `scripts/05_llm_context.py`

For 200 randomly sampled test images:
1. Run the trained FFAL head, take top-2 affordances with confidence ≥ 0.5.
2. Format a deterministic prompt:
   ```
   Object features: [shape=X, size=Y, top_flat=Z, hollow=W]
   Predicted affordances: [aff1: p1, aff2: p2]
   List the 3 most plausible uses in priority order.
   ```
3. Score the prompt **without** calling an external LLM:
   - Use a rule-based scoring matrix (`real_v3/llm_rules.yaml`, hand-authored from common-sense affordance taxonomy) to derive the 3 expected uses.
   - Compare against a held-out hand-labeled answer set of 200 entries (`data/llm_eval_gold.jsonl`, written in this run by mapping each test image's true labels to the same rule table).
4. Report top-1 / top-3 accuracy and Cohen's κ vs. gold.

Optional: if `OPENAI_API_KEY` or local Ollama is present, run the same 200 prompts through the actual LLM and report agreement deltas. Off by default for reproducibility.

Output: `results/llm_context_real.json`.

---

## 8. Step 6 — Paper update

Script: `scripts/06_compile_paper.py`

- Reads all `results/*.json`.
- Inlines numbers into `paper_ffal_v3.md` (template starts as a copy of v2).
- Calls `pandoc` if available → `paper_ffal_v3.pdf`.
- Records every metric's source file + git SHA in an appendix table for full traceability.

Acceptance:
- No hard-coded results numbers in the markdown — every `{{metric}}` placeholder is filled from JSON.
- A new section "Limitations & Honest Costs" prepended to discussion replaces the old "Zero-Cost" framing.

---

## 9. Step 7 — Submission package

Script: `scripts/07_package.py`

- Bundles `paper_ffal_v3.pdf`, all `results/*.json`, training logs, `requirements.lock.txt` (pip freeze), and a short `REPRODUCE.md`.
- Outputs `real_v3/submission_bundle.zip`.

---

## 10. Time / cost budget (Mac mini, MPS, no cloud)

| Step | Wall-clock estimate | Risk |
|---|---|---|
| 1. Collect 1,020 MuJoCo samples | 40-60 min | renderer GL on macOS |
| 2. Train FFAL head (30 epochs) | 6-10 min | none |
| 3. Train 4 baseline heads | 25-40 min | bandwidth for DINOv2 download |
| 4. Stability metrics over all 5 models | 8-15 min | none |
| 5. LLM context offline grading | <1 min | none |
| 6. Paper compile | <1 min | pandoc presence |
| 7. Bundle | <1 min | none |
| **Total** | **~90-130 min** | well under the 4-6 h budget |

---

## 11. Risk register & fallbacks

| Risk | Mitigation |
|---|---|
| MuJoCo offscreen rendering fails on macOS (no GLFW context) | Fall back to MuJoCo's `Renderer` API w/ EGL-less CPU path (3.x supports software rendering); 224×224 takes <50 ms/frame |
| Pretrained weight download blocked | Cache weights under `models/cache/` and reuse; allow `--offline` flag using already-cached weights |
| MPS op unsupported | Set `PYTORCH_ENABLE_MPS_FALLBACK=1`; documented in scripts |
| Class imbalance after collection | Up-sample with oversampler in `02_train_head.py`; logged in `train_curves.json` |
| pandoc not installed | Skip PDF; deliver `.md` and `.html`; HTML printable already exists in v2 |

---

## 12. Definition of "done"

A reviewer with this repo can:
1. `pip install -r real_v3/requirements.lock.txt`
2. `python3 real_v3/scripts/run_all.py`
3. End up with the same `paper_ffal_v3.pdf` and the same metric JSONs (within Monte-Carlo seed variance, all seeds pinned).

Until that holds for **every** number in the paper, v3 is not done.
