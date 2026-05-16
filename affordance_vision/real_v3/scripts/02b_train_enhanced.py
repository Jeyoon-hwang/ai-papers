"""Step 2b (ENHANCED): Same five backbones, but with

  • Stronger image-level augmentation (rotations ±15°, scale ±10%,
    brightness ±20%, plus an explicit affordance-mask jitter of ±5%
    realized through random erasing — proxy for affordance noise).
  • A deeper affordance head: in_dim → 128 → 64 → 32 → K
    with BatchNorm1d + ReLU + Dropout(0.2) on every hidden layer.
  • Tuned optimization: AdamW(lr=0.01, weight_decay=1e-4) + cosine LR
    schedule, batch=64, epochs=200, early stopping patience=20.
  • For FFAL: same enhancements + λ_FI=0.5 cosine FI regularizer
    across 4 augmented views.

All other invariants (frozen backbone, object-level split, AUROC eval,
FIS stability test) are unchanged from 02_train_and_baselines.py so the
enhanced numbers are directly comparable to the baseline JSON.

Outputs:
  - models/head_<backbone>_enhanced.pt
  - results/baselines_enhanced.json
  - results/baselines_enhanced_report.md
"""
from __future__ import annotations
import os, sys, json, math, time, argparse, random, copy
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    ROOT, DATA, IMAGES, MODELS, RESULTS, LOGS,
    seed_everything, get_device, jsonl_iter, log_section, timed, SEED,
)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Reuse: backbones, splits, metrics, dataset, FFAL pieces.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_baselines",
    Path(__file__).resolve().parent / "02_train_and_baselines.py",
)
_bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bl)

AFFORDANCES = _bl.AFFORDANCES
AffordanceDataset = _bl.AffordanceDataset
build_backbone = _bl.build_backbone
make_splits = _bl.make_splits
compute_pos_weight = _bl.compute_pos_weight
macro_auroc = _bl.macro_auroc
evaluate_with_stability = _bl.evaluate_with_stability


# ---------------------------------------------------------------------------
# Enhanced augmentation: rot ±15°, scale ±10%, brightness ±20%,
# plus RandomErasing as a stand-in for "affordance-mask noise ±5%".
# (Applied AFTER the backbone-specific base transform so normalization holds.)
# ---------------------------------------------------------------------------

def enhanced_aug_factory(base_tx):
    """Image-level training augmentation. Applied on PIL before base_tx,
    and a tiny tensor-level erasing afterwards."""
    pil = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10, hue=0.03),
    ])
    return T.Compose([
        pil,
        base_tx,                              # ToTensor + normalize
        T.RandomErasing(p=0.5, scale=(0.02, 0.05), ratio=(0.5, 2.0), value=0),
    ])


def fi_view_aug_factory(base_tx):
    """Augmentations used to generate the V views for FFAL's FI loss.
    Slightly stronger than the train aug; this is what the FI loss tries
    to make invariant."""
    pil = T.Compose([
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        T.RandomResizedCrop(224, scale=(0.80, 1.0)),
    ])
    return T.Compose([pil, base_tx])


# ---------------------------------------------------------------------------
# Enhanced head: in_dim → 128 → 64 → 32 → K  with BN + ReLU + Dropout
# ---------------------------------------------------------------------------

class EnhancedHead(nn.Module):
    def __init__(self, in_dim, n_out=len(AFFORDANCES), p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(32, n_out),
        )
    def forward(self, z): return self.net(z)


# ---------------------------------------------------------------------------
# Training: with on-the-fly augmentation through the frozen backbone.
# We do NOT precompute features for the enhanced run because every epoch
# wants a different random augmentation. Backbone stays frozen → only the
# head receives gradients, and we use torch.no_grad() around the backbone
# to save memory.
# ---------------------------------------------------------------------------

def train_head_online(backbone, extract_fn, ds_tr_aug, ds_va, in_dim, device,
                      epochs=200, batch_size=64, lr=0.01, weight_decay=1e-4,
                      patience=20, log=None):
    head = EnhancedHead(in_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    pos = compute_pos_weight(ds_tr_aug).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos)

    tr_loader = DataLoader(ds_tr_aug, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    best = dict(val_auroc=-1.0, state=None, epoch=-1)
    bad = 0
    history = []
    for epoch in range(epochs):
        head.train(); backbone.eval()
        losses = []
        for x, y, _ in tr_loader:
            x = x.to(device); y = y.to(device)
            with torch.no_grad():
                z = extract_fn(backbone, x).float()
            logits = head(z)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        sched.step()
        # val
        head.eval()
        all_l, all_y = [], []
        with torch.no_grad():
            for x, y, _ in va_loader:
                x = x.to(device)
                z = extract_fn(backbone, x).float()
                all_l.append(head(z).cpu()); all_y.append(y)
        val_auroc = macro_auroc(torch.cat(all_l), torch.cat(all_y))
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
                            val_auroc=val_auroc, lr=opt.param_groups[0]["lr"]))
        if log: log(f"      epoch {epoch:03d} loss={np.mean(losses):.4f} "
                    f"val_AUROC={val_auroc:.4f} lr={opt.param_groups[0]['lr']:.5f}")
        if val_auroc > best["val_auroc"]:
            best = dict(val_auroc=val_auroc, state=copy.deepcopy(head.state_dict()),
                        epoch=epoch)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if log: log(f"      early stop at epoch {epoch}")
                break
    head.load_state_dict(best["state"])
    return head, best, history


def train_ffal_enhanced(backbone, extract_fn, ds_tr_aug, ds_va, fi_aug_tx, in_dim, device,
                        epochs=200, batch_size=64, lr=0.01, weight_decay=1e-4,
                        patience=20, lambda_fi=0.5, n_views=4, log=None):
    """FFAL enhanced: same EnhancedHead + same optimizer, plus FI cosine loss
    across n_views additional augmented views per image. Backbone frozen."""
    head = EnhancedHead(in_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    pos = compute_pos_weight(ds_tr_aug).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos)

    # FI-view dataset: same rows, returns x + n_views additional views per item
    ds_fi = AffordanceDataset(ds_tr_aug.rows, ds_tr_aug.transform,
                              augment=True, aug_transform=fi_aug_tx, n_augs=n_views)
    tr_loader = DataLoader(ds_fi, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    best = dict(val_auroc=-1.0, state=None, epoch=-1)
    bad = 0
    history = []
    for epoch in range(epochs):
        head.train(); backbone.eval()
        losses = []; losses_fi = []
        for x, y, views in tr_loader:
            x = x.to(device); y = y.to(device)
            with torch.no_grad():
                z = extract_fn(backbone, x).float()
            logits = head(z)
            loss_bce = loss_fn(logits, y)
            loss_fi = torch.tensor(0.0, device=device)
            if views.numel() > 0 and lambda_fi > 0:
                B, V, C, H, W = views.shape
                vx = views.view(B*V, C, H, W).to(device)
                with torch.no_grad():
                    zv = extract_fn(backbone, vx).float()
                zv = zv.view(B, V, -1)
                z_norm = F.normalize(z, dim=-1).unsqueeze(1)
                zv_norm = F.normalize(zv, dim=-1)
                cos = (z_norm * zv_norm).sum(-1)
                loss_fi = (1.0 - cos).mean()
            loss = loss_bce + lambda_fi * loss_fi
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss_bce)); losses_fi.append(float(loss_fi))
        sched.step()
        head.eval()
        all_l, all_y = [], []
        with torch.no_grad():
            for x, y, _ in va_loader:
                x = x.to(device)
                z = extract_fn(backbone, x).float()
                all_l.append(head(z).cpu()); all_y.append(y)
        val_auroc = macro_auroc(torch.cat(all_l), torch.cat(all_y))
        history.append(dict(epoch=epoch, train_loss_bce=float(np.mean(losses)),
                            train_loss_fi=float(np.mean(losses_fi)),
                            val_auroc=val_auroc, lr=opt.param_groups[0]["lr"]))
        if log: log(f"      epoch {epoch:03d} bce={np.mean(losses):.4f} "
                    f"fi={np.mean(losses_fi):.4f} val_AUROC={val_auroc:.4f}")
        if val_auroc > best["val_auroc"]:
            best = dict(val_auroc=val_auroc, state=copy.deepcopy(head.state_dict()),
                        epoch=epoch)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if log: log(f"      early stop at epoch {epoch}")
                break
    head.load_state_dict(best["state"])
    return head, best, history


# ---------------------------------------------------------------------------
# Patch evaluate_with_stability to accept our EnhancedHead instances
# (it already does — it only uses head(z), so no change needed).
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["resnet50", "vit_b16", "clip_b32", "dinov2", "ffal"])
    # Defaults: lr=1e-3 (lr=0.01 was too noisy for the small head on frozen
    # backbones; we keep 1e-3 with cosine schedule and stronger augmentation +
    # deeper head as the actual sources of the score gain).
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lambda-fi", type=float, default=0.5)
    ap.add_argument("--n-views-fi", type=int, default=4)
    ap.add_argument("--n-views-stability", type=int, default=6)
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    log_path = LOGS / "02b_train_enhanced.log"
    log_fp = open(log_path, "w")
    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n"); log_fp.flush()

    log_section(f"ENHANCED RUN  device={device}  backbones={args.backbones}")
    log(f"epochs={args.epochs} batch={args.batch} lr={args.lr} "
        f"wd={args.weight_decay} patience={args.patience} "
        f"lambda_fi={args.lambda_fi} n_views_fi={args.n_views_fi}")

    rows = list(jsonl_iter(DATA / "affordances_real.jsonl"))
    log(f"loaded {len(rows)} rows")
    tr_rows, va_rows, te_rows, split_info = make_splits(rows)
    log(f"split: train={len(tr_rows)} val={len(va_rows)} test={len(te_rows)}")

    all_results = {}
    t_global = time.time()
    for name in args.backbones:
        log_section(f"Backbone: {name} (enhanced)")
        with timed(f"backbone-{name} setup"):
            backbone, base_tx, feat_dim, extract = build_backbone(name, device)
            for p in backbone.parameters():
                p.requires_grad_(False)

        # Two augmentation pipelines:
        # 1) train_tx: applied to every training example (the 3x effective
        #    expansion: random rotations / scale / brightness / erasing
        #    means each epoch sees a fresh perturbation).
        train_tx = enhanced_aug_factory(base_tx)
        fi_aug_tx = fi_view_aug_factory(base_tx)
        # Val/test use the clean backbone-default transform.
        ds_tr = AffordanceDataset(tr_rows, train_tx)
        ds_va = AffordanceDataset(va_rows, base_tx)
        ds_te = AffordanceDataset(te_rows, base_tx)

        t0 = time.time()
        if name == "ffal":
            with timed("ffal-enhanced training"):
                head, best, history = train_ffal_enhanced(
                    backbone, extract, ds_tr, ds_va, fi_aug_tx, feat_dim, device,
                    epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                    weight_decay=args.weight_decay, patience=args.patience,
                    lambda_fi=args.lambda_fi, n_views=args.n_views_fi, log=log)
        else:
            with timed(f"head-enhanced training ({name})"):
                head, best, history = train_head_online(
                    backbone, extract, ds_tr, ds_va, feat_dim, device,
                    epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                    weight_decay=args.weight_decay, patience=args.patience, log=log)
        train_time = time.time() - t0

        torch.save(dict(state_dict=head.state_dict(), feat_dim=feat_dim,
                        backbone=name, head_arch="enhanced_4layer_bn_dropout"),
                   MODELS / f"head_{name}_enhanced.pt")

        # Stability eval uses the same FI-aug as v3 baseline (kept identical
        # for direct comparability of the FIS column).
        with timed(f"test+stability ({name} enhanced)"):
            res = evaluate_with_stability(
                name, backbone, extract, head, ds_te, base_tx,
                _bl.aug_transform_factory(base_tx),  # identical stability aug
                device, n_views=args.n_views_stability, log=log)
        res["best_val_auroc"] = best["val_auroc"]
        res["best_epoch"] = best["epoch"]
        res["train_time_s"] = train_time
        res["train_history"] = history
        all_results[name] = res
        log(f"  TEST {name} (enhanced): AUROC={res['test_macro_auroc']:.4f} "
            f"F1={res['test_macro_f1']:.4f} latFIS={res['latent_fis']:.4f} "
            f"outFIS={res['output_fis']:.4f} lat={res['mean_latency_ms']:.1f}ms "
            f"train={train_time:.1f}s")

        del backbone, head
        if device.type == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    total_time = time.time() - t_global
    out = dict(
        run="enhanced",
        device=str(device),
        seed=SEED,
        n_rows=len(rows),
        split_sizes=dict(train=len(tr_rows), val=len(va_rows), test=len(te_rows)),
        split_objects=split_info,
        affordances=AFFORDANCES,
        config=dict(
            epochs=args.epochs, batch=args.batch, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            lambda_fi=args.lambda_fi, n_views_fi=args.n_views_fi,
            augmentation="rot15+scale10+bright20+erasing5",
            head="in→128→64→32→K BN+ReLU+Drop(0.2)",
        ),
        results=all_results,
        total_train_time_s=total_time,
        torch_version=torch.__version__,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    out_json = RESULTS / "baselines_enhanced.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nWrote {out_json}")
    write_report(out)


def write_report(out):
    md = []
    md.append("# Enhanced Baseline Comparison — REAL run\n")
    md.append(f"- Device: `{out['device']}`  | torch `{out['torch_version']}`")
    md.append(f"- N samples: **{out['n_rows']}**  | split (by object_id): "
              f"train={out['split_sizes']['train']} val={out['split_sizes']['val']} "
              f"test={out['split_sizes']['test']}")
    cfg = out["config"]
    md.append(f"- Config: {cfg}\n")
    md.append("## Headline results (enhanced)\n")
    md.append("| Model | Macro AUROC | Macro F1 | Latent FIS | **Output FIS** | Latency (ms/img) | Train (s) |")
    md.append("|---|---|---|---|---|---|---|")
    for name, r in out["results"].items():
        md.append(f"| {name} | {r['test_macro_auroc']:.4f} | {r['test_macro_f1']:.4f} | "
                  f"{r['latent_fis']:.4f} | **{r['output_fis']:.4f}** | "
                  f"{r['mean_latency_ms']:.1f} | {r.get('train_time_s', 0):.1f} |")
    md.append("\n## Per-affordance AUROC\n")
    md.append("| Model | " + " | ".join(AFFORDANCES) + " |")
    md.append("|---|" + "|".join(["---"]*len(AFFORDANCES)) + "|")
    for name, r in out["results"].items():
        md.append("| " + name + " | " +
                  " | ".join(f"{r['per_affordance_auroc'][a]:.3f}" for a in AFFORDANCES) + " |")
    md.append(f"\nTotal training time: **{out['total_train_time_s']:.1f}s**.")
    md.append("\n_Generated by `02b_train_enhanced.py`. Seed = "
              f"{out['seed']}. Reproduce with `python3 scripts/02b_train_enhanced.py`._\n")
    (RESULTS / "baselines_enhanced_report.md").write_text("\n".join(md))
    print(f"Wrote {RESULTS / 'baselines_enhanced_report.md'}")


if __name__ == "__main__":
    main()
