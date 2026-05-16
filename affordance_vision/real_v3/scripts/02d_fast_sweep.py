"""Step 2d (FAST SWEEP): re-evaluate every backbone with the DeepHead and
clean cached features. Phase-2 joint FI fine-tune was empirically worse
on this dataset (see 02c_run.log) so we skip it. We do, however, train
TWO heads for the FFAL slot:

  - "ffal_deep" : same as DINOv2 + DeepHead but trained *with* online
                  FI loss across 4 views on cached anchor features.
                  (FI computed via stored augmented-view features
                  precomputed once.)
  - control: dinov2 + DeepHead, no FI.

Everything uses the v3 dataset, same object split. Reports a single JSON.

Why this design:
  * Speed: cached features ⇒ each epoch is ~0.04 s of head work.
  * Honesty: keeps the FI regularizer as the only thing that
    differentiates FFAL from DINOv2 (FFAL's defining property).
  * Stability: no on-the-fly backbone forward in the BCE term, which
    we showed in 02b destabilized DINOv2.
"""
from __future__ import annotations
import os, sys, json, time, copy, argparse, random, math
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
from torch.utils.data import DataLoader

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_baselines", Path(__file__).resolve().parent / "02_train_and_baselines.py")
_bl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bl)

AFFORDANCES = _bl.AFFORDANCES
AffordanceDataset = _bl.AffordanceDataset
build_backbone = _bl.build_backbone
make_splits = _bl.make_splits
compute_pos_weight = _bl.compute_pos_weight
macro_auroc = _bl.macro_auroc
evaluate_with_stability = _bl.evaluate_with_stability
aug_transform_factory = _bl.aug_transform_factory


class DeepHead(nn.Module):
    def __init__(self, in_dim, n_out=len(AFFORDANCES), p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(64, n_out),
        )
    def forward(self, z): return self.net(z)


@torch.no_grad()
def precompute_features(backbone, extract_fn, loader, device):
    feats, ys = [], []
    backbone.eval()
    for x, y, _ in loader:
        x = x.to(device)
        z = extract_fn(backbone, x).detach().float().cpu()
        feats.append(z); ys.append(y)
    return torch.cat(feats), torch.cat(ys)


@torch.no_grad()
def precompute_view_features(backbone, extract_fn, rows, base_tx, aug_tx, n_views, device,
                             log=None):
    """For each row, precompute features for n_views augmented views once.
    Reused as a fixed FI anchor across all epochs (fast, deterministic-ish)."""
    ds = AffordanceDataset(rows, base_tx, augment=True, aug_transform=aug_tx, n_augs=n_views)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    view_feats = []
    backbone.eval()
    for i, (x, y, views) in enumerate(loader):
        B, V, C, H, W = views.shape
        vx = views.view(B*V, C, H, W).to(device)
        zv = extract_fn(backbone, vx).float().cpu()
        view_feats.append(zv.view(B, V, -1))
        if log and (i % 5 == 0):
            log(f"      precomputed views batch {i}/{len(loader)}")
    return torch.cat(view_feats)  # [N, V, D]


def train_head(feat_tr, y_tr, feat_va, y_va, in_dim, device,
               epochs=80, batch_size=64, lr=1e-3, wd=1e-4, patience=15,
               fi_views=None, lambda_fi=0.0, log=None):
    """Train DeepHead on cached features. If fi_views is provided ([N,V,D]),
    add a head-output-FI loss: encourage logits(anchor) and logits(views)
    to be close in cosine sense."""
    head = DeepHead(in_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos = y_tr.numpy().sum(0); neg = y_tr.shape[0] - pos
    w = np.clip(neg / np.clip(pos, 1, None), 0.5, 10.0)
    pos_w = torch.tensor(w, dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    n = feat_tr.size(0)
    best = dict(val_auroc=-1, state=None, epoch=-1); bad = 0
    history = []
    for epoch in range(epochs):
        head.train()
        idx = torch.randperm(n); losses = []; fis = []
        for i in range(0, n, batch_size):
            j = idx[i:i+batch_size]
            z = feat_tr[j].to(device)
            logits = head(z)
            loss_bce = loss_fn(logits, y_tr[j].to(device))
            if fi_views is not None and lambda_fi > 0:
                vs = fi_views[j].to(device)  # [B, V, D]
                B, V, D = vs.shape
                # head expects 2-D, batch over B*V
                v_logits = head(vs.reshape(B*V, D)).reshape(B, V, -1)
                # FI on probabilities (output-level invariance)
                p_a = torch.sigmoid(logits).unsqueeze(1)            # [B,1,K]
                p_v = torch.sigmoid(v_logits)                       # [B,V,K]
                loss_fi = (p_v - p_a).abs().mean()
                loss = loss_bce + lambda_fi * loss_fi
                fis.append(float(loss_fi.detach()))
            else:
                loss = loss_bce
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss_bce.detach()))
        sched.step()
        head.eval()
        with torch.no_grad():
            vl = head(feat_va.to(device)).cpu()
        val_auroc = macro_auroc(vl, y_va)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
                            train_fi=float(np.mean(fis)) if fis else 0.0,
                            val_auroc=val_auroc))
        if log and epoch % 5 == 0:
            log(f"      e{epoch:03d} loss={np.mean(losses):.4f} "
                f"fi={(np.mean(fis) if fis else 0):.4f} val={val_auroc:.4f}")
        if val_auroc > best["val_auroc"]:
            best = dict(val_auroc=val_auroc, state=copy.deepcopy(head.state_dict()),
                        epoch=epoch); bad = 0
        else:
            bad += 1
            if bad >= patience:
                if log: log(f"      early stop at {epoch}")
                break
    head.load_state_dict(best["state"])
    return head, best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["resnet50", "vit_b16", "clip_b32", "dinov2", "ffal"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lambda-fi", type=float, default=1.0)
    ap.add_argument("--n-views-fi", type=int, default=4)
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    log_path = LOGS / "02d_fast_sweep.log"
    log_fp = open(log_path, "w")
    def log(msg):
        print(msg, flush=True); log_fp.write(msg + "\n"); log_fp.flush()

    log_section(f"FAST SWEEP device={device} backbones={args.backbones}")

    rows = list(jsonl_iter(DATA / "affordances_real.jsonl"))
    tr_rows, va_rows, te_rows, split_info = make_splits(rows)
    log(f"split: train={len(tr_rows)} val={len(va_rows)} test={len(te_rows)}")

    all_results = {}; total_t0 = time.time()
    for name in args.backbones:
        log_section(f"Backbone: {name} (DeepHead)")
        with timed(f"setup-{name}"):
            backbone, base_tx, feat_dim, extract = build_backbone(name, device)
            for p in backbone.parameters(): p.requires_grad_(False)
        aug_tx = aug_transform_factory(base_tx)

        # precompute clean features for everyone
        ds_tr = AffordanceDataset(tr_rows, base_tx)
        ds_va = AffordanceDataset(va_rows, base_tx)
        ds_te = AffordanceDataset(te_rows, base_tx)
        with timed(f"precompute-clean-{name}"):
            tl = DataLoader(ds_tr, batch_size=64, shuffle=False, num_workers=0)
            vl = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=0)
            feat_tr, y_tr = precompute_features(backbone, extract, tl, device)
            feat_va, y_va = precompute_features(backbone, extract, vl, device)

        fi_views = None; lambda_fi = 0.0
        if name == "ffal":
            # Precompute V augmented-view features ONCE, then reuse every epoch.
            # This is the FFAL bit: the head learns to be output-invariant across
            # these fixed augmented-view features.
            with timed(f"precompute-views-{name}"):
                fi_views = precompute_view_features(
                    backbone, extract, tr_rows, base_tx, aug_tx,
                    n_views=args.n_views_fi, device=device, log=log)
            lambda_fi = args.lambda_fi
            log(f"  fi_views shape={tuple(fi_views.shape)}  lambda_fi={lambda_fi}")

        t0 = time.time()
        head, best, history = train_head(
            feat_tr, y_tr, feat_va, y_va, feat_dim, device,
            epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            patience=args.patience, fi_views=fi_views, lambda_fi=lambda_fi, log=log)
        train_time = time.time() - t0
        log(f"  best val={best['val_auroc']:.4f} at epoch {best['epoch']} "
            f"in {train_time:.1f}s")

        torch.save(dict(state_dict=head.state_dict(), feat_dim=feat_dim,
                        backbone=name, head_arch="deep_in-256-128-64-K_BN_Dropout"),
                   MODELS / f"head_{name}_v2.pt")

        with timed(f"test+stability-{name}"):
            res = evaluate_with_stability(name, backbone, extract, head, ds_te, base_tx,
                                          aug_tx, device, n_views=6, log=log)
        res["best_val_auroc"] = best["val_auroc"]
        res["best_epoch"] = best["epoch"]
        res["train_time_s"] = train_time
        res["train_history"] = history
        if name == "ffal":
            res["uses_fi"] = True
            res["lambda_fi"] = lambda_fi
            res["n_views_fi"] = args.n_views_fi
        all_results[name] = res
        log(f"  TEST {name}: AUROC={res['test_macro_auroc']:.4f} "
            f"F1={res['test_macro_f1']:.4f} latFIS={res['latent_fis']:.4f} "
            f"outFIS={res['output_fis']:.4f} lat={res['mean_latency_ms']:.1f}ms")

        del backbone, head
        if device.type == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    total_time = time.time() - total_t0
    out = dict(
        run="enhanced_v2_deephead", device=str(device), seed=SEED,
        n_rows=len(rows), split_sizes=dict(train=len(tr_rows), val=len(va_rows), test=len(te_rows)),
        split_objects=split_info, affordances=AFFORDANCES,
        config=dict(epochs=args.epochs, batch=args.batch, lr=args.lr,
                    patience=args.patience, lambda_fi=args.lambda_fi,
                    n_views_fi=args.n_views_fi,
                    head="DeepHead in→256→128→64→K BN+ReLU+Dropout(0.2)"),
        results=all_results, total_train_time_s=total_time,
        torch_version=torch.__version__,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    out_json = RESULTS / "baselines_enhanced.json"
    with open(out_json, "w") as f: json.dump(out, f, indent=2)
    log(f"\nWrote {out_json}")
    write_report(out)


def write_report(out):
    md = []
    md.append("# Enhanced Baseline Comparison — REAL run (DeepHead + FI for FFAL)\n")
    md.append(f"- Device: `{out['device']}`  | torch `{out['torch_version']}`")
    md.append(f"- N samples: **{out['n_rows']}**  | split: "
              f"train={out['split_sizes']['train']} val={out['split_sizes']['val']} "
              f"test={out['split_sizes']['test']}")
    md.append(f"- Config: {out['config']}\n")
    md.append("## Headline results (enhanced)\n")
    md.append("| Model | Macro AUROC | Macro F1 | Latent FIS | Output FIS | Latency (ms/img) | Train (s) |")
    md.append("|---|---|---|---|---|---|---|")
    for name, r in out["results"].items():
        md.append(f"| {name} | {r['test_macro_auroc']:.4f} | {r['test_macro_f1']:.4f} | "
                  f"{r['latent_fis']:.4f} | **{r['output_fis']:.4f}** | "
                  f"{r['mean_latency_ms']:.1f} | {r.get('train_time_s',0):.1f} |")
    md.append("\n## Per-affordance AUROC\n")
    md.append("| Model | " + " | ".join(AFFORDANCES) + " |")
    md.append("|---|" + "|".join(["---"]*len(AFFORDANCES)) + "|")
    for name, r in out["results"].items():
        md.append("| " + name + " | " +
                  " | ".join(f"{r['per_affordance_auroc'][a]:.3f}" for a in AFFORDANCES) + " |")
    md.append(f"\nTotal training time: **{out['total_train_time_s']:.1f}s**.")
    md.append("\n_Generated by `02d_fast_sweep.py`. Seed = "
              f"{out['seed']}. Reproduce with `python3 scripts/02d_fast_sweep.py`._\n")
    (RESULTS / "baselines_enhanced_report.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
