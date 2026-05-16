"""Step 4 (v3.2): Train all five backbones with a *deeper, residual* head on
the 10,000-sample v3.2 dataset.

Design:
- Same backbone-freezing protocol as v3 (no fine-tuning).
- Same object-level 70/15/15 split (no object leak).
- Same evaluation harness (FIS, AUROC, F1, latency) — imported from v3 so the
  comparison v3 / v3.1 / v3.2 is apples-to-apples.

What changes:
- **DeepResidualHead** replaces the 2- and 4-layer heads from v3 / v3.1:
    in_dim → 512 → 256 → 128 → 64 → 32 → 6
    LayerNorm + GELU + Dropout(0.3) on every hidden layer, with
    residual connections between blocks of matching dimension.
- **All five models share this head** (ResNet50, ViT-B/16, CLIP-B/32,
  DINOv2, FFAL) so the deep-head benefit is not silently confounded with
  the FFAL FI loss.
- FFAL keeps its functional-invariance cosine loss across N augmented views.

Outputs:
  models/head_<name>_v32.pt
  results/baselines_v32_10k.json
  results/baselines_v32_10k_report.md
"""
from __future__ import annotations
import os, sys, json, math, time, argparse, random, copy
from pathlib import Path
import numpy as np

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
import torchvision.transforms as T

# Reuse v3 building blocks
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
compute_pos_weight = _bl.compute_pos_weight
macro_auroc = _bl.macro_auroc
evaluate_with_stability = _bl.evaluate_with_stability
aug_transform_factory = _bl.aug_transform_factory


# ---------------------------------------------------------------------------
# Splits: same 70/15/15 logic as v3, applied to whatever rows you feed in.
# ---------------------------------------------------------------------------

def make_splits_v32(rows, seed=SEED):
    rng = random.Random(seed)
    objs = sorted({r["obj_id"] for r in rows})
    rng.shuffle(objs)
    n = len(objs)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_objs = set(objs[:n_train])
    val_objs = set(objs[n_train:n_train+n_val])
    test_objs = set(objs[n_train+n_val:])
    def split(rows, S): return [r for r in rows if r["obj_id"] in S]
    return split(rows, train_objs), split(rows, val_objs), split(rows, test_objs), \
           dict(train=sorted(train_objs), val=sorted(val_objs), test=sorted(test_objs))


# ---------------------------------------------------------------------------
# Augmentation factories
# ---------------------------------------------------------------------------

def train_aug_factory(base_tx):
    pil = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10, hue=0.03),
    ])
    return T.Compose([
        pil,
        base_tx,
        T.RandomErasing(p=0.5, scale=(0.02, 0.06), ratio=(0.5, 2.0), value=0),
    ])

def fi_view_aug_factory(base_tx):
    pil = T.Compose([
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        T.RandomResizedCrop(224, scale=(0.80, 1.0)),
    ])
    return T.Compose([pil, base_tx])


# ---------------------------------------------------------------------------
# DeepResidualHead: in_dim → 512 → 256 → 128 → 64 → 32 → K
#   - LayerNorm + GELU + Dropout(0.3) on every hidden layer
#   - residual shortcut between blocks of the same hidden width
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """LN → Linear → GELU → Dropout → Linear → +identity, kept at same width."""
    def __init__(self, dim, p_drop=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class DownBlock(nn.Module):
    """LN → Linear (dim_in → dim_out) → GELU → Dropout. No residual (dim changes)."""
    def __init__(self, d_in, d_out, p_drop=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        x = self.ln(x)
        x = F.gelu(self.fc(x))
        x = self.drop(x)
        return x


class DeepResidualHead(nn.Module):
    def __init__(self, in_dim, n_out=len(AFFORDANCES), p_drop=0.3,
                 widths=(512, 256, 128, 64, 32)):
        super().__init__()
        layers = []
        prev = in_dim
        for w in widths:
            layers.append(DownBlock(prev, w, p_drop=p_drop))
            layers.append(ResBlock(w, p_drop=p_drop))
            prev = w
        self.body = nn.Sequential(*layers)
        self.out_ln = nn.LayerNorm(prev)
        self.out = nn.Linear(prev, n_out)
    def forward(self, z):
        h = self.body(z)
        h = self.out_ln(h)
        return self.out(h)


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

def _run_val(head, backbone, extract_fn, va_loader, device):
    head.eval(); backbone.eval()
    all_l, all_y = [], []
    with torch.no_grad():
        for x, y, _ in va_loader:
            x = x.to(device)
            z = extract_fn(backbone, x).float()
            all_l.append(head(z).cpu()); all_y.append(y)
    return macro_auroc(torch.cat(all_l), torch.cat(all_y))


@torch.no_grad()
def _precompute_features(backbone, extract_fn, ds, device, batch_size=128,
                         n_aug_passes=1, log=None, tag=""):
    """Run the (frozen) backbone over ds n_aug_passes times and stack the
    resulting features + labels. With n_aug_passes>1 and ds.transform that
    includes randomness, this gives us augmented features cheaply."""
    backbone.eval()
    feats, ys = [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for pass_i in range(n_aug_passes):
        for x, y, _ in loader:
            x = x.to(device)
            z = extract_fn(backbone, x).detach().float().cpu()
            feats.append(z); ys.append(y)
        if log:
            log(f"      [{tag}] feature precompute pass {pass_i+1}/{n_aug_passes} done")
    return torch.cat(feats), torch.cat(ys)


def train_head_v32(backbone, extract_fn, ds_tr, ds_va, in_dim, device,
                   epochs=500, batch_size=128, lr=1e-3, weight_decay=1e-4,
                   patience=50, n_aug_passes=4, log=None):
    """Fast deep-head training using pre-computed (possibly augmented) features.
    The backbone is forward-passed only n_aug_passes * |train| times, not
    epochs * |train| times — orders of magnitude faster on MPS.
    """
    head = DeepResidualHead(in_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    pos = compute_pos_weight(ds_tr).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos)

    # Precompute features once (with augmented passes for diversity).
    feat_tr, y_tr = _precompute_features(backbone, extract_fn, ds_tr, device,
                                         batch_size=batch_size,
                                         n_aug_passes=n_aug_passes, log=log, tag="train")
    feat_va, y_va = _precompute_features(backbone, extract_fn, ds_va, device,
                                         batch_size=batch_size,
                                         n_aug_passes=1, log=log, tag="val")
    feat_tr = feat_tr.to(device); y_tr = y_tr.to(device)
    feat_va_d = feat_va.to(device)

    n = feat_tr.size(0)
    best = dict(val_auroc=-1.0, state=None, epoch=-1)
    bad = 0
    history = []
    for epoch in range(epochs):
        head.train()
        idx = torch.randperm(n, device=device)
        losses = []
        for i in range(0, n, batch_size):
            j = idx[i:i+batch_size]
            logits = head(feat_tr[j])
            loss = loss_fn(logits, y_tr[j])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(feat_va_d).cpu()
        val_auroc = macro_auroc(val_logits, y_va)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
                            val_auroc=val_auroc, lr=opt.param_groups[0]["lr"]))
        if log and epoch % 20 == 0:
            log(f"      epoch {epoch:03d} loss={np.mean(losses):.4f} "
                f"val_AUROC={val_auroc:.4f} lr={opt.param_groups[0]['lr']:.5f}")
        if val_auroc > best["val_auroc"]:
            best = dict(val_auroc=val_auroc,
                        state=copy.deepcopy(head.state_dict()), epoch=epoch)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if log: log(f"      early stop at epoch {epoch} "
                            f"(best {best['val_auroc']:.4f} @ epoch {best['epoch']})")
                break
    head.load_state_dict(best["state"])
    return head, best, history


def train_ffal_v32(backbone, extract_fn, ds_tr, ds_va, fi_aug_tx, in_dim, device,
                   epochs=500, batch_size=128, lr=1e-3, weight_decay=1e-4,
                   patience=50, lambda_fi=0.5, n_views=4, n_aug_passes=4,
                   p_drop=0.3, log=None):
    """FFAL with the FI loss expressed in feature space.

    To stay on the same fast train track as the other backbones we still
    pre-compute features (n_aug_passes augmented passes for clean BCE), and
    we *additionally* pre-compute n_views FI-augmented feature passes for
    each training sample. FI loss is then applied head-side as
    1 - cosine(head(z_anchor), head(z_view)) — i.e. the head is encouraged
    to map all V views to the same point, which is the spirit of FFAL.
    """
    head = DeepResidualHead(in_dim, p_drop=p_drop).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    pos = compute_pos_weight(ds_tr).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos)

    # Standard augmented training features (multiple passes for diversity)
    feat_tr, y_tr = _precompute_features(backbone, extract_fn, ds_tr, device,
                                         batch_size=batch_size,
                                         n_aug_passes=n_aug_passes, log=log, tag="ffal-train")
    feat_va, y_va = _precompute_features(backbone, extract_fn, ds_va, device,
                                         batch_size=batch_size,
                                         n_aug_passes=1, log=log, tag="ffal-val")
    feat_tr = feat_tr.to(device); y_tr = y_tr.to(device)
    feat_va_d = feat_va.to(device)
    # FI views: same rows, different (stronger) augmentation, K=n_views passes.
    # Each view shares the same label and same anchor index.
    ds_fi_view = AffordanceDataset(ds_tr.rows, fi_aug_tx)
    fi_feats, _ = _precompute_features(backbone, extract_fn, ds_fi_view, device,
                                       batch_size=batch_size,
                                       n_aug_passes=n_views, log=log, tag="ffal-fi-views")
    # fi_feats has shape [n_views * N, D] (one view stacked after another). The
    # n_aug_passes for train features also produced n_aug_passes * N rows.
    # We anchor every n_aug_passes-th slice of feat_tr (the first pass) to its
    # n_views fi views below.
    N = len(ds_tr.rows)
    # Build anchor->fi_views index map: anchor i (0..N-1) uses fi rows [i, i+N, i+2N, ...]
    fi_feats = fi_feats.view(n_views, N, -1).to(device)  # [V, N, D]

    n_total = feat_tr.size(0)              # = n_aug_passes * N
    pass_to_anchor = torch.arange(N, device=device).repeat(n_aug_passes)  # maps each train row to its anchor in 0..N-1

    best = dict(val_auroc=-1.0, state=None, epoch=-1)
    bad = 0
    history = []
    for epoch in range(epochs):
        head.train()
        idx = torch.randperm(n_total, device=device)
        losses_bce, losses_fi = [], []
        for i in range(0, n_total, batch_size):
            j = idx[i:i+batch_size]
            z = feat_tr[j]
            y = y_tr[j]
            logits = head(z)
            loss_bce = loss_fn(logits, y)
            # FI loss: pull head output of anchor toward head output of its V views
            anchor_ids = pass_to_anchor[j]                # [B]
            view_z = fi_feats[:, anchor_ids, :]            # [V, B, D]
            V_, B_, D_ = view_z.shape
            view_logits = head(view_z.view(V_*B_, D_))     # [V*B, K]
            view_logits = view_logits.view(V_, B_, -1)     # [V, B, K]
            anchor_logits = logits.unsqueeze(0)            # [1, B, K]
            l_norm = F.normalize(anchor_logits, dim=-1)
            v_norm = F.normalize(view_logits, dim=-1)
            cos = (l_norm * v_norm).sum(-1)                # [V, B]
            loss_fi = (1.0 - cos).mean()
            loss = loss_bce + lambda_fi * loss_fi
            opt.zero_grad(); loss.backward(); opt.step()
            losses_bce.append(float(loss_bce.detach()))
            losses_fi.append(float(loss_fi.detach()))
        sched.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(feat_va_d).cpu()
        val_auroc = macro_auroc(val_logits, y_va)
        history.append(dict(epoch=epoch,
                            train_loss_bce=float(np.mean(losses_bce)),
                            train_loss_fi=float(np.mean(losses_fi)),
                            val_auroc=val_auroc, lr=opt.param_groups[0]["lr"]))
        if log and epoch % 20 == 0:
            log(f"      epoch {epoch:03d} bce={np.mean(losses_bce):.4f} "
                f"fi={np.mean(losses_fi):.4f} val_AUROC={val_auroc:.4f}")
        if val_auroc > best["val_auroc"]:
            best = dict(val_auroc=val_auroc,
                        state=copy.deepcopy(head.state_dict()), epoch=epoch)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if log: log(f"      early stop at epoch {epoch} "
                            f"(best {best['val_auroc']:.4f} @ epoch {best['epoch']})")
                break
    head.load_state_dict(best["state"])
    return head, best, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["resnet50", "vit_b16", "clip_b32", "dinov2", "ffal"])
    ap.add_argument("--data", type=str, default="v32/affordances_v32.jsonl",
                    help="Path under data/ to the JSONL.")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lambda-fi", type=float, default=0.5)
    ap.add_argument("--n-views-fi", type=int, default=4)
    ap.add_argument("--n-views-stability", type=int, default=6)
    ap.add_argument("--p-drop", type=float, default=0.3)
    ap.add_argument("--n-aug-passes", type=int, default=2)
    ap.add_argument("--out-json", type=str, default="baselines_v32_10k.json")
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    log_path = LOGS / "04_train_v32.log"
    log_fp = open(log_path, "w")
    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n"); log_fp.flush()

    log_section(f"v3.2 training  device={device}  backbones={args.backbones}")
    log(f"epochs={args.epochs} batch={args.batch} lr={args.lr} "
        f"wd={args.weight_decay} patience={args.patience} "
        f"lambda_fi={args.lambda_fi} n_views_fi={args.n_views_fi}")

    data_path = DATA / args.data
    rows = list(jsonl_iter(data_path))
    log(f"loaded {len(rows)} rows from {data_path}")
    tr_rows, va_rows, te_rows, split_info = make_splits_v32(rows)
    log(f"split: train={len(tr_rows)} val={len(va_rows)} test={len(te_rows)} "
        f"objects: train={len(split_info['train'])} val={len(split_info['val'])} test={len(split_info['test'])}")

    all_results = {}
    t_global = time.time()
    for name in args.backbones:
        log_section(f"Backbone: {name} (v3.2 DeepResidualHead)")
        with timed(f"backbone-{name} setup"):
            backbone, base_tx, feat_dim, extract = build_backbone(name, device)
            for p in backbone.parameters():
                p.requires_grad_(False)
        train_tx = train_aug_factory(base_tx)
        fi_aug_tx = fi_view_aug_factory(base_tx)
        ds_tr = AffordanceDataset(tr_rows, train_tx)
        ds_va = AffordanceDataset(va_rows, base_tx)
        ds_te = AffordanceDataset(te_rows, base_tx)

        t0 = time.time()
        if name == "ffal":
            with timed("ffal-v32 training"):
                head, best, history = train_ffal_v32(
                    backbone, extract, ds_tr, ds_va, fi_aug_tx, feat_dim, device,
                    epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                    weight_decay=args.weight_decay, patience=args.patience,
                    lambda_fi=args.lambda_fi, n_views=args.n_views_fi,
                    n_aug_passes=args.n_aug_passes, log=log)
        else:
            with timed(f"head-v32 training ({name})"):
                head, best, history = train_head_v32(
                    backbone, extract, ds_tr, ds_va, feat_dim, device,
                    epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                    weight_decay=args.weight_decay, patience=args.patience,
                    n_aug_passes=args.n_aug_passes, log=log)
        train_time = time.time() - t0

        torch.save(dict(state_dict=head.state_dict(), feat_dim=feat_dim,
                        backbone=name,
                        head_arch="DeepResidualHead-512-256-128-64-32"),
                   MODELS / f"head_{name}_v32.pt")

        with timed(f"test+stability ({name} v3.2)"):
            res = evaluate_with_stability(
                name, backbone, extract, head, ds_te, base_tx,
                aug_transform_factory(base_tx), device,
                n_views=args.n_views_stability, log=log)
        res["best_val_auroc"] = best["val_auroc"]
        res["best_epoch"] = best["epoch"]
        res["train_time_s"] = train_time
        res["n_train_epochs_ran"] = len(history)
        # Keep history light (every 5th) to keep JSON small
        res["train_history"] = [h for h in history if h["epoch"] % 5 == 0 or h["epoch"] == len(history)-1]
        all_results[name] = res
        log(f"  TEST {name} (v3.2): AUROC={res['test_macro_auroc']:.4f} "
            f"F1={res['test_macro_f1']:.4f} latFIS={res['latent_fis']:.4f} "
            f"outFIS={res['output_fis']:.4f} lat={res['mean_latency_ms']:.1f}ms "
            f"train={train_time:.1f}s")

        del backbone, head
        if device.type == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    total_time = time.time() - t_global
    out = dict(
        run="v3.2_deepresidual_10k",
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
            p_drop=args.p_drop,
            head="DeepResidualHead in→512→256→128→64→32→K  LN+GELU+Drop(0.3) + residuals",
            augmentation="rot15+translate5+scale10+bright20+erasing6",
            data=args.data,
        ),
        results=all_results,
        total_train_time_s=total_time,
        torch_version=torch.__version__,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    out_json = RESULTS / args.out_json
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nWrote {out_json}  (total {total_time/60:.1f} min)")
    write_report(out, out_json.with_suffix(".md").name)


def write_report(out, md_name):
    md = []
    md.append("# v3.2 Benchmark — Deep Residual Head on 10K MuJoCo samples\n")
    md.append(f"- Device: `{out['device']}`  | torch `{out['torch_version']}`")
    md.append(f"- N samples: **{out['n_rows']}**  | split (by object_id): "
              f"train={out['split_sizes']['train']} val={out['split_sizes']['val']} "
              f"test={out['split_sizes']['test']}")
    cfg = out["config"]
    md.append(f"- Config: epochs={cfg['epochs']} batch={cfg['batch']} lr={cfg['lr']} "
              f"patience={cfg['patience']} λ_FI={cfg['lambda_fi']} drop={cfg['p_drop']}")
    md.append(f"- Head: `{cfg['head']}`\n")
    md.append("## Headline (v3.2)\n")
    md.append("| Model | Macro AUROC | Macro F1 | Latent FIS | **Output FIS** | Latency (ms/img) | Train (s) | Best epoch |")
    md.append("|---|---|---|---|---|---|---|---|")
    for name, r in out["results"].items():
        md.append(f"| {name} | {r['test_macro_auroc']:.4f} | {r['test_macro_f1']:.4f} | "
                  f"{r['latent_fis']:.4f} | **{r['output_fis']:.4f}** | "
                  f"{r['mean_latency_ms']:.1f} | {r.get('train_time_s', 0):.1f} | "
                  f"{r.get('best_epoch', '-')} |")
    md.append("\n## Per-affordance AUROC\n")
    md.append("| Model | " + " | ".join(AFFORDANCES) + " |")
    md.append("|---|" + "|".join(["---"]*len(AFFORDANCES)) + "|")
    for name, r in out["results"].items():
        md.append("| " + name + " | " +
                  " | ".join(f"{r['per_affordance_auroc'][a]:.3f}" for a in AFFORDANCES) + " |")
    md.append(f"\nTotal training time: **{out['total_train_time_s']/60:.1f} min**.")
    md.append("\n_Generated by `04_train_v32.py`. Seed = "
              f"{out['seed']}. Reproduce with `python3 scripts/04_train_v32.py`._\n")
    (RESULTS / md_name).write_text("\n".join(md))
    print(f"Wrote {RESULTS / md_name}")


if __name__ == "__main__":
    main()
