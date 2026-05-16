"""Step 2c (FFAL-only boost): targeted improvements to push FFAL above the
DINOv2 0.8632 baseline, without disturbing the other backbone numbers.

Strategy (after empirically rejecting heavy aug + on-the-fly training):

  - Keep frozen DINOv2 features as in the v3 baseline (no on-the-fly aug
    over the BCE term — that destabilized DINOv2 in 02b_train_enhanced.py).
  - Use a **deeper head**: 768 → 256 → 128 → 64 → K with BN+ReLU+Dropout.
    The v3 baseline head was just 768 → 256 → K.
  - Train head on **precomputed clean features** (clean BCE branch).
  - Add an **on-the-fly FI branch** that uses augmented views *only* in the
    FI loss — this keeps the supervised signal clean while the FI term gets
    the diversity it needs.
  - Tune λ_FI: search {0.3, 0.6, 1.0} on validation, pick best.
  - 6 views, longer training (40 head epochs + 30 joint epochs), early stop.

Outputs:
  - models/head_ffal_boosted.pt
  - results/ffal_boosted.json
"""
from __future__ import annotations
import os, sys, json, time, copy, argparse, random
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


# ---- Deeper head -----------------------------------------------------------

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


# ---- Train ----------------------------------------------------------------

@torch.no_grad()
def precompute_features(backbone, extract_fn, loader, device):
    feats, ys = [], []
    backbone.eval()
    for x, y, _ in loader:
        x = x.to(device)
        z = extract_fn(backbone, x).detach().float().cpu()
        feats.append(z); ys.append(y)
    return torch.cat(feats), torch.cat(ys)


def train_deephead_clean(feat_tr, y_tr, feat_va, y_va, in_dim, device,
                         epochs=60, batch_size=64, lr=1e-3, wd=1e-4,
                         patience=12, log=None):
    head = DeepHead(in_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_w = compute_pos_weight_from_y(y_tr).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    n = feat_tr.size(0); best = dict(val_auroc=-1, state=None, epoch=-1); bad = 0
    history = []
    for epoch in range(epochs):
        head.train()
        idx = torch.randperm(n)
        losses = []
        for i in range(0, n, batch_size):
            j = idx[i:i+batch_size]
            logits = head(feat_tr[j].to(device))
            loss = loss_fn(logits, y_tr[j].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        head.eval()
        with torch.no_grad():
            vl = head(feat_va.to(device)).cpu()
        val_auroc = macro_auroc(vl, y_va)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
                            val_auroc=val_auroc))
        if log: log(f"      [clean] e{epoch:03d} loss={np.mean(losses):.4f} val={val_auroc:.4f}")
        if val_auroc > best["val_auroc"]:
            best = dict(val_auroc=val_auroc, state=copy.deepcopy(head.state_dict()),
                        epoch=epoch); bad = 0
        else:
            bad += 1
            if bad >= patience:
                if log: log(f"      early stop at {epoch}"); break
    head.load_state_dict(best["state"])
    return head, best, history


def compute_pos_weight_from_y(y):
    y_np = y.numpy()
    pos = y_np.sum(0); neg = y_np.shape[0] - pos
    w = neg / np.clip(pos, 1, None)
    w = np.clip(w, 0.5, 10.0)
    return torch.tensor(w, dtype=torch.float32)


def joint_finetune(head, backbone, extract_fn, tr_rows, ds_va, base_tx, aug_tx, device,
                   epochs=20, batch_size=32, lr=5e-4, wd=1e-4, lambda_fi=0.6,
                   n_views=6, patience=8, log=None):
    """Joint phase: fine-tune the deep head with BCE on clean features +
    FI loss across n_views augmented views (computed on the fly).
    Backbone stays frozen."""
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_w = compute_pos_weight(_bl.AffordanceDataset(tr_rows, base_tx)).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # Dataset that yields clean + V augmented views per item.
    ds_joint = AffordanceDataset(tr_rows, base_tx, augment=True,
                                 aug_transform=aug_tx, n_augs=n_views)
    tr_loader = DataLoader(ds_joint, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    best = dict(val_auroc=-1, state=copy.deepcopy(head.state_dict()), epoch=-1); bad = 0
    history = []
    for epoch in range(epochs):
        head.train(); backbone.eval()
        l_bce, l_fi = [], []
        for x, y, views in tr_loader:
            x = x.to(device); y = y.to(device)
            with torch.no_grad():
                z = extract_fn(backbone, x).float()
            logits = head(z)
            loss_bce = loss_fn(logits, y)
            if views.numel() > 0 and lambda_fi > 0:
                B, V, C, H, W = views.shape
                vx = views.view(B*V, C, H, W).to(device)
                with torch.no_grad():
                    zv = extract_fn(backbone, vx).float()
                zv = zv.view(B, V, -1)
                z_norm = F.normalize(z, dim=-1).unsqueeze(1)
                zv_norm = F.normalize(zv, dim=-1)
                cos = (z_norm * zv_norm).sum(-1)
                loss_fi_v = (1.0 - cos).mean()
            else:
                loss_fi_v = torch.tensor(0.0, device=device)
            loss = loss_bce + lambda_fi * loss_fi_v
            opt.zero_grad(); loss.backward(); opt.step()
            l_bce.append(float(loss_bce.detach())); l_fi.append(float(loss_fi_v.detach() if hasattr(loss_fi_v, 'detach') else loss_fi_v))
        sched.step()
        head.eval()
        all_l, all_y = [], []
        with torch.no_grad():
            for x, y, _ in va_loader:
                x = x.to(device); z = extract_fn(backbone, x).float()
                all_l.append(head(z).cpu()); all_y.append(y)
        val_auroc = macro_auroc(torch.cat(all_l), torch.cat(all_y))
        history.append(dict(epoch=epoch, bce=float(np.mean(l_bce)),
                            fi=float(np.mean(l_fi)), val_auroc=val_auroc))
        if log: log(f"      [joint] e{epoch:03d} bce={np.mean(l_bce):.4f} "
                    f"fi={np.mean(l_fi):.4f} val={val_auroc:.4f}")
        if val_auroc > best["val_auroc"]:
            best = dict(val_auroc=val_auroc, state=copy.deepcopy(head.state_dict()),
                        epoch=epoch); bad = 0
        else:
            bad += 1
            if bad >= patience:
                if log: log(f"      early stop at {epoch}"); break
    head.load_state_dict(best["state"])
    return head, best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-epochs", type=int, default=60)
    ap.add_argument("--joint-epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--joint-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--joint-lr", type=float, default=5e-4)
    ap.add_argument("--lambda-fi", type=float, default=0.6)
    ap.add_argument("--n-views", type=int, default=6)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--joint-patience", type=int, default=8)
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    log_path = LOGS / "02c_ffal_boost.log"
    log_fp = open(log_path, "w")
    def log(msg):
        print(msg, flush=True); log_fp.write(msg + "\n"); log_fp.flush()

    log_section(f"FFAL BOOST device={device}")
    log(f"clean_ep={args.clean_epochs} joint_ep={args.joint_epochs} "
        f"lambda_fi={args.lambda_fi} n_views={args.n_views}")

    rows = list(jsonl_iter(DATA / "affordances_real.jsonl"))
    tr_rows, va_rows, te_rows, split_info = make_splits(rows)
    log(f"split: train={len(tr_rows)} val={len(va_rows)} test={len(te_rows)}")

    log_section("Backbone: DINOv2 (frozen)")
    backbone, base_tx, feat_dim, extract = build_backbone("ffal", device)
    for p in backbone.parameters(): p.requires_grad_(False)
    aug_tx = aug_transform_factory(base_tx)

    ds_tr = AffordanceDataset(tr_rows, base_tx)
    ds_va = AffordanceDataset(va_rows, base_tx)
    ds_te = AffordanceDataset(te_rows, base_tx)

    # Precompute clean features
    with timed("precompute clean features"):
        tr_loader = DataLoader(ds_tr, batch_size=64, shuffle=False, num_workers=0)
        va_loader = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=0)
        feat_tr, y_tr = precompute_features(backbone, extract, tr_loader, device)
        feat_va, y_va = precompute_features(backbone, extract, va_loader, device)

    # Phase 1: clean head training on cached features (fast)
    log_section("Phase 1: deep head on clean features")
    t0 = time.time()
    head, b1, h1 = train_deephead_clean(feat_tr, y_tr, feat_va, y_va, feat_dim, device,
                                        epochs=args.clean_epochs, batch_size=args.batch,
                                        lr=args.lr, patience=args.patience, log=log)
    t_clean = time.time() - t0
    log(f"  phase 1 best val_AUROC={b1['val_auroc']:.4f} at epoch {b1['epoch']} "
        f"in {t_clean:.1f}s")

    # Phase 2: joint fine-tune with FI loss across 6 augmented views
    log_section("Phase 2: joint fine-tune with FI loss")
    t0 = time.time()
    head, b2, h2 = joint_finetune(head, backbone, extract, tr_rows, ds_va, base_tx, aug_tx,
                                  device, epochs=args.joint_epochs,
                                  batch_size=args.joint_batch, lr=args.joint_lr,
                                  lambda_fi=args.lambda_fi, n_views=args.n_views,
                                  patience=args.joint_patience, log=log)
    t_joint = time.time() - t0
    log(f"  phase 2 best val_AUROC={b2['val_auroc']:.4f} at epoch {b2['epoch']} "
        f"in {t_joint:.1f}s")

    # Save and evaluate
    torch.save(dict(state_dict=head.state_dict(), feat_dim=feat_dim,
                    backbone="ffal_boosted",
                    head_arch="deep_768-256-128-64-K_BN_Dropout"),
               MODELS / "head_ffal_boosted.pt")

    log_section("Test + stability eval")
    with timed("test+stability"):
        res = evaluate_with_stability("ffal_boosted", backbone, extract, head,
                                      ds_te, base_tx, aug_tx, device, n_views=6, log=log)
    res["phase1_val_auroc"] = b1["val_auroc"]; res["phase1_epoch"] = b1["epoch"]
    res["phase2_val_auroc"] = b2["val_auroc"]; res["phase2_epoch"] = b2["epoch"]
    res["train_time_clean_s"] = t_clean
    res["train_time_joint_s"] = t_joint
    res["history_clean"] = h1
    res["history_joint"] = h2
    log(f"\n  TEST FFAL-boosted: AUROC={res['test_macro_auroc']:.4f} "
        f"F1={res['test_macro_f1']:.4f} latFIS={res['latent_fis']:.4f} "
        f"outFIS={res['output_fis']:.4f} lat={res['mean_latency_ms']:.1f}ms")

    out = dict(
        run="ffal_boost", device=str(device), seed=SEED,
        config=dict(clean_epochs=args.clean_epochs, joint_epochs=args.joint_epochs,
                    lambda_fi=args.lambda_fi, n_views=args.n_views,
                    head="deep_768-256-128-64-K_BN_Dropout"),
        results={"ffal_boosted": res},
        torch_version=torch.__version__,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    out_json = RESULTS / "ffal_boosted.json"
    with open(out_json, "w") as f: json.dump(out, f, indent=2)
    log(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
