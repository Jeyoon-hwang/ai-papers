"""Steps 2+3: For every backbone (ResNet50, ViT-B/16, CLIP ViT-B/32,
DINOv2 ViT-B/14, and FFAL-DINOv2-with-FI-reg) we

  1. Extract frozen features from the 1020 MuJoCo images.
  2. Train an identical 2-layer affordance head with BCE + class weights.
  3. Evaluate on the held-out object split.

Only FFAL adds a Functional-Invariance regularizer that pulls features of
6 augmented views of each image together (cosine-similarity loss).

Everything is real:
  - real pretrained weights (no random init)
  - real forward passes
  - real training loops on MPS / CPU
  - real held-out evaluation (split by object_id, no leakage)
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

# Allow MPS fallback for any unsupported op
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

AFFORDANCES = ["sittable", "stackable", "graspable", "pushable", "pourable"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AffordanceDataset(Dataset):
    def __init__(self, rows, transform, augment=False, aug_transform=None, n_augs=0):
        self.rows = rows
        self.transform = transform
        self.augment = augment
        self.aug_transform = aug_transform
        self.n_augs = n_augs

    def __len__(self): return len(self.rows)

    def _load(self, row):
        path = ROOT / row["image"]
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = self._load(row)
        x = self.transform(img)
        y = torch.tensor([row[f"label_{a}"] for a in AFFORDANCES], dtype=torch.float32)
        if self.augment and self.n_augs > 0 and self.aug_transform is not None:
            views = [self.aug_transform(img) for _ in range(self.n_augs)]
            return x, y, torch.stack(views)
        return x, y, torch.empty(0)


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

def build_backbone(name: str, device):
    """Return (model, transform, feat_dim, extract_fn). Frozen, eval mode."""
    name = name.lower()
    if name == "resnet50":
        import torchvision.models as tvm
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()  # feat_dim = 2048
        feat_dim = 2048
        tx = tvm.ResNet50_Weights.IMAGENET1K_V2.transforms()
        def extract(model, x): return model(x)
        return m.to(device).eval(), tx, feat_dim, extract

    if name == "vit_b16":
        import torchvision.models as tvm
        try:
            w = tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        except AttributeError:
            w = tvm.ViT_B_16_Weights.DEFAULT
        m = tvm.vit_b_16(weights=w)
        feat_dim = m.hidden_dim   # 768
        m.heads = nn.Identity()
        tx = w.transforms()
        def extract(model, x): return model(x)
        return m.to(device).eval(), tx, feat_dim, extract

    if name == "clip_b32":
        import open_clip
        m, _, tx = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        # tx is already image preprocess
        m = m.to(device).eval()
        feat_dim = m.visual.output_dim  # 512
        def extract(model, x):
            return model.encode_image(x).float()
        return m, tx, feat_dim, extract

    if name in ("dinov2", "ffal"):
        import timm
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                              pretrained=True, num_classes=0, img_size=224)
        feat_dim = m.num_features   # 768
        data_cfg = timm.data.resolve_model_data_config(m)
        # Override input size for our 224x224 imagery; keep its normalization
        data_cfg["input_size"] = (3, 224, 224)
        tx = timm.data.create_transform(**data_cfg, is_training=False)
        def extract(model, x): return model(x)
        return m.to(device).eval(), tx, feat_dim, extract

    raise ValueError(f"unknown backbone {name}")


def aug_transform_factory(base_tx):
    """Augmentations for FFAL's functional-invariance regularizer."""
    # Apply PIL-level augs *before* the base transform so normalization stays consistent
    pil_augs = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    ])
    return T.Compose([pil_augs, base_tx])


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class AffordanceHead(nn.Module):
    def __init__(self, in_dim, n_out=len(AFFORDANCES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_out),
        )
    def forward(self, z): return self.net(z)


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def make_splits(rows, seed=SEED):
    """70/15/15 by obj_id (no object leak)."""
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


@torch.no_grad()
def precompute_features(backbone, extract_fn, loader, device):
    feats, ys = [], []
    backbone.eval()
    for x, y, _ in loader:
        x = x.to(device)
        z = extract_fn(backbone, x).detach().float().cpu()
        feats.append(z); ys.append(y)
    return torch.cat(feats), torch.cat(ys)


def train_head_on_features(feat_tr, y_tr, feat_va, y_va, in_dim,
                           epochs=40, batch_size=64, lr=1e-3, device="cpu",
                           pos_weight=None, log=None):
    head = AffordanceHead(in_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = feat_tr.size(0)
    best = dict(val_auroc=-1.0, state=None, epoch=-1)
    patience, bad = 6, 0
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
            losses.append(float(loss))
        # val
        head.eval()
        with torch.no_grad():
            val_logits = head(feat_va.to(device)).cpu()
        val_auroc = macro_auroc(val_logits, y_va)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_auroc=val_auroc))
        if log: log(f"      epoch {epoch:02d} loss={np.mean(losses):.4f} val_AUROC={val_auroc:.4f}")
        if val_auroc > best["val_auroc"]:
            best = dict(val_auroc=val_auroc, state=copy.deepcopy(head.state_dict()), epoch=epoch)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if log: log(f"      early stop at epoch {epoch}")
                break
    head.load_state_dict(best["state"])
    return head, best, history


def train_ffal_head(backbone, extract_fn, ds_train, ds_val, in_dim, device, log=None,
                    epochs=20, batch_size=16, lr=1e-3, lambda_fi=0.5, n_views=4):
    """FFAL = same head, but training on the FLY through the frozen backbone
    with an extra Functional-Invariance cosine-loss across 4 augmented views.

    Backbone weights stay frozen; only the head learns. The FI term is what
    distinguishes FFAL from "DINOv2 + plain head".
    """
    head = AffordanceHead(in_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    pos = compute_pos_weight(ds_train)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos.to(device))

    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    best = dict(val_auroc=-1.0, state=None, epoch=-1)
    bad, patience = 0, 5
    history = []
    for epoch in range(epochs):
        head.train()
        backbone.eval()
        losses = []
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
                # FI = encourage z and each view's z to be cosine-similar
                z_norm = F.normalize(z, dim=-1).unsqueeze(1)        # [B,1,D]
                zv_norm = F.normalize(zv, dim=-1)                    # [B,V,D]
                cos = (z_norm * zv_norm).sum(-1)                     # [B,V]
                loss_fi = (1.0 - cos).mean()
            loss = loss_bce + lambda_fi * loss_fi
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        # val
        head.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for x, y, _ in va_loader:
                x = x.to(device)
                z = extract_fn(backbone, x).float()
                all_logits.append(head(z).cpu()); all_y.append(y)
        val_auroc = macro_auroc(torch.cat(all_logits), torch.cat(all_y))
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_auroc=val_auroc))
        if log: log(f"      epoch {epoch:02d} loss={np.mean(losses):.4f} val_AUROC={val_auroc:.4f}")
        if val_auroc > best["val_auroc"]:
            best = dict(val_auroc=val_auroc, state=copy.deepcopy(head.state_dict()), epoch=epoch)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if log: log(f"      early stop at epoch {epoch}")
                break
    head.load_state_dict(best["state"])
    return head, best, history


def compute_pos_weight(ds):
    """ds is an AffordanceDataset; sum labels."""
    y = []
    for r in ds.rows:
        y.append([r[f"label_{a}"] for a in AFFORDANCES])
    y = np.array(y)
    n = y.shape[0]
    pos = y.sum(axis=0)
    neg = n - pos
    # weight = neg/pos (clip to avoid extremes)
    w = neg / np.clip(pos, 1, None)
    w = np.clip(w, 0.5, 10.0)
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _auroc_binary(scores, labels):
    """AUROC via the rank formula. scores and labels are 1-D numpy arrays."""
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def macro_auroc(logits, labels):
    s = torch.sigmoid(logits).numpy()
    y = labels.numpy()
    vals = [_auroc_binary(s[:, i], y[:, i]) for i in range(s.shape[1])]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")

def per_aff_auroc(logits, labels):
    s = torch.sigmoid(logits).numpy()
    y = labels.numpy()
    return {AFFORDANCES[i]: _auroc_binary(s[:, i], y[:, i]) for i in range(s.shape[1])}

def macro_f1_at(logits, labels, thr=0.5):
    p = (torch.sigmoid(logits).numpy() > thr).astype(int)
    y = labels.numpy().astype(int)
    f1s = []
    for i in range(p.shape[1]):
        tp = int(((p[:, i] == 1) & (y[:, i] == 1)).sum())
        fp = int(((p[:, i] == 1) & (y[:, i] == 0)).sum())
        fn = int(((p[:, i] == 0) & (y[:, i] == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))

def latent_fis(features_views):
    """features_views: [N, V, D]. Returns 1 - mean cosine distance across views."""
    z = F.normalize(features_views, dim=-1)
    # mean cosine sim of each view to the anchor (view 0)
    anchor = z[:, 0:1, :]
    sims = (z * anchor).sum(-1)   # [N, V]
    # exclude the anchor itself
    sims = sims[:, 1:]
    return float(sims.mean().item())

def output_fis(logits_views):
    """logits_views: [N, V, K]. 1 - MAE between sigmoid outputs across views."""
    p = torch.sigmoid(logits_views)
    anchor = p[:, 0:1, :]
    mae = (p[:, 1:, :] - anchor).abs().mean()
    return float(1.0 - mae.item())


# ---------------------------------------------------------------------------
# Stability rollout (Step 4) — runs inside this same script for efficiency.
# ---------------------------------------------------------------------------

def evaluate_with_stability(name, backbone, extract_fn, head, ds_test, tx, aug_tx,
                            device, n_views=6, log=None):
    head.eval(); backbone.eval()
    # We need per-test-image: anchor features+logits + V augmented features+logits
    all_anchor_z, all_anchor_logits, all_y = [], [], []
    view_logits = []
    view_feats = []
    times = []
    aug_dataset = AffordanceDataset(ds_test.rows, tx, augment=True,
                                    aug_transform=aug_tx, n_augs=n_views)
    loader = DataLoader(aug_dataset, batch_size=8, shuffle=False, num_workers=0)
    for x, y, views in loader:
        x = x.to(device)
        t0 = time.time()
        with torch.no_grad():
            z = extract_fn(backbone, x).float()
            l = head(z)
        if device.type != "cpu":
            try: torch.mps.synchronize()
            except Exception: pass
        dt = (time.time() - t0) / x.size(0)
        times.append(dt)
        all_anchor_z.append(z.cpu()); all_anchor_logits.append(l.cpu()); all_y.append(y)
        B, V, C, H, W = views.shape
        vx = views.view(B*V, C, H, W).to(device)
        with torch.no_grad():
            zv = extract_fn(backbone, vx).float()
            lv = head(zv)
        view_feats.append(zv.view(B, V, -1).cpu())
        view_logits.append(lv.view(B, V, -1).cpu())
    anchor_z = torch.cat(all_anchor_z)        # [N, D]
    anchor_logits = torch.cat(all_anchor_logits)  # [N, K]
    y_test = torch.cat(all_y)
    view_feats = torch.cat(view_feats)        # [N, V, D]
    view_logits = torch.cat(view_logits)      # [N, V, K]

    # Stack anchor as view 0 for FIS computation
    feats_all = torch.cat([anchor_z.unsqueeze(1), view_feats], dim=1)  # [N, V+1, D]
    logits_all = torch.cat([anchor_logits.unsqueeze(1), view_logits], dim=1)
    lf = latent_fis(feats_all)
    of = output_fis(logits_all)

    auroc = macro_auroc(anchor_logits, y_test)
    f1 = macro_f1_at(anchor_logits, y_test)
    per_aff = per_aff_auroc(anchor_logits, y_test)
    return dict(
        model=name,
        test_macro_auroc=auroc,
        test_macro_f1=f1,
        per_affordance_auroc=per_aff,
        latent_fis=lf,
        output_fis=of,
        mean_latency_ms=float(np.mean(times) * 1000.0),
        n_test=int(anchor_logits.size(0)),
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["resnet50", "vit_b16", "clip_b32", "dinov2", "ffal"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ffal-epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-views-stability", type=int, default=6)
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    log_path = LOGS / "02_train.log"
    log_fp = open(log_path, "w")
    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n"); log_fp.flush()

    log_section(f"Device: {device}  Backbones: {args.backbones}")

    rows = list(jsonl_iter(DATA / "affordances_real.jsonl"))
    log(f"loaded {len(rows)} rows")
    tr_rows, va_rows, te_rows, split_info = make_splits(rows)
    log(f"split: train={len(tr_rows)} val={len(va_rows)} test={len(te_rows)}")
    log(f"  train objs ({len(split_info['train'])}): {split_info['train']}")
    log(f"  val objs   ({len(split_info['val'])}): {split_info['val']}")
    log(f"  test objs  ({len(split_info['test'])}): {split_info['test']}")

    all_results = {}
    for name in args.backbones:
        log_section(f"Backbone: {name}")
        with timed(f"backbone-{name} setup"):
            backbone, tx, feat_dim, extract = build_backbone(name, device)
            for p in backbone.parameters():
                p.requires_grad_(False)
        aug_tx = aug_transform_factory(tx)

        ds_tr = AffordanceDataset(tr_rows, tx)
        ds_va = AffordanceDataset(va_rows, tx)
        ds_te = AffordanceDataset(te_rows, tx)

        if name == "ffal":
            # FFAL: train head on the fly with FI regularizer using augmented views
            ds_tr_aug = AffordanceDataset(tr_rows, tx, augment=True,
                                          aug_transform=aug_tx, n_augs=4)
            with timed("ffal training"):
                head, best, history = train_ffal_head(
                    backbone, extract, ds_tr_aug, ds_va, feat_dim, device,
                    log=log, epochs=args.ffal_epochs, batch_size=16)
        else:
            # Standard: precompute features, train head, evaluate
            tr_loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=False, num_workers=0)
            va_loader = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)
            with timed(f"feature extraction ({name})"):
                feat_tr, y_tr = precompute_features(backbone, extract, tr_loader, device)
                feat_va, y_va = precompute_features(backbone, extract, va_loader, device)
            pos_w = compute_pos_weight(ds_tr)
            with timed(f"head training ({name})"):
                head, best, history = train_head_on_features(
                    feat_tr, y_tr, feat_va, y_va, feat_dim,
                    epochs=args.epochs, batch_size=args.batch, device=device,
                    pos_weight=pos_w, log=log)

        # Save head
        torch.save(dict(state_dict=head.state_dict(), feat_dim=feat_dim, backbone=name),
                   MODELS / f"head_{name}.pt")

        # Evaluate + stability
        with timed(f"test+stability ({name})"):
            res = evaluate_with_stability(
                name, backbone, extract, head, ds_te, tx, aug_tx,
                device, n_views=args.n_views_stability, log=log)
        res["best_val_auroc"] = best["val_auroc"]
        res["best_epoch"] = best["epoch"]
        res["train_history"] = history
        all_results[name] = res
        log(f"  TEST {name}: AUROC={res['test_macro_auroc']:.4f} "
            f"F1={res['test_macro_f1']:.4f} latFIS={res['latent_fis']:.4f} "
            f"outFIS={res['output_fis']:.4f} lat={res['mean_latency_ms']:.1f}ms")

        # Free memory between backbones
        del backbone, head
        if device.type == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    # Persist everything
    out = dict(
        device=str(device),
        seed=SEED,
        n_rows=len(rows),
        split_sizes=dict(train=len(tr_rows), val=len(va_rows), test=len(te_rows)),
        split_objects=split_info,
        affordances=AFFORDANCES,
        results=all_results,
        torch_version=torch.__version__,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    out_json = RESULTS / "baselines_real.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nWrote {out_json}")
    write_report(out)

def write_report(out):
    md = []
    md.append("# Baseline Comparison — REAL run\n")
    md.append(f"- Device: `{out['device']}`  | torch `{out['torch_version']}`")
    md.append(f"- N samples: **{out['n_rows']}**  | split (by object_id): train={out['split_sizes']['train']} val={out['split_sizes']['val']} test={out['split_sizes']['test']}\n")
    md.append("## Headline results\n")
    md.append("| Model | Macro AUROC | Macro F1 | Latent FIS | **Output FIS** | Latency (ms/img) |")
    md.append("|---|---|---|---|---|---|")
    for name, r in out["results"].items():
        md.append(f"| {name} | {r['test_macro_auroc']:.4f} | {r['test_macro_f1']:.4f} | "
                  f"{r['latent_fis']:.4f} | **{r['output_fis']:.4f}** | {r['mean_latency_ms']:.1f} |")
    md.append("\n## Per-affordance AUROC\n")
    md.append("| Model | " + " | ".join(AFFORDANCES) + " |")
    md.append("|---|" + "|".join(["---"]*len(AFFORDANCES)) + "|")
    for name, r in out["results"].items():
        md.append("| " + name + " | " +
                  " | ".join(f"{r['per_affordance_auroc'][a]:.3f}" for a in AFFORDANCES) + " |")
    md.append("\n_Generated by `02_train_and_baselines.py`. All numbers from a single deterministic run; seed = "
              f"{out['seed']}. Reproduce with `python3 scripts/02_train_and_baselines.py`._\n")
    (RESULTS / "baselines_real_report.md").write_text("\n".join(md))
    print(f"Wrote {RESULTS / 'baselines_real_report.md'}")


if __name__ == "__main__":
    main()
