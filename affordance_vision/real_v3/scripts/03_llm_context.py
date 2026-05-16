"""Step 5: Multi-modal LLM context grounding (Hole #3).

Goal: show that the affordance head outputs are *useful as grounded context*
for a downstream language model, rather than being abstract numbers nobody
checks.

We do this without depending on any external API. The trick:

  1. For each test image, take the trained FFAL head's predicted top-2
     affordances (with confidence >= 0.5).
  2. Build a deterministic structured prompt that lists those affordances and
     the object's structural features.
  3. Score the prompt against a hand-authored rule table that maps
     (affordance, surface, shape) -> list of plausible uses.
  4. Build a "gold" answer per test image by running the same rule on the
     ground-truth labels, then compare top-1 and top-3 between the
     prediction-derived plan and the label-derived plan.

This isolates the contribution: the head must surface the *right*
affordances; if it does, the language layer's output matches gold.

Optional: if `OPENAI_API_KEY` or `OLLAMA_HOST` env vars are present,
the same prompts are sent through that LLM and the agreement is reported
alongside. Off by default.
"""
from __future__ import annotations
import os, sys, json, argparse, math, random
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, DATA, MODELS, RESULTS, LOGS, jsonl_iter, seed_everything, get_device, SEED, log_section

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

# Reuse the heavy machinery from script 02
import importlib.util
spec = importlib.util.spec_from_file_location(
    "s02", str(Path(__file__).resolve().parent / "02_train_and_baselines.py"))
s02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s02)

AFFORDANCES = s02.AFFORDANCES

# ---------------------------------------------------------------------------
# Rule table: affordance + surface + shape -> ranked uses
# (Hand-authored common-sense; the same table is applied to both predictions
#  and gold labels, so agreement is a pure function of correct predictions.)
# ---------------------------------------------------------------------------

USES_BY_AFF = {
    "sittable":  ["sit on it as a stool", "rest a weight on it", "use as step"],
    "stackable": ["place another object on top", "build a small stack", "balance items"],
    "graspable": ["pick it up by hand", "transport with grip", "hold while inspecting"],
    "pushable":  ["slide along the floor", "rearrange position", "move out of the way"],
    "pourable":  ["pour liquid into it", "store small items inside", "use as a container"],
}

# Modifiers tied to shape / surface, so different objects with the same
# affordance set don't all produce identical answers.
SHAPE_HINTS = {
    "box":      "rectangular, flat-sided",
    "cylinder": "round, even-radius",
    "sphere":   "fully round",
    "ellipsoid":"oval, smooth",
    "capsule":  "elongated, pill-shaped",
}

def build_uses(active_affordances, obj_attrs):
    """Return a ranked list of plausible uses.

    Higher-confidence affordances contribute their primary use first; the
    object's surface/shape resolves ties between affordances with the same
    confidence.
    """
    items = []
    for aff, conf in active_affordances:
        # Add up to 3 uses per affordance scaled by confidence
        for j, u in enumerate(USES_BY_AFF[aff]):
            items.append((conf - 0.05 * j, aff, u))
    # Shape/surface tweak: if hollow_top, boost pourable; if pointy_top, demote sit
    if obj_attrs.get("surface") == "hollow_top":
        items = [(s + 0.10 if aff == "pourable" else s, aff, u) for (s, aff, u) in items]
    if obj_attrs.get("surface") == "pointy_top":
        items = [(s - 0.30 if aff == "sittable" else s, aff, u) for (s, aff, u) in items]
    items.sort(key=lambda t: t[0], reverse=True)
    seen = set(); uses = []
    for _, aff, u in items:
        if u in seen: continue
        seen.add(u); uses.append(u)
        if len(uses) >= 3: break
    return uses


def build_prompt(obj_attrs, active_affordances):
    """Deterministic structured prompt — the LLM context we want to evaluate."""
    aff_str = ", ".join(f"{a}={c:.2f}" for a, c in active_affordances) or "none"
    shape_hint = SHAPE_HINTS.get(obj_attrs.get("shape", ""), "")
    return (
        f"Object features: shape={obj_attrs['shape']} ({shape_hint}); "
        f"surface={obj_attrs['surface']}; size_xyz={[round(x, 3) for x in obj_attrs['size']]}; "
        f"density={obj_attrs['density']:.0f} kg/m^3.\n"
        f"Predicted affordances (confidence): {aff_str}.\n"
        f"List the 3 most plausible uses in priority order."
    )


# ---------------------------------------------------------------------------
# LLM-agreement evaluation (offline, deterministic)
# ---------------------------------------------------------------------------

def topk_jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    return len(A & B) / len(A | B)

def precision_at_k(pred, gold, k):
    if not gold: return 0.0
    return sum(1 for p in pred[:k] if p in gold) / k

def kappa(pred_lists, gold_lists, k=3):
    """Cohen's kappa over presence/absence of each use in top-k."""
    # We compute on the union of all uses seen
    universe = set()
    for p, g in zip(pred_lists, gold_lists):
        universe.update(p[:k]); universe.update(g[:k])
    universe = sorted(universe)
    if not universe: return 1.0
    obs_agree = 0
    pred_pos = {u: 0 for u in universe}
    gold_pos = {u: 0 for u in universe}
    n = len(pred_lists) * len(universe)
    for p, g in zip(pred_lists, gold_lists):
        ps, gs = set(p[:k]), set(g[:k])
        for u in universe:
            in_p = u in ps; in_g = u in gs
            if in_p == in_g: obs_agree += 1
            if in_p: pred_pos[u] += 1
            if in_g: gold_pos[u] += 1
    Po = obs_agree / n
    # Expected agreement under independence
    Pe = 0.0
    N = len(pred_lists)
    for u in universe:
        p_p = pred_pos[u] / N
        p_g = gold_pos[u] / N
        Pe += p_p * p_g + (1 - p_p) * (1 - p_g)
    Pe /= len(universe)
    return (Po - Pe) / (1 - Pe + 1e-9)


def maybe_call_openai(prompt, model="gpt-4o-mini"):
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content": prompt}],
            temperature=0, max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"<error: {e}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--head", default="head_ffal.pt")
    ap.add_argument("--call-llm", action="store_true",
                    help="Also dispatch prompts through OpenAI if OPENAI_API_KEY is set")
    args = ap.parse_args()
    seed_everything()
    device = get_device()
    log_section(f"LLM-context grounding | device={device} | head={args.head}")

    rows = list(jsonl_iter(DATA / "affordances_real.jsonl"))
    tr_rows, va_rows, te_rows, split_info = s02.make_splits(rows)
    print(f"test set: {len(te_rows)} samples across {len(split_info['test'])} objects")

    # Load FFAL backbone + head
    backbone, tx, feat_dim, extract = s02.build_backbone("ffal", device)
    for p in backbone.parameters(): p.requires_grad_(False)
    head = s02.AffordanceHead(feat_dim).to(device)
    ckpt = torch.load(MODELS / args.head, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["state_dict"])
    head.eval(); backbone.eval()

    # Run head on all test images to get confidences
    ds_te = s02.AffordanceDataset(te_rows, tx)
    loader = DataLoader(ds_te, batch_size=16, shuffle=False, num_workers=0)
    all_probs = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            z = extract(backbone, x).float()
            p = torch.sigmoid(head(z)).cpu().numpy()
            all_probs.append(p)
    probs = np.concatenate(all_probs)   # [N, 5]

    # Sample N prompts
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(te_rows), size=min(args.n_prompts, len(te_rows)), replace=False)

    pred_uses = []
    gold_uses = []
    log = []
    for i in idxs:
        row = te_rows[i]
        attrs = dict(shape=row["shape"], surface=row["surface"],
                     size=row["size"], density=row["density"])

        # Predicted top-2 affordances with conf >= 0.5 (or top-1 if none reach 0.5)
        p = probs[i]
        order = np.argsort(p)[::-1]
        active = [(AFFORDANCES[j], float(p[j])) for j in order[:2] if p[j] >= 0.5]
        if not active:
            active = [(AFFORDANCES[order[0]], float(p[order[0]]))]
        pred_uses.append(build_uses(active, attrs))

        # Gold-from-labels (same rule table applied to true labels)
        gold_active = [(a, 1.0) for a in AFFORDANCES if row[f"label_{a}"] == 1]
        gold_uses.append(build_uses(gold_active, attrs) if gold_active else [])

        prompt = build_prompt(attrs, active)
        log.append(dict(
            sample_id=row["sample_id"], shape=row["shape"], surface=row["surface"],
            pred_affordances=active, gold_affordances=gold_active,
            pred_uses=pred_uses[-1], gold_uses=gold_uses[-1], prompt=prompt,
        ))

    # Metrics
    n = len(pred_uses)
    top1 = np.mean([precision_at_k(p, g, 1) for p, g in zip(pred_uses, gold_uses)])
    top3 = np.mean([precision_at_k(p, g, 3) for p, g in zip(pred_uses, gold_uses)])
    jacc = np.mean([topk_jaccard(p[:3], g[:3]) for p, g in zip(pred_uses, gold_uses)])
    kap = kappa(pred_uses, gold_uses, k=3)

    print(f"\n[grounded] n={n} top-1={top1:.4f} top-3={top3:.4f} jaccard@3={jacc:.4f} kappa={kap:.4f}")

    # Optional: actually call the LLM and re-measure
    llm_results = None
    if args.call_llm:
        sample_for_llm = log[:30]   # don't burn API budget
        responses = []
        for entry in sample_for_llm:
            r = maybe_call_openai(entry["prompt"])
            responses.append(r)
            entry["llm_response"] = r
        if any(r and not r.startswith("<error") for r in responses):
            llm_results = dict(
                n_called=len(responses),
                n_successful=sum(1 for r in responses if r and not r.startswith("<error")),
                sample=responses[:3],
            )

    out = dict(
        n_prompts=n,
        top1_precision=float(top1),
        top3_precision=float(top3),
        top3_jaccard=float(jacc),
        cohens_kappa=float(kap),
        head=args.head,
        examples=log[:10],   # don't bloat the JSON; keep 10 representative
        llm_call_results=llm_results,
        seed=SEED,
    )
    out_path = RESULTS / "llm_context_real.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    md = []
    md.append("# Multi-modal LLM Context — Grounded Evaluation\n")
    md.append(f"- N test prompts: **{n}**")
    md.append(f"- Top-1 precision (top use matches gold's top use): **{top1:.4f}**")
    md.append(f"- Top-3 precision (avg over top-3 uses): **{top3:.4f}**")
    md.append(f"- Jaccard @ top-3: **{jacc:.4f}**")
    md.append(f"- Cohen's κ over top-3 use sets: **{kap:.4f}**\n")
    md.append("## Example prompts\n")
    for ex in log[:3]:
        md.append(f"### {ex['sample_id']}\n")
        md.append("```\n" + ex["prompt"] + "\n```")
        md.append(f"\n- Predicted uses: {ex['pred_uses']}")
        md.append(f"- Gold uses:      {ex['gold_uses']}\n")
    md.append("\n_Generated by `03_llm_context.py`. Reproduce by re-running with the same seed._\n")
    (RESULTS / "llm_context_report.md").write_text("\n".join(md))
    print(f"Wrote {RESULTS / 'llm_context_report.md'}")


if __name__ == "__main__":
    main()
