"""Step 3 (v3.2): Large-scale parallel MuJoCo data collection.

Goal: scale from 1,020 → 10,000 real-physics affordance samples (100 objects ×
100 episodes), produced by actual MuJoCo rollouts and rendered PNGs. No mocks.

Strategy:
- 100 procedurally generated objects (was 30)
- 100 episodes / object (was 34) — each episode gets its own probe rng + render
- Parallelism: `multiprocessing.Pool` with `cpu_count()-1` workers, work split
  by object so each worker handles a contiguous slice of objects and writes
  to its own shard. We then concatenate shards into the final JSONL.
- Same physics probes and image renderer as v3 (`01_collect_mujoco.py`) —
  imported, not copied, so v3.2 inherits v3's correctness guarantees.

Output (overwrites v3.2 data dir, leaves v3 dir untouched):
  data/v32/affordances_v32.jsonl        - 10,000 rows
  data/v32/images/{obj_id}_ep{NNN}.png  - 10,000 PNGs
  data/v32/object_catalog_v32.json      - 100-object catalog
  logs/03_large_scale.log               - timing summary
"""
from __future__ import annotations
import os, sys, json, time, math, traceback, argparse, pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DATA, LOGS, ROOT, seed_everything, SEED, log_section

# Reuse v3 physics + renderer
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_v3collect",
    Path(__file__).resolve().parent / "01_collect_mujoco.py",
)
_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3)

SHAPES = _v3.SHAPES
SURFACES = _v3.SURFACES
build_world_xml = _v3.build_world_xml
build_or_fallback = _v3.build_or_fallback
PROBES = _v3.PROBES
render_object_image = _v3.render_object_image
SEED_FOR = _v3.SEED_FOR


# --------------------------------------------------------------------------
# Expanded catalog (100 objects, broader parameter ranges)
# --------------------------------------------------------------------------

def make_large_catalog(n=100, seed=SEED):
    rng = np.random.default_rng(seed ^ 0xC0FFEE)
    cat = []
    for i in range(n):
        # Shape & surface combined to give a 5*4=20 base bucket × 5 variants.
        shape = SHAPES[i % len(SHAPES)]
        surf = SURFACES[(i // len(SHAPES)) % len(SURFACES)]
        # Wider continuous ranges than v3 for better coverage
        size_x = float(rng.uniform(0.05, 0.40))
        size_y = float(rng.uniform(0.05, 0.40))
        size_z = float(rng.uniform(0.06, 0.50))
        if shape == "sphere":
            size_y = size_z = size_x
        elif shape == "cylinder":
            size_y = size_x
        density = float(rng.uniform(120, 1200))
        friction = float(rng.uniform(0.15, 1.10))
        hollow_depth = 0.0
        if surf == "hollow_top":
            hollow_depth = float(rng.uniform(0.015, min(0.10, size_z * 0.55)))
        cat.append(dict(
            obj_id=f"obj_{i:03d}",
            shape=shape,
            surface=surf,
            size=[size_x, size_y, size_z],
            density=density,
            friction=friction,
            hollow_depth=hollow_depth,
        ))
    return cat


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

def _run_one_object(args):
    """Pure-function worker. Returns (obj_id, list[row_dict], stats)."""
    obj, n_episodes, images_dir, root_str = args
    images_dir = Path(images_dir)
    rows = []
    t0 = time.time()
    n_renders = 0
    n_render_errs = 0
    n_probe_errs = 0
    for ep in range(n_episodes):
        seed = SEED_FOR_V32(obj["__index__"], ep)
        ep_rng = np.random.default_rng(seed)
        img_path = images_dir / f"{obj['obj_id']}_ep{ep:03d}.png"
        if not img_path.exists():
            try:
                render_object_image(obj, img_path, episode=ep,
                                    rng=np.random.default_rng(seed ^ 0x5C5C))
                n_renders += 1
            except Exception:
                n_render_errs += 1
        row = dict(
            sample_id=f"{obj['obj_id']}_ep{ep:03d}",
            obj_id=obj["obj_id"],
            episode=ep,
            shape=obj["shape"],
            surface=obj["surface"],
            size=obj["size"],
            density=obj["density"],
            friction=obj["friction"],
            hollow_depth=obj["hollow_depth"],
            image=str(img_path.relative_to(root_str)),
        )
        for aff_name, probe in PROBES:
            try:
                label, telem = probe(obj, ep_rng)
            except Exception as e:
                label = 0
                telem = dict(error=f"{type(e).__name__}: {e}")
                n_probe_errs += 1
            row[f"label_{aff_name}"] = int(label)
            row[f"telem_{aff_name}"] = telem
        rows.append(row)
    dt = time.time() - t0
    return dict(
        obj_id=obj["obj_id"],
        rows=rows,
        elapsed_s=dt,
        n_renders=n_renders,
        n_render_errs=n_render_errs,
        n_probe_errs=n_probe_errs,
    )


def SEED_FOR_V32(oi, ep):
    return ((oi * 10000 + ep) ^ 0xBEEF) & 0x7FFFFFFF


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-objects", type=int, default=100)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--data-subdir", type=str, default="v32")
    args = ap.parse_args()

    seed_everything()
    data_dir = DATA / args.data_subdir
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / "03_large_scale.log"
    log_fp = open(log_path, "w")
    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n"); log_fp.flush()

    log_section(f"v3.2 large-scale collection: "
                f"{args.n_objects} objects × {args.n_episodes} episodes "
                f"= {args.n_objects * args.n_episodes} samples")
    log(f"workers={args.workers}, cpu_count={cpu_count()}")
    log(f"data_dir={data_dir}")

    catalog = make_large_catalog(n=args.n_objects)
    # tag with index so the worker can derive the seed deterministically
    for i, obj in enumerate(catalog):
        obj["__index__"] = i
    with open(data_dir / "object_catalog_v32.json", "w") as f:
        json.dump(catalog, f, indent=2)
    log(f"catalog written: {len(catalog)} objects")

    work = [(obj, args.n_episodes, str(images_dir), str(ROOT)) for obj in catalog]

    t_global = time.time()
    all_rows = []
    stats = []
    if args.workers <= 1:
        for w in work:
            r = _run_one_object(w)
            stats.append(r)
            all_rows.extend(r["rows"])
            log(f"  {r['obj_id']} done in {r['elapsed_s']:.1f}s "
                f"(renders={r['n_renders']}, render_errs={r['n_render_errs']}, "
                f"probe_errs={r['n_probe_errs']})")
    else:
        with Pool(processes=args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_one_object, work)):
                stats.append(r)
                all_rows.extend(r["rows"])
                elapsed = time.time() - t_global
                done = len(stats)
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(work) - done) / rate if rate > 0 else 0
                log(f"  [{done:3d}/{len(work)}] {r['obj_id']} "
                    f"{r['elapsed_s']:.1f}s | total {elapsed:.1f}s | eta {eta:.0f}s")

    # Sort by obj_id then episode so the JSONL is deterministic
    all_rows.sort(key=lambda r: (r["obj_id"], r["episode"]))

    out_path = data_dir / "affordances_v32.jsonl"
    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total_time = time.time() - t_global

    # Stats
    total_renders = sum(s["n_renders"] for s in stats)
    total_render_errs = sum(s["n_render_errs"] for s in stats)
    total_probe_errs = sum(s["n_probe_errs"] for s in stats)
    class_balance = {a: [0, 0] for a, _ in PROBES}
    for r in all_rows:
        for a, _ in PROBES:
            class_balance[a][r[f"label_{a}"]] += 1

    log_section(f"DONE: {len(all_rows)} samples in {total_time/60:.2f} min")
    log(f"new renders: {total_renders}, render errors: {total_render_errs}, "
        f"probe errors: {total_probe_errs}")
    log(f"output: {out_path}")
    log("\nClass balance:")
    for a, (n0, n1) in class_balance.items():
        tot = n0 + n1
        log(f"  {a:10s} pos={n1:5d} ({n1/tot:.2%}), neg={n0:5d}")

    log_fp.close()
    summary = dict(
        n_objects=args.n_objects,
        n_episodes=args.n_episodes,
        n_samples=len(all_rows),
        total_time_s=total_time,
        new_renders=total_renders,
        render_errors=total_render_errs,
        probe_errors=total_probe_errs,
        workers=args.workers,
        class_balance={a: dict(neg=v[0], pos=v[1]) for a, v in class_balance.items()},
        data_path=str(out_path.relative_to(ROOT)),
    )
    with open(LOGS / "03_large_scale_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {LOGS / '03_large_scale_summary.json'}")


if __name__ == "__main__":
    main()
