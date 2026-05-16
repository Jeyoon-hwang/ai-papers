"""Shared utilities for FFAL v3 real pipeline.

All scripts import from here so seeds, paths, and device choice are consistent.
No mock data anywhere in the pipeline.
"""
from __future__ import annotations
import os, sys, json, time, random, hashlib, pathlib, contextlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
for p in (DATA, IMAGES, MODELS, RESULTS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20260516

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def jsonl_writer(path: pathlib.Path):
    f = open(path, "w", encoding="utf-8")
    def write(obj):
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
    def close():
        f.close()
    return write, close

def jsonl_iter(path: pathlib.Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def log_section(title: str):
    bar = "=" * 68
    print(f"\n{bar}\n{title}\n{bar}", flush=True)

@contextlib.contextmanager
def timed(name: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[timed] {name}: {dt:.2f}s", flush=True)
