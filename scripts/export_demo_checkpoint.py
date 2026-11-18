"""Retrain the best hybrid configuration (seed 0, same config as the study) and save the checkpoint used by the demo."""
import os, sys, json, yaml
ROOT = os.path.join(os.path.dirname(__file__), ".."); sys.path.insert(0, ROOT)
from src.train import run
res = os.path.join(ROOT, "experiments", "results")
best = max([json.load(open(os.path.join(res, f))) for f in os.listdir(res) if f.startswith("hybrid_lr") and f.endswith(".json")], key=lambda r: r["val_macro_f1"])
cfg = dict(best["cfg"]); cfg["name"] = "hybrid_demo"; cfg["save_ckpt"] = True
run(cfg)
