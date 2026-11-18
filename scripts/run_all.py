"""Run every experiment of the study sequentially (resumable).  python scripts/run_all.py"""
import os, sys, json, yaml, subprocess
ROOT = os.path.join(os.path.dirname(__file__), ".."); sys.path.insert(0, ROOT)
CFG, RES = os.path.join(ROOT, "configs"), os.path.join(ROOT, "experiments", "results")
EPOCHS, LRS = 25, [3e-4, 1e-3, 3e-3]
BASE_AUG = {"shift": True, "specaug": True}

def go(name, model, lr, seed, aug=None, mk=None, pretrained=None, ckpt=False):
    cfg = {"name": name, "model": model, "lr": lr, "seed": seed, "epochs": EPOCHS, "bs": 32, "wd": 0.01,
           "aug": BASE_AUG if aug is None else aug, "model_kwargs": mk or {}, "save_ckpt": ckpt}
    if pretrained: cfg["pretrained"] = pretrained
    yaml.safe_dump(cfg, open(os.path.join(CFG, name + ".yaml"), "w"))
    if not os.path.exists(os.path.join(RES, name + ".json")):
        subprocess.run([sys.executable, "-m", "src.train", "--config", f"configs/{name}.yaml"], cwd=ROOT, check=True)

def best_lr(prefix):
    r = {lr: json.load(open(os.path.join(RES, f"{prefix}_lr{lr}_s0.json")))["val_macro_f1"] for lr in LRS}
    return max(r, key=r.get)

def study(prefix, model, **kw):
    """Same tuning budget for every model: 3-value LR grid (seed 0, selected on VALIDATION), then seeds 1,2 at the best LR."""
    for lr in LRS: go(f"{prefix}_lr{lr}_s0", model, lr, 0, **kw)
    b = best_lr(prefix)
    for s in (1, 2): go(f"{prefix}_final_s{s}", model, b, s, ckpt=(s == 1 and False), **kw)
    return b

if __name__ == "__main__":
    os.makedirs(CFG, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "main"):
        for p, m in [("cnn", "cnn"), ("ast", "ast"), ("hybrid", "hybrid")]: study(p, m)
        if not os.path.exists(os.path.join(ROOT, "checkpoints", "ast_ssl_pretrained.pt")):
            subprocess.run([sys.executable, "-m", "src.pretrain_ssl"], cwd=ROOT, check=True)
        study("astssl", "ast", pretrained="checkpoints/ast_ssl_pretrained.pt")
    if stage in ("all", "extra"):
        b = best_lr("cnn")                                    # data-strategy study on the CNN (seed 0)
        for s in (0,):
            go(f"data_cnn_none_s{s}", "cnn", b, s, aug={})
            go(f"data_cnn_noise_s{s}", "cnn", b, s, aug={"shift": True, "specaug": True, "noise": True})
            go(f"data_cnn_mixup_s{s}", "cnn", b, s, aug={"shift": True, "specaug": True, "mixup": True})
        bh = best_lr("hybrid")                                # ablations on the hybrid (3 seeds each)
        for s in (0, 1, 2):
            go(f"abl_hybrid_noattn_s{s}", "hybrid", bh, s, mk={"use_attention": False})
            if s == 0: go(f"abl_hybrid_nospecaug_s{s}", "hybrid", bh, s, aug={"shift": True})
