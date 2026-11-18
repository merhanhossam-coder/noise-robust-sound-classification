"""Train / fine-tune one model from a YAML config.   python -m src.train --config configs/xxx.yaml"""
import argparse, json, os, time, yaml, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from src.datasets.esc50 import (load_split, LogMel, add_noise, random_shift, spec_augment, mixup)
from src.models.networks import build
from src.evaluate import metrics, predict, bootstrap_ci

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cpu": torch.set_num_threads(1)
HW = ("GPU: " + torch.cuda.get_device_name(0)) if DEV.type == "cuda" else f"CPU ({os.cpu_count()} cores, no GPU)"

def run(cfg):
    seed = cfg["seed"]; torch.manual_seed(seed); np.random.seed(seed)
    if DEV.type == "cuda": torch.cuda.manual_seed_all(seed)
    data, classes = load_split(); n_cls = len(classes)
    Xtr, ytr, _ = data["train"]; Xva, yva, _ = data["val"]; Xte, yte, _ = data["test"]
    front = LogMel(); front.fit_norm(Xtr); front = front.to(DEV)
    model = build(cfg["model"], n_cls=n_cls, **cfg.get("model_kwargs", {}))
    if cfg.get("pretrained"):
        sd = torch.load(os.path.join(ROOT, cfg["pretrained"]), map_location="cpu")
        print("loaded pretrained:", model.load_state_dict(sd, strict=False))
    model = model.to(DEV)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.01))
    E, bs = cfg["epochs"], cfg.get("bs", 32); steps = E * int(np.ceil(len(Xtr) / bs)); warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / (steps - warm))))
    aug = cfg.get("aug", {}); best, best_state, hist = -1, None, []
    t0 = time.time()
    for ep in range(E):
        model.train(); perm = torch.randperm(len(Xtr)); tl = 0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]; x, y = Xtr[idx], ytr[idx]
            if aug.get("shift"): x = random_shift(x)
            if aug.get("noise"): x = add_noise(x, torch.empty(len(x)).uniform_(5, 30), np.random.choice(["white", "pink"])) if np.random.rand() < 0.5 else x
            S = front(x.to(DEV))
            if aug.get("specaug"): S = spec_augment(S)
            if aug.get("mixup"): S, Y = mixup(S, y, n_cls)
            else: Y = F.one_hot(y, n_cls).float()
            Y = Y * 0.9 + 0.1 / n_cls                                   # label smoothing
            loss = -(Y.to(DEV) * F.log_softmax(model(S), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
        v = metrics(yva.numpy(), predict(model, front, Xva))
        hist.append({"epoch": ep, "loss": tl / len(Xtr), "val_f1": v["macro_f1"], "val_acc": v["acc"]})
        if v["macro_f1"] > best: best = v["macro_f1"]; best_state = {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 5 == 0 or ep == E - 1: print(f"[{cfg['name']}] ep{ep} loss {tl/len(Xtr):.3f} val_f1 {v['macro_f1']:.3f}", flush=True)
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    # ---- final test evaluation (test set touched only here) ----
    lt = predict(model, front, Xte); res = metrics(yte.numpy(), lt); val = metrics(yva.numpy(), predict(model, front, Xva))
    res["ci95_macro_f1"] = bootstrap_ci(yte.numpy(), lt.argmax(1))
    t1 = time.time(); predict(model, front, Xte[:100]); inf_ms = (time.time() - t1) / 100 * 1000
    rob = {}
    for snr in [20, 10, 0]:
        rob[f"white_{snr}dB"] = metrics(yte.numpy(), predict(model, front, Xte, snr=snr))["macro_f1"]
    rob["pink_10dB"] = metrics(yte.numpy(), predict(model, front, Xte, snr=10, kind="pink"))["macro_f1"]
    out = {"name": cfg["name"], "cfg": cfg, "val_macro_f1": val["macro_f1"], "test": res, "robustness_macro_f1": rob,
           "params": params, "size_mb": params * 4 / 1e6, "train_time_s": train_time, "inference_ms_per_clip": inf_ms,
           "best_epoch": int(np.argmax([h["val_f1"] for h in hist])), "history": hist, "hardware": HW}
    os.makedirs(os.path.join(ROOT, "experiments", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "experiments", "results", cfg["name"] + ".json"), "w"))
    np.save(os.path.join(ROOT, "experiments", "results", cfg["name"] + "_logits.npy"), lt)
    if cfg.get("save_ckpt"):
        os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
        torch.save({"state": best_state, "cfg": cfg, "mu": front.mu, "sd": front.sd, "classes": classes}, os.path.join(ROOT, "checkpoints", cfg["name"] + ".pt"))
    print(f"[{cfg['name']}] DONE val_f1 {val['macro_f1']:.3f} test_f1 {res['macro_f1']:.3f} time {train_time:.0f}s", flush=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    run(yaml.safe_load(open(ap.parse_args().config)))
