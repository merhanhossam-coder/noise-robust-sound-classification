"""Shared evaluation for ALL models: macro-F1 (primary), accuracy, top-5, per-class F1, bootstrap CI, noise robustness."""
import numpy as np, torch
from sklearn.metrics import f1_score, accuracy_score

def metrics(y, logits):
    p = logits.argmax(1)
    top5 = np.mean([y[i] in np.argsort(-logits[i])[:5] for i in range(len(y))])
    return {"macro_f1": float(f1_score(y, p, average="macro")), "acc": float(accuracy_score(y, p)), "top5": float(top5),
            "per_class_f1": f1_score(y, p, average=None, labels=range(logits.shape[1])).tolist()}

def bootstrap_ci(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); v.append(f1_score(y[i], pred[i], average="macro"))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

@torch.no_grad()
def predict(model, front, wav, bs=50, snr=None, kind="white", seed=123):
    model.eval(); out = []; dev = next(model.parameters()).device
    g = torch.get_rng_state(); torch.default_generator.manual_seed(seed)   # CPU-only seeding: identical noisy test audio for every model, GPU RNG untouched
    for i in range(0, len(wav), bs):
        x = wav[i:i + bs]
        if snr is not None:
            from src.datasets.esc50 import add_noise
            x = add_noise(x, snr, kind)                                    # noise always generated on CPU
        out.append(model(front(x.to(dev))).cpu())
    torch.set_rng_state(g)
    return torch.cat(out).numpy()
