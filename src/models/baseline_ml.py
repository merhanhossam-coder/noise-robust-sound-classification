"""Classical-ML baseline: clip-level statistics of log-mel / MFCC + Random Forest or RBF-SVM (same split & metrics)."""
import json, os, time, sys, numpy as np, torch
from scipy.fft import dct
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.datasets.esc50 import load_split, LogMel, add_noise
from src.evaluate import metrics, bootstrap_ci
torch.set_num_threads(1)
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

def feats(front, X, snr=None, kind="white", bs=50):
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs]
        if snr is not None: x = add_noise(x, snr, kind)
        S = front.raw(x).squeeze(1)                              # [B, 64, T] raw log-mel
        mf = torch.from_numpy(dct(S.numpy(), axis=1, norm="ortho")[:, :20])          # 20 MFCC
        d = mf[:, :, 1:] - mf[:, :, :-1]
        out.append(torch.cat([S.mean(2), S.std(2), mf.mean(2), mf.std(2), d.abs().mean(2)], 1))
    return torch.cat(out).numpy()

def main():
    data, classes = load_split(); Xtr, ytr, _ = data["train"]; Xva, yva, _ = data["val"]; Xte, yte, _ = data["test"]
    front = LogMel(); Ftr, Fva, Fte = feats(front, Xtr), feats(front, Xva), feats(front, Xte)
    Fnoise = {f"white_{s}dB": feats(front, Xte, s) for s in (20, 10, 0)}; Fnoise["pink_10dB"] = feats(front, Xte, 10, "pink")
    cands = {"svm_C1": lambda s: make_pipeline(StandardScaler(), SVC(C=1, probability=False, random_state=s)),
             "svm_C10": lambda s: make_pipeline(StandardScaler(), SVC(C=10, random_state=s)),
             "svm_C100": lambda s: make_pipeline(StandardScaler(), SVC(C=100, random_state=s))}   # same 3-value tuning budget
    val = {k: metrics(yva.numpy(), np.eye(50)[f(0).fit(Ftr, ytr.numpy()).predict(Fva)])["macro_f1"] for k, f in cands.items()}
    best = max(val, key=val.get); print("baseline val:", val, "->", best)
    for seed in range(3):
        t = time.time(); m = cands[best](seed).fit(Ftr, ytr.numpy()); tt = time.time() - t
        t = time.time(); p = m.predict(Fte); inf = (time.time() - t) / len(Fte) * 1000
        L = np.eye(50)[p]; res = metrics(yte.numpy(), L); res["ci95_macro_f1"] = bootstrap_ci(yte.numpy(), p)
        rob = {k: metrics(yte.numpy(), np.eye(50)[m.predict(F)])["macro_f1"] for k, F in Fnoise.items()}
        n_sv = int(sum(m[-1].n_support_)); 
        out = {"name": f"baseline_s{seed}", "model": "baseline", "cfg": {"features": "log-mel+MFCC clip statistics (188-d)", "classifier": best},
               "val_macro_f1": val[best], "test": res, "robustness_macro_f1": rob, "params": n_sv * Ftr.shape[1], "size_mb": n_sv * Ftr.shape[1] * 8 / 1e6,
               "train_time_s": tt, "inference_ms_per_clip": inf, "hardware": "1 vCPU (no GPU), 3 GB RAM", "history": []}
        json.dump(out, open(os.path.join(ROOT, "experiments", "results", f"baseline_s{seed}.json"), "w"))
        np.save(os.path.join(ROOT, "experiments", "results", f"baseline_s{seed}_logits.npy"), L)
        print(seed, res["macro_f1"], rob)

if __name__ == "__main__": main()
