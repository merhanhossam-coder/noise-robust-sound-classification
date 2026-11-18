"""Collect all result JSONs -> tables (CSV/markdown), figures, error analysis."""
import os, sys, json, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import Counter
ROOT = os.path.join(os.path.dirname(__file__), ".."); sys.path.insert(0, ROOT)
from sklearn.metrics import confusion_matrix
R, F = os.path.join(ROOT, "experiments", "results"), os.path.join(ROOT, "experiments", "figures")
T = os.path.join(ROOT, "experiments", "tables"); os.makedirs(F, exist_ok=True); os.makedirs(T, exist_ok=True)
J = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(R + "/*.json")}
meta = pd.read_csv(os.path.join(ROOT, "data", "processed", "meta.csv")); classes = meta.drop_duplicates("target").sort_values("target").category.tolist()
yte = meta[meta.fold == 5].target.values; NAMES = {"baseline": "Baseline (MFCC+SVM)", "cnn": "CNN (classic)", "ast": "AST scratch (modern)",
       "astssl": "AST + SSL pre-train (modern)", "hybrid": "ConvTransformer (advanced)"}
def finals(prefix):
    if prefix == "baseline": return [J[k] for k in sorted(J) if k.startswith("baseline_s")]
    runs = [J[k] for k in sorted(J) if k.startswith(prefix + "_final_s")]
    best = max([J[k] for k in J if k.startswith(prefix + "_lr") and k.endswith("_s0")], key=lambda r: r["val_macro_f1"])
    return [best] + runs
def ms(v): return f"{np.mean(v)*100:.1f} ± {np.std(v, ddof=0)*100:.1f}"

# ---- Table 4 main results, Table 7 cost, robustness ----
rows4, rows7, rowsR, L = [], [], [], {}
for p, n in NAMES.items():
    rs = finals(p); L[p] = rs
    rows4.append({"Model": n, "Macro-F1 (%)": ms([r["test"]["macro_f1"] for r in rs]), "Accuracy (%)": ms([r["test"]["acc"] for r in rs]),
                  "Top-5 (%)": ms([r["test"]["top5"] for r in rs]), "Bootstrap 95% CI macro-F1 (seed 0)": f"[{rs[0]['test']['ci95_macro_f1'][0]*100:.1f}, {rs[0]['test']['ci95_macro_f1'][1]*100:.1f}]",
                  "Val macro-F1 (%)": ms([r["val_macro_f1"] for r in rs]), "LR": rs[0]["cfg"].get("lr", rs[0]["cfg"].get("classifier")), "n seeds": len(rs)})
    rows7.append({"Model": n, "Params": int(rs[0]["params"]), "Size (MB)": round(rs[0]["size_mb"], 2), "Train time (s)": round(np.mean([r["train_time_s"] for r in rs])),
                  "Inference (ms/clip)": round(np.mean([r["inference_ms_per_clip"] for r in rs]), 2), "Hardware": rs[0]["hardware"]})
    rowsR.append({"Model": n, "Clean": ms([r["test"]["macro_f1"] for r in rs])} | {k.replace("_", " "): ms([r["robustness_macro_f1"][k] for r in rs]) for k in ["white_20dB", "white_10dB", "white_0dB", "pink_10dB"]})
for nm, rr in [("table4_main_results", rows4), ("table7_cost", rows7), ("table_robustness", rowsR)]:
    pd.DataFrame(rr).to_csv(f"{T}/{nm}.csv", index=False); open(f"{T}/{nm}.md", "w").write(pd.DataFrame(rr).to_markdown(index=False))

# ---- hyper-parameter study (val F1 vs LR) ----
hp = []
for p in ["cnn", "ast", "astssl", "hybrid"]:
    for k in sorted(J):
        if k.startswith(p + "_lr") and k.endswith("_s0"): hp.append({"Model": NAMES[p], "LR": J[k]["cfg"]["lr"], "Val macro-F1 (%)": round(J[k]["val_macro_f1"] * 100, 1)})
pd.DataFrame(hp).to_csv(f"{T}/hparam_study.csv", index=False)
plt.figure(figsize=(5.5, 3.6))
for n, g in pd.DataFrame(hp).groupby("Model"): plt.plot(g.LR, g["Val macro-F1 (%)"], "o-", label=n)
plt.xscale("log"); plt.xlabel("learning rate"); plt.ylabel("validation macro-F1 (%)"); plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{F}/hparam_lr.png", dpi=200); plt.close()

# ---- data strategy (CNN, seed 0) ----
base0 = next(r for r in L["cnn"] if r["cfg"]["seed"] == 0)
ds = [("No augmentation", J.get("data_cnn_none_s0")), ("Shift + SpecAugment (default)", base0), ("+ noise mixing", J.get("data_cnn_noise_s0")), ("+ mixup", J.get("data_cnn_mixup_s0"))]
dsr = [{"Strategy": n, "Val F1 (%)": round(r["val_macro_f1"] * 100, 1), "Test macro-F1 (%)": round(r["test"]["macro_f1"] * 100, 1), "Test F1 @ white 10 dB (%)": round(r["robustness_macro_f1"]["white_10dB"] * 100, 1)} for n, r in ds if r]
pd.DataFrame(dsr).to_csv(f"{T}/data_strategy.csv", index=False)

# ---- ablation (hybrid) ----
full = [r["test"]["macro_f1"] for r in L["hybrid"]]
abl = [{"Variant": "Full ConvTransformer", "Macro-F1 (%)": ms(full), "Δ vs full (pts)": "—", "n seeds": len(full)}]
for key, nm in [("abl_hybrid_noattn", "− Transformer + attention pooling (mean pool)"), ("abl_hybrid_nospecaug", "− SpecAugment")]:
    rs = [J[k] for k in sorted(J) if k.startswith(key)]
    if rs: v = [r["test"]["macro_f1"] for r in rs]; abl.append({"Variant": nm, "Macro-F1 (%)": ms(v), "Δ vs full (pts)": f"{(np.mean(v)-np.mean(full))*100:+.1f}", "n seeds": len(v)})
a, s = L["ast"], L["astssl"]; abl.append({"Variant": "AST: − SSL pre-training (scratch)", "Macro-F1 (%)": ms([r["test"]["macro_f1"] for r in a]), "Δ vs full (pts)": f"{(np.mean([r['test']['macro_f1'] for r in a])-np.mean([r['test']['macro_f1'] for r in s]))*100:+.1f} (ref: AST+SSL {ms([r['test']['macro_f1'] for r in s])})", "n seeds": len(a)})
pd.DataFrame(abl).to_csv(f"{T}/table5_ablation.csv", index=False); open(f"{T}/table5_ablation.md", "w").write(pd.DataFrame(abl).to_markdown(index=False))

# ---- figures: curves, robustness, confusion, per-class ----
plt.figure(figsize=(5.5, 3.6))
for p, n in list(NAMES.items())[1:]:
    h = np.array([[e["val_f1"] for e in r["history"]] for r in L[p]]); plt.plot(h.mean(0) * 100, label=n)
plt.xlabel("epoch"); plt.ylabel("validation macro-F1 (%)"); plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{F}/val_curves.png", dpi=200); plt.close()
plt.figure(figsize=(5.5, 3.6)); xs = ["clean", "20 dB", "10 dB", "0 dB"]
for p, n in NAMES.items():
    v = [[r["test"]["macro_f1"] for r in L[p]]] + [[r["robustness_macro_f1"][f"white_{s}dB"] for r in L[p]] for s in (20, 10, 0)]
    plt.errorbar(range(4), [np.mean(x) * 100 for x in v], [np.std(x) * 100 for x in v], marker="o", capsize=2, label=n)
plt.xticks(range(4), xs); plt.xlabel("white-noise SNR (test set)"); plt.ylabel("macro-F1 (%)"); plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{F}/robustness.png", dpi=200); plt.close()
best_p = max(["cnn", "ast", "astssl", "hybrid"], key=lambda p: np.mean([r["test"]["macro_f1"] for r in L[p]]))
def logits(r): return np.load(f"{R}/{r['name']}_logits.npy")
pred = logits(L[best_p][0]).argmax(1); cm = confusion_matrix(yte, pred, labels=range(50))
plt.figure(figsize=(8, 7)); plt.imshow(cm, cmap="Blues"); plt.colorbar(shrink=.7); plt.xticks(range(50), classes, rotation=90, fontsize=5); plt.yticks(range(50), classes, fontsize=5)
plt.title(f"Confusion matrix – {NAMES[best_p]} (test, seed 0)"); plt.tight_layout(); plt.savefig(f"{F}/confusion_best.png", dpi=200); plt.close()
pc = np.mean([r["test"]["per_class_f1"] for r in L[best_p]], 0); o = np.argsort(pc)
plt.figure(figsize=(8, 3.4)); plt.bar(range(50), pc[o]); plt.xticks(range(50), [classes[i] for i in o], rotation=90, fontsize=5); plt.ylabel("F1"); plt.title(f"Per-class F1 – {NAMES[best_p]} (mean of seeds)"); plt.tight_layout(); plt.savefig(f"{F}/per_class_f1.png", dpi=200); plt.close()
pd.DataFrame({"class": classes, **{p: np.mean([r["test"]["per_class_f1"] for r in L[p]], 0).round(3) for p in NAMES}}).to_csv(f"{T}/per_class_f1_all_models.csv", index=False)

# ---- error analysis (Table 6) ----
pairs = Counter((classes[a], classes[b]) for a, b in zip(yte, pred) if a != b); errs = int((pred != yte).sum())
PHYS = {frozenset(["rain", "sea_waves"]): "broadband noise-like textures with similar spectra", frozenset(["rain", "pouring_water"]): "water sounds share broadband energy",
        frozenset(["wind", "vacuum_cleaner"]): "stationary broadband noise", frozenset(["engine", "airplane"]): "low-frequency stationary drone",
        frozenset(["mouse_click", "clock_tick"]): "short periodic clicks", frozenset(["keyboard_typing", "clock_tick"]): "short periodic clicks",
        frozenset(["crying_baby", "cat"]): "similar harmonic, wailing timbre", frozenset(["door_wood_knock", "mouse_click"]): "short impulsive events",
        frozenset(["footsteps", "door_wood_knock"]): "impulsive low-frequency transients", frozenset(["hen", "rooster"]): "same species family, overlapping vocalisation",
        frozenset(["church_bells", "clock_alarm"]): "tonal ringing sounds", frozenset(["helicopter", "chainsaw"]): "periodic modulated engine noise"}
rows6 = []
for (a, b), c in pairs.most_common(8):
    ex = int(np.where((yte == classes.index(a)) & (pred == classes.index(b)))[0][0]); fn = meta[meta.fold == 5].filename.values[ex]
    rows6.append({"Error type (true → predicted)": f"{a} → {b}", "Count": c, "Example (test clip)": fn, "Likely cause": PHYS.get(frozenset([a, b]), "overlapping spectro-temporal structure (to be verified by listening/spectrogram inspection)")})
pd.DataFrame(rows6).to_csv(f"{T}/table6_error_analysis.csv", index=False); open(f"{T}/table6_error_analysis.md", "w").write(pd.DataFrame(rows6).to_markdown(index=False))
# error vs noise & low-confidence stats
lg = logits(L[best_p][0]); pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True); conf = pr.max(1)
stats = {"best_model": best_p, "n_test": len(yte), "n_errors": errs, "mean_conf_correct": float(conf[pred == yte].mean()), "mean_conf_wrong": float(conf[pred != yte].mean()),
         "worst_classes": [(classes[i], float(pc[i])) for i in o[:8]], "best_classes": [(classes[i], float(pc[i])) for i in o[-5:]]}
json.dump(stats, open(f"{T}/error_stats.json", "w"), indent=1)
# example spectrograms of top confusion
WAV = os.path.join(ROOT, "data", "processed", "waveforms.npy")
if os.path.exists(WAV):   # needs the audio; otherwise the figure shipped in the repo is kept
    import torch; from src.datasets.esc50 import LogMel
    wav = np.load(WAV, mmap_mode="r"); idx5 = np.where(meta.fold == 5)[0]; fr = LogMel()
    fig, ax = plt.subplots(2, 3, figsize=(9, 4.2))
    for j, ((a, b), c) in enumerate(pairs.most_common(3)):
        ex = int(np.where((yte == classes.index(a)) & (pred == classes.index(b)))[0][0]); ref = int(np.where(yte == classes.index(b))[0][0])
        for r_, (i, t) in enumerate([(ex, f"TRUE {a} (predicted {b})"), (ref, f"TRUE {b}")]):
            ax[r_, j].imshow(fr.raw(torch.from_numpy(np.asarray(wav[idx5[i]], dtype=np.float32))[None])[0, 0].numpy(), origin="lower", aspect="auto"); ax[r_, j].set_title(t, fontsize=7); ax[r_, j].axis("off")
    plt.tight_layout(); plt.savefig(f"{F}/error_examples.png", dpi=200); plt.close()
print(pd.DataFrame(rows4).to_string()); print(pd.DataFrame(abl).to_string()); print(pd.DataFrame(dsr).to_string()); print(json.dumps(stats, indent=1))
