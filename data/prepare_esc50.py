"""Download ESC-50 (GitHub release of the dataset) and convert it to a compact array.

Output (data/processed/):
    waveforms.npy  float16 array [2000, 80000]  (5 s @ 16 kHz, peak-normalised)
    meta.csv       filename, fold, target, category

Usage:  python data/prepare_esc50.py
"""
import io, os, sys, zipfile, urllib.request
import numpy as np, pandas as pd, librosa, soundfile as sf

URL = "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master"
OUT = os.path.join(os.path.dirname(__file__), "processed")
SR, DUR = 16000, 5

def main(zip_path=None):
    os.makedirs(OUT, exist_ok=True)
    if zip_path is None:
        zip_path = os.path.join(OUT, "esc50.zip")
        if not os.path.exists(zip_path):
            print("Downloading ESC-50 (~600 MB)...")
            urllib.request.urlretrieve(URL, zip_path)
    z = zipfile.ZipFile(zip_path)
    meta = pd.read_csv(io.BytesIO(z.read("ESC-50-master/meta/esc50.csv")))
    wav = np.zeros((len(meta), SR * DUR), dtype=np.float16)
    for i, fn in enumerate(meta.filename):
        x, sr = sf.read(io.BytesIO(z.read(f"ESC-50-master/audio/{fn}")), dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)[: SR * DUR]
        x = np.pad(x, (0, SR * DUR - len(x)))
        wav[i] = (x / (np.abs(x).max() + 1e-8)).astype(np.float16)
        if i % 400 == 0: print(i, "/", len(meta))
    np.save(os.path.join(OUT, "waveforms.npy"), wav)
    meta[["filename", "fold", "target", "category"]].to_csv(os.path.join(OUT, "meta.csv"), index=False)
    print("done", wav.shape)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
