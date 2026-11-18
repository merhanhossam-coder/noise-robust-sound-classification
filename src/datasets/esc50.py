"""ESC-50 loading, leakage-free split, waveform augmentation and a hand-written log-mel front-end."""
import os, math
import numpy as np, pandas as pd, torch, torch.nn as nn

PROC = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
SR, N_FFT, HOP, N_MELS = 16000, 1024, 640, 64   # 5 s -> 126 frames
TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS = [1, 2, 3], [4], [5]   # official folds keep clips of one Freesound source together


def load_split():
    wav = np.load(os.path.join(PROC, "waveforms.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(PROC, "meta.csv"))
    out = {}
    for name, folds in [("train", TRAIN_FOLDS), ("val", VAL_FOLDS), ("test", TEST_FOLDS)]:
        idx = np.where(meta.fold.isin(folds))[0]
        out[name] = (torch.from_numpy(np.asarray(wav[idx], dtype=np.float32)),
                     torch.from_numpy(meta.target.values[idx]).long(), idx)
    classes = meta.drop_duplicates("target").sort_values("target").category.tolist()
    return out, classes


# ---------- hand-written mel filter-bank + log-mel (key component #1) ----------
def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)

def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=20.0, fmax=None):
    fmax = fmax or sr / 2
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = pts[m], pts[m + 1], pts[m + 2]
        up, down = (bins - l) / (c - l), (r - bins) / (r - c)
        fb[m] = np.maximum(0, np.minimum(up, down))
    fb /= fb.sum(1, keepdims=True) + 1e-8
    return torch.from_numpy(fb)


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fb", mel_filterbank())
        self.register_buffer("win", torch.hann_window(N_FFT))
        self.register_buffer("mu", torch.zeros(1)); self.register_buffer("sd", torch.ones(1))

    def raw(self, x):                       # x [B, T] -> [B, 1, 64, frames]
        S = torch.stft(x, N_FFT, HOP, window=self.win, center=True, return_complex=True).abs() ** 2
        return torch.log(torch.matmul(self.fb, S) + 1e-6).unsqueeze(1)

    @torch.no_grad()
    def fit_norm(self, x, bs=64):
        v = torch.cat([self.raw(x[i:i + bs]).flatten() for i in range(0, len(x), bs)])
        self.mu.copy_(v.mean()); self.sd.copy_(v.std())

    def forward(self, x): return (self.raw(x) - self.mu) / self.sd


# ---------- waveform / spectrogram augmentation ----------
def add_noise(x, snr_db, kind="white"):
    """Add noise at a given SNR (dB). x: [B, T]. snr_db: float or tensor [B]."""
    n = torch.randn_like(x)
    if kind == "pink":
        F = torch.fft.rfft(n); f = torch.arange(F.shape[-1], dtype=torch.float32).clamp(min=1)
        n = torch.fft.irfft(F / f.sqrt(), n=x.shape[-1])
    snr = torch.as_tensor(snr_db, dtype=torch.float32).expand(x.shape[0]).unsqueeze(1)
    ps, pn = x.pow(2).mean(1, keepdim=True), n.pow(2).mean(1, keepdim=True) + 1e-12
    return x + n * torch.sqrt(ps / (pn * 10 ** (snr / 10)))

def random_shift(x):
    return torch.stack([torch.roll(w, int(np.random.randint(0, w.numel())), 0) for w in x])

def spec_augment(S, n_f=2, max_f=8, n_t=2, max_t=20):
    """SpecAugment (frequency & time masking), written from scratch. S: [B,1,F,T]."""
    S = S.clone(); B, _, F, T = S.shape
    for b in range(B):
        for _ in range(n_f):
            w = np.random.randint(0, max_f + 1); s = np.random.randint(0, F - w + 1); S[b, :, s:s + w] = 0
        for _ in range(n_t):
            w = np.random.randint(0, max_t + 1); s = np.random.randint(0, T - w + 1); S[b, :, :, s:s + w] = 0
    return S

def mixup(S, y, n_cls, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha)); perm = torch.randperm(S.shape[0])
    Y = torch.nn.functional.one_hot(y, n_cls).float()
    return lam * S + (1 - lam) * S[perm], lam * Y + (1 - lam) * Y[perm]
