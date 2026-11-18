# Dataset documentation — ESC-50
| Item | Value |
|---|---|
| Name / source | ESC-50: Dataset for Environmental Sound Classification — https://github.com/karolpiczak/ESC-50 (clips from Freesound.org) |
| Version | GitHub `master` (2000 clips; downloaded Nov 2026) |
| Licence | Dataset CC BY-NC 3.0 (clips carry individual CC licences listed in the repo) — used for non-commercial coursework |
| Size / format | 2000 × 5 s, mono, 44.1 kHz WAV, 50 classes × 40 clips (perfectly balanced), 5 official folds |
| Preprocessing | mono → resample 16 kHz → pad/trim to 5 s → peak normalise → stored as float16 array; log-mel (n_fft 1024, hop 640, 64 mels, 20 Hz–8 kHz) computed on the fly; global mean/std normalisation fitted on the training split only |
| Split | train = folds 1-3 (1200), validation = fold 4 (400), test = fold 5 (400). Folds are source-disjoint (clips from one Freesound recording share a fold) → no leakage. Test used once, for final evaluation |
| Noise for robustness | white and pink Gaussian noise added at SNR 20/10/0 dB (white) and 10 dB (pink) to **test** audio with a fixed RNG seed |
