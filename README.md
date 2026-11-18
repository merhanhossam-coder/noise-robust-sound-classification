# Noise-Robust Environmental Sound Classification (ESC-50) — AI403 Deep Learning Final Project

**Team:** Team_ESC50 — Merhan Hossam Mohamed (202301681), Amira Samy Ebrahim (202301561), Gana Mohamed Ragab (202303206), Omnya Aly Mohamed (202303221)  **Idea chosen:** #17 *Environmental sound classification under noise* (Audio → class)
**Course:** Deep Learning AI403, Pharos University in Alexandria, Fall 2026 (Dr. Sahar Ghanem)

## 1. Task and best result
Classify 5-second clips into 50 environmental-sound classes (ESC-50) and measure how well each model degrades when
white / pink noise is added at test time. Primary metric: **macro-F1** (balanced classes, so it equals per-class average);
also accuracy, top-5, per-class F1, bootstrap CI and seed variance. Test macro-F1 (mean ± SD, 3 seeds, CPU run): **ConvTransformer 62.9 ± 1.5**, CNN 57.9 ± 0.5 (a GPU re-run gave 62.0 ± 0.1 vs 61.5 ± 1.2, see `experiments/colab_replication.md`), MFCC+SVM 50.9, AST 31.0 ± 2.6, AST+SSL 29.7 ± 2.6. Full tables: `experiments/tables/table4_main_results.md`
(mean ± SD over 3 seeds) and the report.

| Family | Model | File |
|---|---|---|
| Baseline | log-mel + MFCC clip statistics → RBF-SVM | `src/models/baseline_ml.py` |
| Classic | 4-block CNN | `src/models/networks.py::SmallCNN` |
| Modern | small Audio-Spectrogram-Transformer (scratch) and the same AST with **self-supervised masked-spectrogram pre-training** | `networks.py::AST`, `src/pretrain_ssl.py` |
| Advanced | ConvTransformer (CNN stem + squeeze-excitation → Transformer encoder → attention pooling) | `networks.py::ConvTransformer` |

Components written from scratch: mel filter-bank & log-mel front-end (`src/datasets/esc50.py`), SpecAugment, mixup,
noise-at-SNR augmentation, multi-head self-attention / Transformer block / attention pooling (`src/models/attention.py`),
masked-spectrogram pre-training loss, training loop, metrics wrapper.

## 2. Environment
```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
```
## 3. Dataset download + preprocessing
```bash
python data/prepare_esc50.py        # downloads ESC-50 from GitHub (CC BY-NC 3.0, ~600 MB), resamples to 16 kHz -> data/processed/
```
Split (leakage-free): official ESC-50 folds — **train = folds 1-3, validation = fold 4, test = fold 5**. ESC-50 folds keep all clips
cut from the same Freesound recording in one fold, so no source leaks across splits. The test set is only used in the final
evaluation inside `train.py` after the best-validation epoch has been chosen. See `data/DATASET.md`.

## 4. Training commands (one config per experiment in `configs/`)
```bash
python -m src.models.baseline_ml                      # baseline (3 seeds)
python -m src.train --config configs/cnn_lr0.001_s0.yaml
python -m src.pretrain_ssl                            # SSL pre-training for AST (labels unused)
python -m src.train --config configs/astssl_final_s1.yaml
python scripts/run_all.py all                         # EVERYTHING (resumable): tuning grid, 3-seed finals, data-strategy, ablations
```
## 5. Evaluation
`src/evaluate.py` is shared by every model. `python scripts/aggregate.py` builds all tables/figures into
`experiments/tables/` and `experiments/figures/` (including error analysis).
## 6. Demo
```bash
python scripts/export_demo_checkpoint.py   # saves checkpoints/hybrid_demo.pt (or download from the drive link below)
python demo/app.py                         # Gradio UI: upload/record audio, choose noise SNR, see top-5 classes
```
## 6b. Running in Google Colab
Open `notebooks/colab_run.ipynb` in Colab (Runtime → T4 GPU), upload `project_repo.zip` (or clone your GitHub repo) and follow the cells: *Mode A* = rebuild tables + launch the demo from the shipped results; *Mode B* = full retraining (GPU used automatically, resumable).

## 7. Hardware and runtimes
The reported results were produced on **1 vCPU, 3 GB RAM, no GPU**; the code now also uses a GPU when available (untested on GPU by the authors of this package). One 25-epoch run takes ~3-5 min; the full study ≈ 2-3 h.
Per-model training/inference time is in `experiments/tables/table7_cost.md`.
## 8. Credits
* ESC-50: K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM MM 2015 (CC BY-NC 3.0).
* Ideas adapted (re-implemented, no pretrained weights, no external AI APIs): SpecAugment (Park et al., 2019), mixup (Zhang et al., 2018),
  AST (Gong et al., 2021), masked autoencoders (He et al., 2022), Squeeze-and-Excitation (Hu et al., 2018), Conformer-style CNN+Transformer hybrids (Gulati et al., 2020).
* Libraries: PyTorch, librosa (resampling only), scikit-learn, scipy, matplotlib, Gradio.
* Checkpoints / large files: *Not needed: small checkpoints are included in checkpoints/; the dataset is downloaded by data/prepare_esc50.py.*

## Layout
```
configs/ data/ src/{datasets,models,train.py,evaluate.py,pretrain_ssl.py} scripts/ experiments/{results,tables,figures}
checkpoints/ demo/ report/ contributions/
```
