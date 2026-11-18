# Viva preparation (every member must be able to answer ALL of these)
1. **Task / data:** ESC-50, 2000 clips, 50 classes, 5 folds. Why is the fold split leakage-free? (clips from one Freesound recording stay in one fold). Why macro-F1?
2. **Front-end:** how a log-mel is computed (STFT → power → mel filter-bank → log → normalise). Where is the code? (`src/datasets/esc50.py`)
3. **Models:** CNN, AST (patches, [CLS], position embeddings, MHSA formula softmax(QKᵀ/√d)V), ConvTransformer (CNN stem + SE + Transformer + attention pooling).
4. **Fairness:** same split/preprocessing/LR grid/epochs/seeds; selection on validation only; test touched once.
5. **Results (be able to quote):** hybrid 62.9±1.5, CNN 57.9±0.5, SVM 50.9, AST 31.0±2.6, AST+SSL 29.7±2.6 (macro-F1, test).
6. **Why does the Transformer lose?** No locality prior, 1,200 clips, not converged in 25 epochs, no ImageNet pre-training.
7. **SSL pre-training:** masked-patch reconstruction; why it did not help (tiny data, short schedule).
8. **Ablation:** removing Transformer + attention pooling −4.1 points (3 seeds); SpecAugment −2.4 but single seed → inconclusive.
9. **Noise:** why all models collapse; why noise mixing helps (44.8 vs 14.7 at 10 dB, 1 seed); zero-padding caveat.
10. **Errors:** fireworks/clock tick/frog are mostly silence; engine/washing machine/helicopter are stationary textures; only 8 test clips per class.
11. **Limitations:** 1 CPU, 25 epochs, 3 seeds, single-seed studies, no 5-fold CV, synthetic noise.
12. **Demo:** `python demo/app.py`; what the slider does.
