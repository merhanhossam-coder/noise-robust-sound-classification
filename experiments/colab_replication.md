# Independent replication on a Google Colab Tesla T4 GPU (same code and configs)
Test macro-F1 (%), mean ± SD over 3 seeds. Primary results (experiments/results/) come from the CPU runs; this re-run is reported in Section 13.4 of the report.
| Model | CPU run | GPU re-run |
|---|---|---|
| MFCC + SVM | 50.9 ± 0.0 | 50.9 ± 0.0 |
| CNN | 57.9 ± 0.5 | 61.5 ± 1.2 |
| AST scratch | 31.0 ± 2.6 | 31.1 ± 2.2 |
| AST + SSL | 29.7 ± 2.6 | 30.4 ± 2.1 |
| ConvTransformer | 62.9 ± 1.5 | 62.0 ± 0.1 |
Ablation (hybrid without Transformer/attention pooling): 58.8 ± 1.0 (CPU) vs 58.5 ± 1.0 (GPU). Source: output of notebooks/colab_run.ipynb (Deep_Learning_Project.ipynb).
