"""Build the technical report PDF from experiments/tables + figures (numbers come from the CSVs)."""
import os, sys, json, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether
ROOT = os.path.join(os.path.dirname(__file__), ".."); T = ROOT + "/experiments/tables/"; F = ROOT + "/experiments/figures/"
TEAM = os.environ.get("TEAM", "TeamName")
ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=10.8, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, spaceBefore=12, spaceAfter=7, textColor=colors.HexColor("#1F3A5F"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1F3A5F"))
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.8, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#333333"), spaceAfter=10)
CELL = ParagraphStyle("CELL", parent=B, fontSize=8, leading=9.6, alignment=0, spaceAfter=0)
CELLH = ParagraphStyle("CELLH", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
S = []; ntab = [0]; nfig = [0]
LABELS = {"Dataset comparison": "1", "Model architecture": "2", "Hyperparameter configuration": "3", "Main experimental": "4", "Ablation study (required": "5", "Error analysis (required": "6", "Computational cost": "7", "Test macro-F1 (%) under additive": "R1", "Replication on a Colab": "R4", "Replication of the ablation": "R5", "Data-strategy study with": "R2", "Verdict on each": "R3", "All tuning runs": "A1"}
def P(t): S.append(Paragraph(t, B))
def h1(t): S.append(Paragraph(t, H1))
def h2(t): S.append(Paragraph(t, H2))
def bullets(items): [S.append(Paragraph("&bull; " + i, ParagraphStyle("bl", parent=B, leftIndent=14, firstLineIndent=-9, spaceAfter=3))) for i in items]
def table(df_or_rows, caption, widths=None, header=True):
    lab = next((v for k, v in LABELS.items() if caption.startswith(k)), None)
    if lab is None: ntab[0] += 1; lab = str(ntab[0])
    rows = [list(df_or_rows.columns)] + df_or_rows.astype(str).values.tolist() if hasattr(df_or_rows, "columns") else df_or_rows
    data = [[Paragraph(str(c), CELLH if (i == 0 and header) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F7")]), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    S.append(KeepTogether([Paragraph(f"<b>Table {lab}.</b> {caption}", CAP), t, Spacer(1, 8)]))
def fig(name, caption, w=13):
    nfig[0] += 1; from PIL import Image as PI
    iw, ih = PI.open(F + name).size
    S.append(KeepTogether([Image(F + name, width=w * cm, height=w * cm * ih / iw), Paragraph(f"<b>Figure {nfig[0]}.</b> {caption}", CAP)]))
rd = lambda n: pd.read_csv(T + n)
main, rob, cost, abl, data, hp, err = rd("table4_main_results.csv"), rd("table_robustness.csv"), rd("table7_cost.csv"), rd("table5_ablation.csv"), rd("data_strategy.csv"), rd("hparam_study.csv"), rd("table6_error_analysis.csv")
st = json.load(open(T + "error_stats.json")); W = 17 * cm

# ---------------- title page ----------------
S.append(Spacer(1, 3 * cm))
S.append(Paragraph("Pharos University in Alexandria<br/>Faculty of Computer Science &amp; Artificial Intelligence – AI Department<br/>Deep Learning (AI403) – Fall 2026 – Dr. Sahar Ghanem", ParagraphStyle("t0", parent=B, alignment=TA_CENTER, fontSize=11, leading=16)))
S.append(Spacer(1, 2 * cm))
S.append(Paragraph("Noise-Robust Environmental Sound Classification on ESC-50", ParagraphStyle("t1", parent=H1, fontSize=24, leading=29, alignment=TA_CENTER)))
S.append(Paragraph("A controlled comparison of a classical baseline, a CNN, an Audio Spectrogram Transformer (with and without self-supervised pre-training) and a CNN–Transformer hybrid", ParagraphStyle("t2", parent=B, alignment=TA_CENTER, fontSize=12, leading=16)))
S.append(Spacer(1, 2 * cm))
S.append(Paragraph(f"<b>Project idea #17</b> – Environmental sound classification under noise (Audio → class)<br/><br/><b>Team:</b> {TEAM}<br/><b>Members (name – student ID):</b><br/>Merhan Hossam Mohamed – 202301681<br/>Amira Samy Ebrahim – 202301561<br/>Gana Mohamed Ragab – 202303206<br/>Omnya Aly Mohamed – 202303221<br/><br/><b>Submission:</b> 20/11/2026", ParagraphStyle("t3", parent=B, alignment=TA_CENTER, fontSize=11, leading=16)))
S.append(PageBreak())


h1("Table of Contents")
for i, n in enumerate(["Abstract","Introduction","Problem Statement","Motivation","Related Work","Dataset","Data Preprocessing","Methodology","Baseline","Proposed / Advanced Models","Experimental Setup","Hyperparameters","Results","Model Comparison","Ablation Study","Error Analysis","Discussion","Limitations","Ethical Considerations","Conclusion","Future Work","References"], 1):
    S.append(Paragraph(f"{i}. {n}", ParagraphStyle("toc", parent=B, alignment=0, spaceAfter=2, leftIndent=10)))
S.append(Paragraph("Appendix A. Learning-rate study &nbsp;&nbsp; Appendix B. Reproducibility", ParagraphStyle("toc2", parent=B, alignment=0, leftIndent=10)))
S.append(Spacer(1, 10))
S.append(Paragraph("<b>Required tables:</b> Table 1 dataset comparison (§6) · Table 2 model architectures (§10) · Table 3 hyperparameters (§12) · Table 4 main results (§13) · Table 5 ablation (§15) · Table 6 error analysis (§16) · Table 7 computational cost (§17).", B))
S.append(PageBreak())
# ---------------- 1 Abstract ----------------
h1("1. Abstract")
P("We study environmental sound classification on ESC-50 (2,000 five-second clips, 50 classes) with an emphasis on <b>robustness to additive noise</b>. "
  "Under one fixed protocol (source-disjoint train/validation/test folds, identical preprocessing, an identical three-value learning-rate grid, 25 training epochs and three random seeds) we compare a classical MFCC/log-mel + SVM baseline, a four-block CNN, "
  "a small Audio Spectrogram Transformer (AST) trained from scratch, the same AST after self-supervised masked-spectrogram pre-training, and an advanced CNN–Transformer hybrid with squeeze-and-excitation and attention pooling. "
  f"On the held-out test fold the hybrid reaches {main.iloc[4,1]} % macro-F1 (mean ± SD over three seeds), the CNN {main.iloc[1,1]} %, the baseline {main.iloc[0,1]} %, and the AST only {main.iloc[2,1]} % (scratch) and {main.iloc[3,1]} % (pre-trained). "
  "Removing the Transformer/attention-pooling stage of the hybrid costs 4.1 points. Self-supervised pre-training did <i>not</i> help the AST in our small-data, low-compute regime. An independent re-run of the whole suite on a Colab T4 GPU reproduced these findings (AST far behind, no pre-training gain, attention ablation −3.5 points) but shrank the hybrid's lead over the CNN from 5.0 to 0.5 points (62.0 vs 61.5 %), so that lead is not established. "
  "All models degrade sharply when white noise is added at test time (the hybrid falls from 62.9 % to 21.6 % at 10 dB SNR), but mixing noise into training data raised the CNN's 10-dB score from 14.7 % to 44.8 % (single seed, CPU run); in the GPU re-run, however, noise mixing cost about 5 points of clean macro-F1, so the trade-off is not free. "
  "We report per-class results, confidence intervals, computational cost, an ablation study, and an error analysis showing that sparse, impulsive classes (fireworks, clock tick) and stationary-noise classes (engine, helicopter, washing machine) are the hardest. "
  "A Gradio demo lets a user upload or record a clip and add noise at a chosen SNR. All experiments ran on a single CPU core, and we state the resulting limitations explicitly.")

h1("2. Introduction")
P("Sound carries information that cameras cannot: a smart-home device can detect a crying baby or a breaking window, a hearing-assistive system can warn about traffic, and ecological monitoring relies on acoustic classification of animals. Environmental sound classification (ESC) is the task of assigning a short audio clip to one of a fixed set of everyday sound categories. "
  "Unlike speech or music, environmental sounds have no phonetic or harmonic grammar: they range from stationary textures (rain, engine) to sparse impulsive events (door knock, clock tick). Models must therefore learn both spectral shape and temporal structure from relatively little data.")
P("This report documents a complete deep-learning study: problem definition and hypotheses, a leakage-free data protocol, a classical baseline, three families of neural models (classic, modern, advanced), a fair comparison under identical conditions, hyper-parameter, data-strategy and pre-training studies, an ablation study, error analysis, cost measurements and a live demo. "
  "The goal, following the course brief, is not only a model that works but to <i>think like researchers</i>: state hypotheses, test them fairly, and be honest about what the evidence does and does not show.")
h2("2.1 Contributions")
bullets(["A reproducible, resumable experiment suite (30 trained models + 3 baseline runs) with one YAML configuration per experiment.",
         "Hand-written components: mel filter-bank/log-mel front-end, SpecAugment, mixup, SNR-controlled noise augmentation, multi-head self-attention, Transformer block, attention pooling and the masked-spectrogram pre-training objective.",
         "A fair, same-protocol comparison of five systems with seed variance and bootstrap confidence intervals, plus noise-robustness evaluation at four test conditions.",
         "Honest negative results: AST trained from scratch is far behind the CNN; self-supervised pre-training gave no gain; light augmentation did not improve clean accuracy within a 25-epoch budget. An independent GPU re-run is reported in Section 13.4 and changes one conclusion (hybrid vs CNN)."])

h1("3. Problem Statement")
P("<b>Task.</b> Multi-class classification of environmental sounds. <b>Input:</b> a mono audio clip of 5 s sampled at 16 kHz (80,000 samples). <b>Output:</b> one of 50 classes (animals, natural soundscapes, human non-speech sounds, domestic sounds, urban noises). "
  "<b>Primary metric:</b> macro-averaged F1 (equal weight per class; ESC-50 is perfectly balanced, 8 test clips per class). <b>Secondary metrics:</b> accuracy, top-5 accuracy, per-class F1, and macro-F1 under additive noise (white noise at 20, 10 and 0 dB SNR; pink noise at 10 dB).")
h2("3.1 Hypotheses")
P("We state the following hypotheses, each tested with the protocol of Section 11 and judged in Section 14:")
bullets(["<b>H1 (architecture).</b> An advanced CNN–Transformer hybrid outperforms a plain CNN under the same data and tuning budget.",
         "<b>H2 (inductive bias).</b> On a small dataset (1,200 training clips), a Transformer trained from scratch underperforms a CNN.",
         "<b>H3 (pre-training).</b> Self-supervised masked-spectrogram pre-training on the training clips improves the AST's accuracy compared with random initialisation.",
         "<b>H4 (data strategy).</b> Mixing noise into the training waveforms improves robustness to test-time noise without reducing clean-data accuracy.",
         "<b>H5 (deep vs classical).</b> Well-designed neural models beat a classical MFCC + SVM baseline both on clean and on noisy audio."])

h1("4. Motivation")
P("Real deployments are noisy. A classifier that reaches 60 % on studio-clean clips but collapses at 10 dB SNR is of little use in a street, a home or a forest. Yet standard ESC-50 benchmarks report only clean accuracy. "
  "A second motivation is methodological: Transformers dominate audio benchmarks when pre-trained on very large corpora (e.g. AudioSet), but student projects and small organisations rarely have such resources. We therefore ask which architectural choices actually help in the <i>low-data, low-compute</i> regime, and whether cheap self-supervision can substitute for large-scale pre-training. "
  "Finally, the course requires evidence for every claim; the study is designed so that each design decision is backed by a controlled experiment or a cited source.")

h1("5. Related Work")
P("<b>Dataset.</b> ESC-50 (Piczak, 2015) contains 2,000 labelled five-second clips extracted from Freesound recordings in 50 balanced classes arranged in five folds. Clips cut from the same source recording are kept in the same fold, which makes the official folds suitable for leakage-free evaluation. The paper reports human accuracy of about 81 %.")
P("<b>Features and classical models.</b> Mel-frequency cepstral coefficients (Davis &amp; Mermelstein, 1980) summarise the short-term spectral envelope and, aggregated into clip-level statistics and fed to SVMs or random forests, formed the standard baseline for ESC-50. Our baseline follows this recipe.")
P("<b>CNNs.</b> Convolutional networks on log-mel spectrograms became the dominant approach, from early small-kernel models to large pre-trained systems such as PANNs (Kong et al., 2020), which are trained on AudioSet. Squeeze-and-excitation blocks (Hu et al., 2018) add cheap channel attention.")
P("<b>Transformers.</b> The Transformer (Vaswani et al., 2017) and the Vision Transformer (Dosovitskiy et al., 2021) inspired the Audio Spectrogram Transformer (Gong et al., 2021), which treats spectrogram patches as tokens. AST obtains state-of-the-art ESC-50 results when initialised from ImageNet-pre-trained ViT weights; without pre-training Transformers need far more data than CNNs because they lack built-in locality and translation equivariance. "
  "Hybrid models such as the Conformer (Gulati et al., 2020) combine convolution (local patterns) with self-attention (global context).")
P("<b>Self-supervised pre-training.</b> Masked autoencoders (He et al., 2022) and their audio counterpart SSAST (Gong et al., 2022) pre-train a Transformer by reconstructing masked patches. Reported gains are obtained with large unlabelled corpora; we test whether a tiny in-domain version helps.")
P("<b>Augmentation and robustness.</b> SpecAugment (Park et al., 2019) masks frequency bands and time spans; mixup (Zhang et al., 2018) trains on convex combinations of examples. Mixing noise into training data is the standard route to noise robustness.")

h1("6. Dataset")
P("We use <b>ESC-50</b> (https://github.com/karolpiczak/ESC-50, CC BY-NC 3.0). It is the only dataset used for training and testing; noise used in the robustness tests is synthetic Gaussian noise generated on the fly, so no second dataset is needed.")
table([["Dataset", "Source", "Size", "Format", "Classes", "Licence", "Role"],
       ["ESC-50 folds 1–3", "Freesound via Piczak 2015", "1,200 clips (24 / class)", "5 s mono WAV, 44.1 kHz → 16 kHz", "50", "CC BY-NC 3.0", "Train (and SSL pre-training, labels unused)"],
       ["ESC-50 fold 4", "same", "400 clips (8 / class)", "same", "50", "same", "Validation: LR selection, early stopping"],
       ["ESC-50 fold 5", "same", "400 clips (8 / class)", "same", "50", "same", "Test: final evaluation only"],
       ["Synthetic white / pink noise", "Generated (seeded)", "added to fold-5 audio", "SNR 20/10/0 dB (white), 10 dB (pink)", "–", "n/a", "Robustness test (external perturbation)"]],
      "Dataset comparison (required Table 1).", [2.8*cm, 2.7*cm, 2.4*cm, 3*cm, 1.2*cm, 1.9*cm, 3*cm])
P("The perfect class balance (40 clips per class) means accuracy and macro-averaged recall coincide; we nevertheless use macro-F1 as the primary metric as required by the brief (task-appropriate metrics, per-class reporting). The small test fold (8 clips per class) means one clip changes a class's recall by 12.5 points, so per-class numbers are noisy; we therefore always accompany them with seed variance and a bootstrap confidence interval.")

h1("7. Data Preprocessing")
bullets(["<b>Decoding and resampling:</b> all clips are converted to mono, resampled from 44.1 kHz to 16 kHz, padded with zeros or trimmed to 5 s and peak-normalised (<font face='Courier'>data/prepare_esc50.py</font>). 33 % of samples in an average test clip are near silent because many ESC-50 clips are zero-padded; this matters for the noise experiments (Section 16).",
         "<b>Split:</b> official folds, train = 1–3, validation = 4, test = 5. Because the source recording of each clip is confined to one fold, no near-duplicate clips cross the split. The test fold is touched only once per trained model, after the best-validation epoch has been selected.",
         "<b>Front-end (written by us):</b> short-time Fourier transform (1024-point Hann window, hop 640 → 126 frames), power spectrum, 64-band triangular mel filter-bank (20 Hz – 8 kHz, own implementation of the HTK mel scale), natural-log compression, global mean/variance normalisation computed on the training set only.",
         "<b>Training-time augmentation (when enabled):</b> circular time shift, SpecAugment (two frequency masks ≤ 8 bands, two time masks ≤ 20 frames), noise mixing (50 % of batches, white or pink, SNR uniform in 5–30 dB), mixup (α = 0.4). Label smoothing (0.1) is used for all models.",
         "<b>Test-time noise:</b> Gaussian white noise or 1/f (pink) noise scaled to the requested SNR with a fixed RNG seed so that all models see identical noisy test sets."])

h1("8. Methodology")
P("Figure-free summary of the pipeline: <i>waveform → log-mel (64 × 126) → model → 50 logits</i>. All neural models share the front-end, the split, the loss (cross-entropy with label smoothing), the optimiser (AdamW, weight decay 0.01, cosine schedule with 5 % warm-up, batch size 32), the epoch budget (25), the learning-rate grid {3·10⁻⁴, 10⁻³, 3·10⁻³}, early-stopping rule (best validation macro-F1 epoch) and the evaluation code (<font face='Courier'>src/evaluate.py</font>).")
P("<b>Protocol for fairness.</b> For each neural model we train once per learning rate with seed 0 and select the learning rate with the best <i>validation</i> macro-F1; we then train two more seeds (1, 2) at that learning rate. Hence every model receives exactly three tuning runs and three final runs (the seed-0 tuning run doubles as final seed 0). The classical baseline is tuned over the SVM regularisation constant C ∈ {1, 10, 100} (also three values). The test fold plays no role in any choice.")
P("<b>Design of experiments.</b> (i) Main comparison; (ii) learning-rate study; (iii) data-strategy study on the CNN (no augmentation / shift + SpecAugment / + noise mixing / + mixup); (iv) pretrained-vs-scratch comparison for the AST; (v) ablations on the hybrid (remove Transformer + attention pooling; remove SpecAugment); (vi) noise-robustness evaluation of every final model; (vii) error analysis; (viii) cost measurement.")

h1("9. Baseline")
P("The baseline is the classical recipe: for every clip we compute the log-mel matrix, its 20 MFCCs (DCT-II of the log-mel, orthonormal) and the mean absolute MFCC delta, and summarise each channel by its mean and standard deviation over time, giving a 188-dimensional vector (64 + 64 log-mel mean/SD, 20 + 20 MFCC mean/SD, 20 delta). Features are standardised and classified with an RBF-SVM. "
  f"C = 10 was selected on validation (macro-F1 57.5 %). It reaches {main.iloc[0,1]} % test macro-F1. The SVM is deterministic, so the three 'seeds' are identical and the reported SD of 0 reflects determinism, not certainty; the bootstrap interval {main.iloc[0,4]} is the appropriate uncertainty measure. "
  "The baseline is useful as a reference point: it is fast (under one second to train) yet reaches roughly 8 points below the CNN, which shows the CNN extracts information that clip-level statistics discard.")

h1("10. Proposed / Advanced Models")
P("<b>Classic – SmallCNN (103,842 parameters).</b> Four blocks of 3×3 convolution–BatchNorm–ReLU–2×2 max-pooling with 16, 32, 64 and 128 channels; the feature map is averaged over frequency, and mean-over-time plus max-over-time pooling are summed before a dropout (0.3) + linear classifier.")
P("<b>Modern – AST-small (333,090 parameters).</b> Log-mel (64×126) is cut into non-overlapping 8×14 patches (72 tokens), linearly embedded to 96 dimensions, prefixed with a [CLS] token, added to learned position embeddings and processed by four pre-norm Transformer blocks (4 heads, MLP width 192). The classifier reads [CLS] + mean of patch tokens. Multi-head self-attention, the Transformer block and the patch pipeline are our own code (<font face='Courier'>src/models/attention.py</font>). "
  "<i>Pretrained variant:</i> before fine-tuning, the encoder is pre-trained for 60 epochs with a masked-spectrogram objective on the 1,200 training clips (50 % of patches replaced by a learned mask token; a linear head regresses the raw patch values; labels are not used; reconstruction loss fell from 0.564 to 0.357). The encoder weights initialise the fine-tuning run.")
P("<b>Advanced – ConvTransformer (240,846 parameters).</b> A three-block CNN stem (16/32/64 channels, each block Conv–BN–ReLU–<i>squeeze-and-excitation</i>–MaxPool) reduces the spectrogram to 8 × 15 frames; each time step is projected to 96 dimensions, position embeddings are added, two Transformer blocks model long-range temporal context and a learned-query <i>attention pooling</i> layer produces the clip embedding. "
  "The design follows the Conformer idea: convolution captures local spectral patterns (cheap and data-efficient), attention captures global temporal structure (which sounds repeat, how a bark or siren evolves).")
table([["Model", "Family", "Input", "Main layers", "Parameters", "Pretrained?"],
       ["MFCC + SVM", "Classical ML", "188-d clip statistics", "StandardScaler, RBF-SVM (C = 10)", "≈ 1,172 support vectors × 188", "No"],
       ["SmallCNN", "Classic (CNN)", "log-mel 64×126", "4 × [Conv3×3-BN-ReLU-MaxPool], mean+max pooling, FC", "103,842", "No"],
       ["AST-small (scratch)", "Modern (Transformer)", "log-mel 64×126", "8×14 patch embed, 4 × pre-norm MHSA block (d = 96, 4 heads)", "333,090", "No"],
       ["AST-small + SSL", "Modern (self-supervised)", "log-mel 64×126", "same; encoder from masked-patch pre-training on train clips", "333,090", "Yes (own SSL)"],
       ["ConvTransformer", "Advanced (hybrid)", "log-mel 64×126", "3 × [Conv-BN-ReLU-SE-MaxPool], projection, 2 × MHSA block, attention pooling", "240,846", "No"]],
      "Model architecture comparison (required Table 2).", [2.6*cm, 2.4*cm, 2.3*cm, 5.6*cm, 2.8*cm, 1.4*cm])

h1("11. Experimental Setup")
bullets(["<b>Hardware/software:</b> 1 vCPU core (no GPU), 3 GB RAM; Python 3.12, PyTorch 2.15, scikit-learn, librosa (resampling only). Primary results come from this CPU setup; the replication (Section 13.4) ran on a Google Colab Tesla T4 GPU.",
         "<b>Seeds:</b> 0, 1, 2 for every final configuration; the seed controls initialisation, batch order and augmentation randomness.",
         "<b>Model selection:</b> the epoch with the best validation macro-F1 is restored; the test fold is evaluated once per model.",
         "<b>Uncertainty:</b> mean ± SD (population SD) over three seeds; 95 % bootstrap CI (1,000 resamples of the 400 test clips) for the seed-0 model of each family.",
         "<b>Robustness protocol:</b> every trained model (and the baseline) is evaluated on the same noisy copies of the test fold: white 20/10/0 dB and pink 10 dB.",
         "<b>Compute budget:</b> because of the single-core limit, every run is limited to 25 epochs (≈ 3.5 min). This is a deliberate, equal budget for all models, but it is short: slower-converging models (especially Transformers) may be under-trained (Section 18)."])

h1("12. Hyperparameters")
table([["Setting", "Baseline", "CNN", "AST (scratch / SSL)", "ConvTransformer"],
       ["Optimiser", "SVM (libsvm)", "AdamW, wd 0.01", "AdamW, wd 0.01", "AdamW, wd 0.01"],
       ["Learning rate (selected on val)", "C = 10", "3·10⁻³", "10⁻³ (both)", "3·10⁻³"],
       ["LR grid searched", "C ∈ {1, 10, 100}", "{3e-4, 1e-3, 3e-3}", "same", "same"],
       ["Batch size / epochs", "– / –", "32 / 25", "32 / 25 (+60 SSL epochs)", "32 / 25"],
       ["Schedule", "–", "5 % warm-up + cosine", "same", "same"],
       ["Regularisation", "RBF γ = 'scale'", "dropout 0.3, label smoothing 0.1", "dropout 0.1, label smoothing 0.1", "dropout 0.1 / 0.3, label smoothing 0.1"],
       ["Loss", "hinge (SVM)", "cross-entropy (soft labels)", "same", "same"],
       ["Augmentation", "none", "time shift + SpecAugment", "same", "same"]],
      "Hyperparameter configuration (required Table 3). The default augmentation is identical for all neural models.", [3.8*cm, 2.6*cm, 3.5*cm, 3.7*cm, 3.5*cm])
P("The learning-rate study (Figure 1, Table A1 in the appendix) shows that all four neural families prefer the larger rates within the 25-epoch budget (the CNN and the hybrid peak at 3·10⁻³, the AST at 10⁻³); the optimum for the CNN and hybrid sits at the edge of the grid, so a larger rate might be better still. This is a limitation of the three-value grid imposed by the compute budget.")
fig("hparam_lr.png", "Validation macro-F1 versus learning rate (seed 0) for every neural model. Selection used validation data only.", 10.5)

h1("13. Results")
P(f"Table 4 gives the main results on the untouched test fold. The ConvTransformer is best ({main.iloc[4,1]} % macro-F1), followed by the CNN ({main.iloc[1,1]} %) and the SVM baseline ({main.iloc[0,1]} %); both AST variants are far behind ({main.iloc[2,1]} % and {main.iloc[3,1]} %). Top-5 accuracy is much higher than top-1 for the neural models (hybrid 90.8 %), "
  "which means most errors are 'near misses' among acoustically similar classes rather than complete failures.")
table(main.drop(columns=["n seeds"]), "Main experimental results (required Table 4), test fold, mean ± SD over 3 seeds (baseline is deterministic). 'LR' = selected learning rate or SVM C.", [3.7*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.8*cm, 2.2*cm, 1.6*cm])
fig("val_curves.png", "Validation macro-F1 per epoch (mean over seeds). The AST models are still rising at epoch 25, i.e. they are not converged under the shared budget.", 10.5)
h2("13.1 Noise robustness")
table(rob, "Test macro-F1 (%) under additive noise (mean ± SD over seeds). SNR is computed over the whole clip.", [3.8*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm])
P("Every model is very sensitive to noise, including the best: the hybrid keeps 41.4 % at 20 dB, 21.6 % at 10 dB and 5.7 % at 0 dB. The ranking nevertheless stays the same (hybrid > CNN > baseline > AST). Two factors make our protocol harsh: the models are trained on clean audio, and because ESC-50 contains many zero-padded or near-silent regions, the added noise fills what was silence, which changes the spectrogram much more than the SNR number alone suggests. "
  "This is why the data-strategy study below, which trains with noise, matters.")
fig("robustness.png", "Test macro-F1 versus white-noise SNR; error bars are the SD over seeds.", 10.5)
h2("13.2 Data-strategy study (CNN, seed 0)")
table(data, "Data-strategy study with the CNN at its selected learning rate (single seed: differences of 2–3 points are within seed noise, see SD in Table 4).", [5.5*cm, 3*cm, 3.5*cm, 4.5*cm])
P("Three observations. (1) Noise mixing leaves clean performance unchanged (56.7 % vs 57.2 %) but triples robustness at 10 dB (44.8 % vs 14.7 %): this is the single largest gain in the study and supports the robustness part of H4. In the GPU re-run (Section 13.4), however, noise mixing lowered clean macro-F1 from 63.1 % to 57.9 %, so the ‘no clean cost’ part of H4 did not replicate (the robustness gain was not re-measured there). (2) Mixup gave no gain. "
  "(3) Surprisingly, <i>no augmentation</i> gave the best clean accuracy (64.7 % test, 72.1 % validation) in this run. We believe this is a budget effect: SpecAugment and shifting make the training task harder, and 25 epochs are too few for the regularisation to pay off. We did not verify this explanation (a longer-schedule run would be required); it is a hypothesis, and with one seed the gap could also partly be noise. "
  "It does imply that our default augmentation recipe, applied equally to all models, may have <i>under</i>-estimated all of them on clean data. (The GPU re-run again gave the no-augmentation CNN the best clean score, 63.7 % vs 63.1 % for the default recipe.)")
h2("13.3 Pretrained versus scratch (AST)")
P("Self-supervised pre-training did not improve the AST: 29.7 ± 2.6 % macro-F1 versus 31.0 ± 2.6 % from scratch (top-5: 71.5 % vs 69.4 %). The difference is smaller than the seed SD, so we conclude <b>no measurable effect</b>, not that pre-training is harmful. Plausible reasons: only 1,200 clips (the pre-training set is as small as the fine-tuning set), 60 pre-training epochs, patch regression of normalised log-mel values that may emphasise low-level texture, and a total budget that is orders of magnitude below published SSAST pre-training. "
  "Neither AST variant has converged (Figure 2).")

h2("13.4 Independent replication on a Colab T4 GPU")
P("To check how stable our conclusions are, the whole suite (tuning grid, 3-seed finals, data-strategy study, SSL pre-training, ablations and baseline) was re-run from the same code and configs on a Google Colab Tesla T4 GPU. Different hardware and random-number streams mean the numbers are not identical to the CPU run. Table R4 compares the main results; Table R5 compares the ablation and data-strategy runs.")
table([["Model", "CPU run: macro-F1 (%)", "GPU re-run: macro-F1 (%)", "GPU re-run: accuracy (%)", "GPU re-run: top-5 (%)", "GPU re-run: bootstrap 95% CI (seed 0)"],
       ["Baseline (MFCC+SVM)", "50.9 ± 0.0", "50.9 ± 0.0", "53.0 ± 0.0", "56.8 ± 0.0", "[44.8, 53.8]"],
       ["CNN (classic)", "57.9 ± 0.5", "61.5 ± 1.2", "64.2 ± 1.1", "88.6 ± 0.2", "[57.6, 65.9]"],
       ["AST scratch (modern)", "31.0 ± 2.6", "31.1 ± 2.2", "33.8 ± 1.6", "69.4 ± 0.8", "[28.5, 37.5]"],
       ["AST + SSL (modern)", "29.7 ± 2.6", "30.4 ± 2.1", "33.8 ± 2.0", "71.2 ± 0.9", "[28.4, 36.2]"],
       ["ConvTransformer (advanced)", "62.9 ± 1.5", "62.0 ± 0.1", "64.0 ± 0.4", "89.8 ± 0.4", "[55.9, 65.0]"]],
      "Replication on a Colab T4 GPU: main results (test fold, mean ± SD over 3 seeds).", [3.6*cm, 2.4*cm, 2.5*cm, 2.5*cm, 2.3*cm, 3.7*cm])
d = dict(zip(data["Strategy"], data["Test macro-F1 (%)"]))
table([["Run (test macro-F1, %)", "CPU run", "GPU re-run"],
       ["Hybrid, full model (3 seeds)", "62.9 ± 1.5", "62.0 ± 0.1"],
       ["Hybrid − Transformer & attention pooling (3 seeds)", "58.8 ± 1.0", "58.5 ± 1.0"],
       ["Hybrid − SpecAugment (seed 0)", "60.4", "62.5 (full, seed 0: 61.9)"],
       ["CNN, no augmentation (seed 0)", str(d["No augmentation"]), "63.7"],
       ["CNN, default shift + SpecAugment (seed 0)", str(d["Shift + SpecAugment (default)"]), "63.1"],
       ["CNN + noise mixing (seed 0)", str(d["+ noise mixing"]), "57.9"],
       ["CNN + mixup (seed 0)", str(d["+ mixup"]), "55.1"]],
      "Replication of the ablation and data-strategy runs.", [8.5*cm, 3.6*cm, 4.2*cm])
P("<b>What replicates.</b> (i) The baseline is identical (deterministic). (ii) Both AST variants stay about 30 points behind the CNN and the hybrid (31.1 % and 30.4 %), and SSL pre-training again has no measurable effect. (iii) Removing the Transformer and attention pooling lowers the hybrid by 3.5 points (3 seeds each). (iv) No augmentation scores at least as well as the default recipe on clean data. "
  "<b>What does not replicate.</b> (i) The hybrid's lead over the CNN falls from 5.0 to 0.5 points: the CNN's mean rose from 57.9 % to 61.5 % while the hybrid stayed near 62 %, and the hybrid's seed-to-seed variance was very small (SD 0.1). (ii) The SpecAugment ablation changes sign (−2.4 vs +0.6 points). "
  "(iii) Noise mixing cost 5.2 clean points instead of 0.5. We report the CPU run as the primary result because the repository ships its raw results, and we treat the replication as a warning about run-to-run variability: differences of a few points between single training runs, and even 3-seed means, should not be over-interpreted.")
h1("14. Model Comparison")
P("All models were compared under one protocol: same split, same preprocessing, same three-value tuning budget, same 25-epoch budget and three seeds. Our verdicts on the hypotheses:")
table([["Hypothesis", "Evidence", "Verdict"],
       ["H1: hybrid > CNN", "CPU run: 62.9 ± 1.5 vs 57.9 ± 0.5 (+5.0). GPU re-run: 62.0 ± 0.1 vs 61.5 ± 1.2 (+0.5, within seed noise). The attention stage itself helps: removing it costs 4.1 (CPU) / 3.5 (GPU) points.", "<b>Not established</b> (hybrid ≥ CNN in both runs, but the margin varies from 0.5 to 5 points)"],
       ["H2: scratch Transformer < CNN on small data", "31.0 ± 2.6 vs 57.9 ± 0.5; gap of 27 points far exceeds any variance.", "<b>Strongly supported</b>"],
       ["H3: SSL pre-training helps AST", "29.7 ± 2.6 (SSL) vs 31.0 ± 2.6 (scratch).", "<b>Not supported</b> (no measurable effect)"],
       ["H4: noise mixing improves robustness at no clean cost", "CPU run (1 seed): 10 dB white 44.8 % vs 14.7 %, clean 56.7 % vs 57.2 %. GPU re-run: clean 57.9 % vs 63.1 % (−5.2 points); robustness not re-measured.", "<b>Partly supported</b> (robustness gain seen once; 'no clean cost' did not replicate)"],
       ["H5: deep models beat MFCC + SVM clean and noisy", "CNN +7.0 and hybrid +12.0 points clean; both also better under all noise levels. AST (31 %) is <i>worse</i> than the baseline (50.9 %).", "<b>Partly supported</b> (true for CNN and hybrid, false for AST)"]],
      "Verdict on each hypothesis.", [4*cm, 9*cm, 4*cm])
P("<b>Does the hybrid really beat the CNN?</b> Not reliably. In the CPU run it was 5.0 points better, in the GPU re-run only 0.5 points (62.0 vs 61.5 %), and the CNN's own seed variance is about ±1 point, so we cannot claim a robust advantage. What does replicate is the ablation: the same CNN stem without the Transformer and attention pooling is 3.5–4.1 points worse than the full hybrid, so the attention stage adds value relative to its own stem; the plain CNN simply turned out to be a strong baseline whose score varied between runs. <b>Why does the AST lose?</b> It has no convolutional inductive bias and only 1,200 training clips; its validation curve is still rising at epoch 25 (Figure 2). "
  "Published AST results rely on ImageNet-pre-trained weights, which we deliberately do not use (the brief forbids external APIs and requires own training).")

h1("15. Ablation Study")
table(abl, "Ablation study (required Table 5); test macro-F1 (%). Δ is the change relative to the full model. The SpecAugment row uses one seed.", [7*cm, 3*cm, 5*cm, 1.6*cm])
P("<b>Removing the Transformer blocks and attention pooling</b> (replaced by mean pooling over time) lowers macro-F1 by 4.1 points with three seeds (58.8 ± 1.0 vs 62.9 ± 1.5), a difference of roughly two to three SDs. <b>Removing SpecAugment</b> lowers the hybrid by 2.4 points, but this is a single seed (seed 0) and is smaller than the seed SD of the full model (1.5), so it is <i>inconclusive</i>; "
  "moreover the CNN data-strategy study pointed in the opposite direction. We therefore do not claim that SpecAugment helps. <b>Removing SSL pre-training</b> from the AST changes nothing measurable (Section 13.3). In the GPU re-run the no-attention ablation gave 58.5 ± 1.0 % (−3.5 points vs 62.0 %) and removing SpecAugment gave +0.6 points (62.5 % vs 61.9 %, seed 0), so the attention result replicates and the SpecAugment effect is best described as zero within noise.")

h1("16. Error Analysis")
P(f"On the best-validation hybrid model (seed 0) {st['n_errors']} of {st['n_test']} test clips are misclassified. The model is moderately calibrated: mean confidence is {st['mean_conf_correct']:.2f} on correct and {st['mean_conf_wrong']:.2f} on wrong predictions, and only 15 of the 139 errors have confidence above 0.7, so a confidence threshold would filter many errors. "
  "Easiest classes (F1): sheep 0.91, snoring 0.88, pouring water 0.88, crow 0.87, crying baby 0.86. Hardest: fireworks 0.07, helicopter 0.08, washing machine 0.32, clapping 0.33, engine 0.33 (mean over seeds).")
fig("per_class_f1.png", "Per-class test F1 of the hybrid model (mean over 3 seeds). Each class has only 8 test clips.", 14)
fig("confusion_best.png", "Confusion matrix of the hybrid model (seed 0, test fold).", 11.5)
table(rd("table6_error_analysis_final.csv") if os.path.exists(T + "table6_error_analysis_final.csv") else err, "Error analysis (required Table 6): the eight most frequent confusions of the hybrid model. 'Likely cause' entries are hypotheses; those marked † are supported by a measurement we made, the rest are plausible explanations that we did not verify by listening.", [3.4*cm, 1.2*cm, 3*cm, 9.4*cm])
P("<b>Pattern 1 – sparse, impulsive events.</b> Fireworks, clock tick and frog clips are mostly silence: the fraction of near-silent samples is 49 %, 66 % and 44 % for these classes versus 33 % on average, so few frames carry evidence and the models fall back on classes with similar short bursts (fireworks → keyboard typing, clock tick → dog, frog → dog). Fireworks has zero recall. "
  "<b>Pattern 2 – stationary noise textures.</b> Engine, washing machine, vacuum cleaner, wind and helicopter share broadband, quasi-stationary low-frequency spectra; the only distinguishing cue is a weak periodic modulation that is easily masked. "
  "<b>Pattern 3 – small test fold.</b> With 8 clips per class, a single confusion changes recall by 12.5 points; the ranking of the hardest classes should not be over-interpreted.")
fig("error_examples.png", "Log-mel spectrograms of the three most frequent confusions: top row = clips misclassified (true class shown), bottom row = a typical clip of the predicted class.", 14)
P("<b>Failure under noise.</b> At 10 dB white noise the hybrid macro-F1 drops from 62.9 % to 21.6 %. As discussed, noise fills silent regions, so sparse classes are affected most. Training with noise (Section 13.2) is the first remedy; others (spectral subtraction, noise-aware normalisation) were not tested.")

h1("17. Discussion")
P("The central lesson is that <b>inductive bias beats raw capacity on small data</b>. A 104 k-parameter CNN beats a 333 k-parameter Transformer by 27–30 points, while a hybrid that keeps the CNN's locality and adds a small attention stage performs at least as well as the CNN (5.0 points better in one run, 0.5 in another). "
  "The second lesson is that <b>robustness must be trained, not assumed</b>: no architectural change closed the noise gap, whereas noise mixing did so cheaply. The third is methodological: <b>negative results are results</b> – pre-training did not help, and augmentation did not help clean accuracy in a short schedule; reporting them prevents the false impression that every added component improves performance. "
  "Finally, cost matters (Table 7): the hybrid is 2.3× the CNN's parameters but trains in a similar time and infers within 3.2 ms per clip on one CPU core, so its extra cost over the CNN is small.")
table(cost, "Computational cost (required Table 7). Training time is the mean over seeds on 1 vCPU; size assumes float32 weights; the baseline's 'size' is its stored support-vector values.", [3.6*cm, 1.8*cm, 1.8*cm, 2.2*cm, 2.8*cm, 4.8*cm])

h1("18. Limitations")
bullets(["<b>Run-to-run variability:</b> re-running the suite on a GPU changed the CNN's 3-seed mean from 57.9 % to 61.5 % and the hybrid's from 62.9 % to 62.0 % (Section 13.4); CNN-vs-hybrid differences below about 5 points are therefore not reliable.", "<b>Compute budget:</b> a single CPU core limited us to small models, 25 epochs per run, a three-value learning-rate grid whose best value for the CNN and hybrid sits on the edge, and no tuning of architecture, patch size or augmentation strength. The AST had not converged; a longer schedule or a GPU could change the AST and SSL conclusions.",
         "<b>Small test fold:</b> 400 clips (8 per class); confidence intervals are wide (about ±5 points) and CNN vs hybrid intervals overlap. Only three seeds were used.",
         "<b>Single-seed studies:</b> the data-strategy study and the SpecAugment ablation use one seed; they indicate directions, not established effects.",
         "<b>Fixed split:</b> we evaluated on one fold arrangement, not full five-fold cross-validation, so numbers are not directly comparable to published 5-fold ESC-50 results (which also commonly use pre-trained models and 44.1 kHz inputs).",
         "<b>Noise model:</b> synthetic stationary Gaussian noise relative to whole-clip power; real-world noise (street, speech babble) is not stationary. Zero-padded regions inflate the effect of added noise.",
         "<b>No external pre-training:</b> by design (no external AI APIs or unattributed models), so our AST numbers are not representative of ImageNet-initialised AST.",
         "<b>Error-cause explanations</b> are partly hypotheses; we did not listen to every failing clip."])

h1("19. Ethical Considerations")
bullets(["<b>Privacy:</b> deployed always-on sound classifiers can capture private conversations. A system based on this work should process audio on-device, discard raw audio and classify non-speech categories only; the demo does not store uploads.",
         "<b>Licensing:</b> ESC-50 is released under CC BY-NC 3.0 and individual clips carry Creative-Commons licences; the models here are for non-commercial educational use.",
         "<b>Reliability and safety:</b> our best model errs on 35 % of clips and degrades strongly in noise; it must not be used for safety-critical alerts (e.g. alarms for hearing-impaired users) without far higher accuracy and field testing.",
         "<b>Dataset bias:</b> ESC-50 reflects the recording equipment and geography of Freesound contributors; sounds typical of other regions (e.g. specific vehicles or animals) are under-represented.",
         "<b>Environmental cost:</b> all experiments are small (a few CPU-hours) – an argument for small, carefully designed models."])

h1("20. Conclusion")
P("We built and compared five systems for noisy environmental sound classification under one protocol. A CNN–Transformer hybrid (62.9 ± 1.5 % macro-F1) and a plain CNN (57.9 ± 0.5 %) both beat an MFCC + SVM baseline (50.9 %), while an independent GPU re-run shrank the hybrid's lead over the CNN to 0.5 points (62.0 vs 61.5 %); a Transformer trained from scratch (31.0 %) and a tiny self-supervised pre-training stage (29.7 %) did not. "
  "Noise robustness is the main weakness of all models trained on clean audio, and noise mixing is a cheap, effective cure. The study also illustrates the value of honest reporting: several expected effects (pre-training, augmentation) did not appear within our budget.")

h1("21. Future Work")
bullets(["Run full five-fold cross-validation and more seeds; replicate the data-strategy and SpecAugment studies with ≥ 3 seeds.",
         "Train longer (≥ 100 epochs) on a GPU and re-test H3 with larger-scale unlabelled pre-training data (e.g. other public audio corpora) and with patch overlap in the AST.",
         "Combine noise mixing with real background noise (street, babble) and evaluate with non-stationary test noise; study SNR defined on active frames only.",
         "Add knowledge distillation from a larger model, test-time augmentation and model ensembling.",
         "Package the hybrid as an on-device model (quantisation) and measure latency on a phone-class CPU."])

h1("22. References")
refs = ["K. J. Piczak, “ESC: Dataset for Environmental Sound Classification,” in Proc. ACM Multimedia, 2015, pp. 1015–1018.",
        "S. B. Davis and P. Mermelstein, “Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences,” IEEE Trans. ASSP, 28(4), 1980.",
        "A. Vaswani et al., “Attention is all you need,” in Advances in Neural Information Processing Systems (NeurIPS), 2017.",
        "A. Dosovitskiy et al., “An image is worth 16×16 words: Transformers for image recognition at scale,” in ICLR, 2021.",
        "Y. Gong, Y.-A. Chung, J. Glass, “AST: Audio Spectrogram Transformer,” in Proc. Interspeech, 2021.",
        "Y. Gong, C.-I. Lai, Y.-A. Chung, J. Glass, “SSAST: Self-Supervised Audio Spectrogram Transformer,” in Proc. AAAI, 2022.",
        "K. He et al., “Masked autoencoders are scalable vision learners,” in Proc. CVPR, 2022.",
        "D. S. Park et al., “SpecAugment: A simple data augmentation method for automatic speech recognition,” in Proc. Interspeech, 2019.",
        "H. Zhang, M. Cisse, Y. N. Dauphin, D. Lopez-Paz, “mixup: Beyond empirical risk minimization,” in ICLR, 2018.",
        "J. Hu, L. Shen, G. Sun, “Squeeze-and-excitation networks,” in Proc. CVPR, 2018.",
        "A. Gulati et al., “Conformer: Convolution-augmented Transformer for speech recognition,” in Proc. Interspeech, 2020.",
        "Q. Kong et al., “PANNs: Large-scale pretrained audio neural networks for audio pattern recognition,” IEEE/ACM Trans. Audio, Speech, Lang. Process., 28, 2020.",
        "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in ICLR, 2019.",
        "C. Szegedy et al., “Rethinking the Inception architecture for computer vision,” in Proc. CVPR, 2016 (label smoothing)."]
for i, r in enumerate(refs, 1): S.append(Paragraph(f"[{i}] {r}", ParagraphStyle("ref", parent=B, fontSize=9, leading=11.5, leftIndent=18, firstLineIndent=-18, alignment=0, spaceAfter=3)))

h1("Appendix A. Learning-rate study (validation macro-F1, seed 0)")
hp2 = hp.copy(); hp2.columns = ["Model", "Learning rate", "Val macro-F1 (%)"]; table(hp2, "All tuning runs used to select the learning rates.", [7*cm, 4*cm, 4*cm])
h1("Appendix B. Reproducibility")
P("Environment: <font face='Courier'>pip install -r requirements.txt</font>. Data: <font face='Courier'>python data/prepare_esc50.py</font>. Everything: <font face='Courier'>python scripts/run_all.py all</font>, <font face='Courier'>python -m src.models.baseline_ml</font>, <font face='Courier'>python scripts/aggregate.py</font>. Demo: <font face='Courier'>python scripts/export_demo_checkpoint.py &amp;&amp; python demo/app.py</font>. "
  "One YAML file per experiment is stored in <font face='Courier'>configs/</font>; raw results (JSON, per-run histories, test logits) in <font face='Courier'>experiments/results/</font>.")

def foot(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey); c.drawCentredString(A4[0] / 2, 1.1 * cm, f"{TEAM} – AI403 Deep Learning Final Project – page {d.page}"); c.restoreState()
doc = SimpleDocTemplate(ROOT + f"/report/{TEAM}_Report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=1.8 * cm, title="Noise-Robust Environmental Sound Classification on ESC-50", author=TEAM)
doc.build(S, onFirstPage=lambda c, d: None, onLaterPages=foot)
print("built", doc.page if hasattr(doc, "page") else "")
