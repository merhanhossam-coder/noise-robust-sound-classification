"""Live demo (Gradio): upload / record a sound, add noise at a chosen SNR, get the top-5 predicted ESC-50 classes.
Run:  python demo/app.py   (needs checkpoints/hybrid_demo.pt, created by scripts/export_demo_checkpoint.py)
The result updates automatically when you move the slider, and the *noisy* audio that the model really received is played back."""
import os, sys, numpy as np, torch, librosa, gradio as gr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.esc50 import LogMel, add_noise, SR
from src.models.networks import build
torch.set_num_threads(1)
CK = torch.load(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "hybrid_demo.pt"), map_location="cpu")
cfg, classes = CK["cfg"], CK["classes"]
model = build(cfg["model"], n_cls=len(classes), **cfg.get("model_kwargs", {})); model.load_state_dict(CK["state"]); model.eval()
front = LogMel(); front.mu.copy_(CK["mu"]); front.sd.copy_(CK["sd"])

def prepare(audio):
    sr, x = audio; x = x.astype(np.float32)
    if x.ndim > 1: x = x.mean(1)
    if np.abs(x).max() > 1: x /= 32768.0
    x = librosa.resample(x, orig_sr=sr, target_sr=SR)[: SR * 5]
    x = np.pad(x, (0, SR * 5 - len(x)))
    return x / (np.abs(x).max() + 1e-8)

def predict(audio, snr):
    if audio is None: return {}, None
    xt = torch.from_numpy(prepare(audio)).unsqueeze(0)
    if snr < 40: xt = add_noise(xt, float(snr))                 # 40 = no noise
    with torch.no_grad(): p = torch.softmax(model(front(xt)), 1)[0]
    heard = xt[0].numpy(); heard = heard / (np.abs(heard).max() + 1e-8) * 0.9
    return {classes[i]: float(p[i]) for i in p.argsort(descending=True)[:5]}, (SR, heard.astype(np.float32))

if __name__ == "__main__":
    gr.Interface(predict, [gr.Audio(sources=["upload", "microphone"], label="Sound (first 5 s are used)"),
                           gr.Slider(-5, 40, value=40, step=1, label="White-noise SNR in dB (40 = no noise; lower = noisier)")],
                 [gr.Label(num_top_classes=5, label="Top-5 predictions"), gr.Audio(label="Audio the model actually heard (play it to check the noise)")],
                 live=True, title="Noise-robust environmental sound classifier (ESC-50)",
                 description="ConvTransformer trained from scratch on ESC-50 (50 classes). Move the slider: the prediction updates by itself.").launch(share=bool(os.environ.get("COLAB_RELEASE_TAG")))
