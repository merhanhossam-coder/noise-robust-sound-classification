"""Self-supervised masked-spectrogram pre-training of the AST (labels NOT used; training-fold clips only)."""
import os, sys, time, numpy as np, torch
from src.datasets.esc50 import load_split, LogMel, random_shift, add_noise
from src.models.networks import AST
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cpu": torch.set_num_threads(1)
ROOT = os.path.join(os.path.dirname(__file__), "..")

def main(epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    data, classes = load_split(); X = data["train"][0]
    front = LogMel(); front.fit_norm(X); front = front.to(DEV); m = AST(n_cls=len(classes)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * int(np.ceil(len(X) / 32))); t0 = time.time()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), 32):
            x = random_shift(X[perm[i:i + 32]]); loss = m.mae_loss(front(x.to(DEV)))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(x)
        if ep % 10 == 0 or ep == epochs - 1: print(f"ssl ep{ep} recon-loss {tot/len(X):.4f}", flush=True)
    sd = {k: v.cpu() for k, v in m.state_dict().items() if not k.startswith("head")}
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    torch.save(sd, os.path.join(ROOT, "checkpoints", "ast_ssl_pretrained.pt")); print("pretrain time", time.time() - t0)

if __name__ == "__main__": main()
