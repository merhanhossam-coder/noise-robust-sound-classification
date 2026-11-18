import torch, torch.nn as nn
from .attention import Block, AttnPool

def conv_bn(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class SmallCNN(nn.Module):
    """Classic model: 4-block CNN on log-mel."""
    def __init__(self, n_cls=50, ch=(16, 32, 64, 128), drop=0.3):
        super().__init__()
        L, i = [], 1
        for c in ch: L += [conv_bn(i, c), nn.MaxPool2d(2)]; i = c
        self.features = nn.Sequential(*L)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(ch[-1], n_cls))
    def forward(self, x):
        f = self.features(x).mean(2)                       # mean over frequency -> [B, C, T]
        return self.head(f.mean(2) + f.amax(2))

class AST(nn.Module):
    """Modern model: small Audio-Spectrogram-Transformer (patch embedding + own MHSA blocks)."""
    def __init__(self, n_cls=50, dim=96, depth=4, heads=4, patch=(8, 14), size=(64, 126), drop=0.1):
        super().__init__()
        self.patch, self.dim = patch, dim
        self.nf, self.nt = size[0] // patch[0], size[1] // patch[1]
        self.embed = nn.Linear(patch[0] * patch[1], dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)); self.pos = nn.Parameter(torch.randn(1, 1 + self.nf * self.nt, dim) * 0.02)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, 2 * dim, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, n_cls)
        self.recon = nn.Linear(dim, patch[0] * patch[1])   # only used for masked pre-training

    def patchify(self, x):                                 # [B,1,64,126] -> [B, N, pf*pt]
        pf, pt = self.patch; x = x[:, :, : self.nf * pf, : self.nt * pt]
        B = x.shape[0]
        return x.reshape(B, self.nf, pf, self.nt, pt).permute(0, 1, 3, 2, 4).reshape(B, self.nf * self.nt, pf * pt)

    def encode(self, x, mask=None):
        p = self.patchify(x); t = self.embed(p)
        if mask is not None: t = torch.where(mask.unsqueeze(-1), self.mask_tok.expand_as(t), t)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos
        for b in self.blocks: t = b(t)
        return self.norm(t), p

    def forward(self, x):
        t, _ = self.encode(x); return self.head(t[:, 0] + t[:, 1:].mean(1))

    def mae_loss(self, x, ratio=0.5):
        B = x.shape[0]; N = self.nf * self.nt
        mask = torch.rand(B, N, device=x.device).argsort(1).argsort(1) < int(ratio * N)
        t, p = self.encode(x, mask); pred = self.recon(t[:, 1:])
        return ((pred - p) ** 2).mean(-1)[mask].mean()

class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__(); self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x.mean((2, 3)))[:, :, None, None]

class ConvTransformer(nn.Module):
    """Advanced model: CNN stem with squeeze-excitation -> Transformer encoder over time -> attention pooling."""
    def __init__(self, n_cls=50, ch=(16, 32, 64), dim=96, depth=2, heads=4, drop=0.1, use_attention=True):
        super().__init__()
        L, i = [], 1
        for c in ch: L += [conv_bn(i, c), SE(c), nn.MaxPool2d(2)]; i = c
        self.stem = nn.Sequential(*L)
        f = 64 // 2 ** len(ch)
        self.proj = nn.Linear(ch[-1] * f, dim); self.use_attention = use_attention
        self.pos = nn.Parameter(torch.randn(1, 126 // 2 ** len(ch) + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, 2 * dim, drop) for _ in range(depth)]) if use_attention else nn.ModuleList()
        self.pool = AttnPool(dim) if use_attention else None
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, n_cls))
    def forward(self, x):
        z = self.stem(x); B, C, F, T = z.shape
        t = self.proj(z.permute(0, 3, 1, 2).reshape(B, T, C * F)); t = t + self.pos[:, :T]
        for b in self.blocks: t = b(t)
        return self.head(self.pool(t) if self.use_attention else t.mean(1))

def build(name, **kw):
    return {"cnn": SmallCNN, "ast": AST, "hybrid": ConvTransformer}[name](**kw)
