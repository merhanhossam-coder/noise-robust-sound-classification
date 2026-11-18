"""Transformer building blocks written from scratch (key component #2)."""
import math, torch, torch.nn as nn, torch.nn.functional as F

class MHSA(nn.Module):
    def __init__(self, dim, heads, drop=0.1):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.qkv, self.proj, self.drop = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim), nn.Dropout(drop)

    def forward(self, x):                                   # x [B, N, D]
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        a = self.drop(torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dk), dim=-1))
        return self.proj((a @ v).transpose(1, 2).reshape(B, N, D))

class Block(nn.Module):
    def __init__(self, dim, heads, mlp, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.att = MHSA(dim, heads, drop)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp, dim), nn.Dropout(drop))
    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.mlp(self.n2(x))

class AttnPool(nn.Module):
    """Learned-query attention pooling over tokens."""
    def __init__(self, dim):
        super().__init__(); self.q = nn.Parameter(torch.randn(dim) * 0.02); self.k = nn.Linear(dim, dim)
    def forward(self, x):
        w = torch.softmax(self.k(x) @ self.q / math.sqrt(x.shape[-1]), dim=1)
        return (w.unsqueeze(-1) * x).sum(1)
