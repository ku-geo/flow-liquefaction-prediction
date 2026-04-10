import numpy as np
import torch.nn as nn


def extract_fft(X, n_fft):
    N, L, C = X.shape
    feat = np.zeros((N, C * n_fft), dtype=np.float32)
    for ch in range(C):
        spec = np.fft.rfft(X[:, :, ch], axis=1)
        mag = np.abs(spec[:, 1:n_fft + 1]) / L
        feat[:, ch * n_fft:(ch + 1) * n_fft] = mag
    return feat


class FFTMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout, n_thr):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.shared = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(prev)
        self.reg_head = nn.Linear(prev, 1)
        self.thr_heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(n_thr)])

    def forward(self, x):
        feat = self.norm(self.shared(x))
        return (self.reg_head(feat).squeeze(-1),
                [h(feat).squeeze(-1) for h in self.thr_heads])
