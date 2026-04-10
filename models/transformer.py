import torch
import torch.nn as nn


class NoPatchTransformer(nn.Module):
    def __init__(self, n_in, L, d_model, n_heads, n_layers, dim_ff, dropout, n_thr):
        super().__init__()
        self.input_proj = nn.Linear(n_in, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, L, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.reg_head = nn.Linear(d_model, 1)
        self.thr_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_thr)])

    def forward(self, x):
        tokens = self.input_proj(x) + self.pos_embed
        tokens = self.transformer(tokens)
        pooled = self.norm(tokens.mean(dim=1))
        return self.reg_head(pooled).squeeze(-1), [h(pooled).squeeze(-1) for h in self.thr_heads]
