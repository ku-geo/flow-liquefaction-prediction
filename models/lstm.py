import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, n_in, hidden_size, num_layers, dropout, n_thr):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_in, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.thr_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_thr)])

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        pooled = self.norm(h_n[-1])
        return (self.reg_head(pooled).squeeze(-1),
                [h(pooled).squeeze(-1) for h in self.thr_heads])
