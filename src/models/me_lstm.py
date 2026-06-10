"""Mean-Embedding LSTM (Kratzert et al., 2019 / Entity-Aware LSTM)."""
import torch
import torch.nn as nn


class MELSTMModel(nn.Module):
    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        static_embed_dim: int = 32,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.n_static = n_static
        self.embed_dim = static_embed_dim if n_static > 0 else 0

        if n_static > 0:
            self.static_embed = nn.Sequential(
                nn.Linear(n_static, static_embed_dim),
                nn.ReLU(),
            )
        else:
            self.static_embed = None

        lstm_input = n_dynamic + self.embed_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x_dyn: torch.Tensor,      # (B, seq_len, n_dynamic)
        x_static: torch.Tensor,   # (B, n_static)
    ) -> torch.Tensor:             # (B,)
        if self.static_embed is not None and self.n_static > 0:
            emb = self.static_embed(x_static)                       # (B, embed_dim)
            emb_exp = emb.unsqueeze(1).expand(-1, x_dyn.size(1), -1)
            x = torch.cat([x_dyn, emb_exp], dim=-1)
        else:
            x = x_dyn
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)
