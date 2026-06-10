import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        input_size = n_dynamic + n_static
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.n_static = n_static

    def forward(
        self,
        x_dyn: torch.Tensor,      # (B, seq_len, n_dynamic)
        x_static: torch.Tensor,   # (B, n_static)
    ) -> torch.Tensor:             # (B,)
        if self.n_static > 0:
            static_exp = x_static.unsqueeze(1).expand(-1, x_dyn.size(1), -1)
            x = torch.cat([x_dyn, static_exp], dim=-1)
        else:
            x = x_dyn
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)
