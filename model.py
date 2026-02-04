# model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0  # GRU dropout only applies when num_layers>1
    layer_norm: bool = True


class PredictiveGRU(nn.Module):
    """
    One GRU backbone, three heads:
      - pred_head: predict x_{t+1} from h_t (next-step prediction)
      - cls_head:  classify dev_pos at trial end states h_{t_end} -> {4,5,6}
      - rt_head:   classify deviant vs standard at each time step (for RT readout)

    Notes:
      - cls_head outputs 3 logits corresponding to classes [4,5,6]
      - rt_head outputs 2 logits corresponding to classes [standard, deviant]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.ln = nn.LayerNorm(cfg.hidden_dim) if cfg.layer_norm else nn.Identity()

        # Heads
        self.pred_head = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        self.cls_head = nn.Linear(cfg.hidden_dim, 3)
        self.rt_head = nn.Linear(cfg.hidden_dim, 2)  # [standard, deviant]

    def _ensure_h0(
        self,
        x_chunk: torch.Tensor,
        h0: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Ensure h0 is on the same device/dtype as x_chunk.
        Critical for XLA/TPU where GRU may be implemented via scan.
        """
        B = int(x_chunk.shape[0])
        device = x_chunk.device
        dtype = x_chunk.dtype

        if h0 is None:
            return torch.zeros(
                (self.cfg.num_layers, B, self.cfg.hidden_dim),
                device=device,
                dtype=dtype,
            )

        return h0.to(device=device, dtype=dtype)

    def forward_chunk(
        self,
        x_chunk: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_chunk: (B, L, D)
        h0: (num_layers, B, H) or None

        returns:
          h_seq: (B, L, H)         hidden states for each time step
          hN:    (num_layers,B,H)  last hidden state
          x_hat: (B, L, D)         prediction of x_{t+1} from h_t
        """
        h0 = self._ensure_h0(x_chunk, h0)

        h_seq, hN = self.gru(x_chunk, h0)  # (B,L,H), (num_layers,B,H)
        h_seq = self.ln(h_seq)
        x_hat = self.pred_head(h_seq)      # (B,L,D)
        return h_seq, hN, x_hat

    def classify_from_states(self, h_end: torch.Tensor) -> torch.Tensor:
        """
        h_end: (B, N_trials, H) or (B, H)
        returns logits: (B, N_trials, 3) or (B, 3)
        """
        return self.cls_head(h_end)

    def classify_rt_from_seq(self, h_seq: torch.Tensor) -> torch.Tensor:
        """
        h_seq: (B, L, H)
        returns rt_logits: (B, L, 2) where classes are [standard, deviant]
        """
        return self.rt_head(h_seq)
