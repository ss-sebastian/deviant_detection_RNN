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
    One GRU backbone, one shared 3-class head (classes correspond to dev_pos {4,5,6}):

      - token_head: logits at each token (B,L,3)
      - trial_end:  logits at selected end states (B,N_trials,3) using the same head

    No binary rt_head anymore. RT is computed from token logits during evaluation.
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
        self.head = nn.Linear(cfg.hidden_dim, 3)  # classes: {4,5,6} -> indices {0,1,2}

    def _ensure_h0(self, x_chunk: torch.Tensor, h0: Optional[torch.Tensor]) -> torch.Tensor:
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
        x_chunk: torch.Tensor,           # (B,L,D)
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
          h_seq: (B,L,H)
          hN:    (num_layers,B,H)
        """
        h0 = self._ensure_h0(x_chunk, h0)
        h_seq, hN = self.gru(x_chunk, h0)
        h_seq = self.ln(h_seq)
        return h_seq, hN

    def classify_tokens(self, h_seq: torch.Tensor) -> torch.Tensor:
        """h_seq: (B,L,H) -> logits: (B,L,3)"""
        return self.head(h_seq)

    def classify_from_states(self, h_end: torch.Tensor) -> torch.Tensor:
        """h_end: (B,N,H) or (B,H) -> logits: (B,N,3) or (B,3)"""
        return self.head(h_end)
