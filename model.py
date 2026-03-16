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
    dropout: float = 0.0
    layer_norm: bool = True


# Type alias for LSTM state
LSTMState = Tuple[torch.Tensor, torch.Tensor]


class PredictiveGRU(nn.Module):
    """
    LSTM backbone with a shared 3-class head (classes correspond to dev_pos {4,5,6}):

      - token_head: logits at each token (B,L,3)
      - trial_end:  logits at selected end states (B,N_trials,3) using the same head

    h0 / hN are now (h, c) tuples as required by nn.LSTM.
    The class is kept as PredictiveGRU for compatibility with existing checkpoints/code.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.rnn = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.ln = nn.LayerNorm(cfg.hidden_dim) if cfg.layer_norm else nn.Identity()
        self.head = nn.Linear(cfg.hidden_dim, 3)

    def _make_zero_state(self, x_chunk: torch.Tensor) -> LSTMState:
        """Create (h_0, c_0) zero tensors matching batch size and device."""
        B = int(x_chunk.shape[0])
        device = x_chunk.device
        dtype = x_chunk.dtype
        shape = (self.cfg.num_layers, B, self.cfg.hidden_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def _ensure_state(
        self,
        x_chunk: torch.Tensor,
        h0: Optional[LSTMState],
    ) -> LSTMState:
        """Return a valid (h, c) state, initialising to zeros if None."""
        if h0 is None:
            return self._make_zero_state(x_chunk)
        h, c = h0
        device = x_chunk.device
        dtype = x_chunk.dtype
        return h.to(device=device, dtype=dtype), c.to(device=device, dtype=dtype)

    def forward_chunk(
        self,
        x_chunk: torch.Tensor,            # (B, L, D)
        h0: Optional[LSTMState] = None,
    ) -> Tuple[torch.Tensor, LSTMState]:
        """
        Returns:
          h_seq : (B, L, H)   — layer-normed output at every token
          hN    : (h, c) tuple, each (num_layers, B, H)  — state to pass to next chunk
        """
        state = self._ensure_state(x_chunk, h0)
        h_seq, (hN, cN) = self.rnn(x_chunk, state)
        h_seq = self.ln(h_seq)
        return h_seq, (hN, cN)

    def classify_tokens(self, h_seq: torch.Tensor) -> torch.Tensor:
        """h_seq: (B, L, H)  ->  logits: (B, L, 3)"""
        return self.head(h_seq)

    def classify_from_states(self, h_end: torch.Tensor) -> torch.Tensor:
        """h_end: (B, N, H) or (B, H)  ->  logits: (B, N, 3) or (B, 3)"""
        return self.head(h_end)