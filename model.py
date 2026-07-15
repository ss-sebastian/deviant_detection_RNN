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
    num_classes: int = 3
    next_tone_num_classes: int = 2
    num_layers: int = 1
    dropout: float = 0.0  # GRU dropout only applies when num_layers>1
    layer_norm: bool = True
    hidden_noise_std: float = 0.0
    use_stop_head: bool = False
    use_event_head: bool = False
    use_next_tone_head: bool = False
    # --- new: response head + position embedding ---
    use_response_head: bool = False
    add_tone_position_embedding: bool = False
    tone_position_embed_dim: int = 16  # dim of learned position embedding


class PredictiveGRU(nn.Module):
    """
    One GRU backbone, one shared class head.

      - token_head: logits at each token (B,L,C)
      - trial_end:  logits at selected end states (B,N_trials,C) using the same head

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
        self.head = nn.Linear(cfg.hidden_dim, int(cfg.num_classes))
        self.stop_head = nn.Linear(cfg.hidden_dim, 1) if bool(cfg.use_stop_head) else None
        self.event_head = nn.Linear(cfg.hidden_dim, 1) if bool(cfg.use_event_head) else None
        self.next_tone_head = (
            nn.Linear(cfg.hidden_dim, int(cfg.next_tone_num_classes))
            if bool(cfg.use_next_tone_head)
            else None
        )

        # --- new: response head ---
        self.response_head = nn.Linear(cfg.hidden_dim, 1) if bool(cfg.use_response_head) else None

        # --- new: tone-position embedding (injected into hidden state) ---
        if bool(cfg.add_tone_position_embedding):
            self.tone_pos_embed = nn.Embedding(8, cfg.tone_position_embed_dim)
            self.tone_pos_proj = nn.Linear(cfg.tone_position_embed_dim, cfg.hidden_dim)
        else:
            self.tone_pos_embed = None
            self.tone_pos_proj = None

    def _apply_training_hidden_noise(self, h_seq: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (float(self.cfg.hidden_noise_std) <= 0.0):
            return h_seq
        noise = torch.randn_like(h_seq) * float(self.cfg.hidden_noise_std)
        return h_seq + noise

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
        h_seq = self._apply_training_hidden_noise(h_seq)
        return h_seq, hN

    def classify_tokens(self, h_seq: torch.Tensor) -> torch.Tensor:
        """h_seq: (B,L,H) -> logits: (B,L,C)"""
        return self.head(h_seq)

    def classify_from_states(self, h_end: torch.Tensor) -> torch.Tensor:
        """h_end: (B,N,H) or (B,H) -> logits: (B,N,C) or (B,C)"""
        return self.head(h_end)

    def classify_stop(self, h_seq: torch.Tensor) -> Optional[torch.Tensor]:
        """h_seq: (B,L,H) -> stop_logits: (B,L,1), or None when head disabled."""
        if self.stop_head is None:
            return None
        return self.stop_head(h_seq)

    def classify_event(self, h_seq: torch.Tensor) -> Optional[torch.Tensor]:
        """h_seq: (B,L,H) -> event_logits: (B,L,1), or None when disabled."""
        if self.event_head is None:
            return None
        return self.event_head(h_seq)

    def classify_next_tone(self, h_seq: torch.Tensor) -> Optional[torch.Tensor]:
        """h_seq: (B,L,H) -> next-tone logits: (B,L,C), or None when disabled."""
        if self.next_tone_head is None:
            return None
        return self.next_tone_head(h_seq)

    def classify_response(self, h_seq: torch.Tensor) -> Optional[torch.Tensor]:
        """h_seq: (B,L,H) -> response_logits: (B,L,1). Returns p_respond after sigmoid."""
        if self.response_head is None:
            return None
        return self.response_head(h_seq)

    def inject_tone_position(self, h_seq: torch.Tensor, tone_positions: torch.Tensor) -> torch.Tensor:
        """Add learned tone-position embedding to hidden states.

        h_seq: (B, L, H)
        tone_positions: (B, L) with values in {0..7}, or -1 for no position
        Returns: h_seq with position embedding added
        """
        if self.tone_pos_embed is None or self.tone_pos_proj is None:
            return h_seq
        mask = (tone_positions >= 0) & (tone_positions < 8)
        if not mask.any():
            return h_seq
        emb = self.tone_pos_embed(tone_positions.clamp(0, 7))  # (B, L, embed_dim)
        proj = self.tone_pos_proj(emb)  # (B, L, H)
        return h_seq + proj * mask.unsqueeze(-1).float()
