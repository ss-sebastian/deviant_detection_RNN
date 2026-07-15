from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from stimulus_encoding import (
    StimulusEncodingConfig,
    freq_to_bin_erb,
    make_erb_edges,
    render_trial_tokens_from_freqs,
)

from .common import load_blocks_or_single, load_prerendered_input_blocks


def compute_trial_gap_hz(
    freqs_block: torch.Tensor,
    y_pos_456: torch.Tensor,
) -> torch.Tensor:
    """
    Per-trial absolute frequency gap |f_dev - f_std| in Hz.

    The first tone is the standard frequency because each trial starts with
    standard tones before the deviant.
    """
    dev_idx = (y_pos_456.long() - 1).clamp(min=0, max=7)
    std_freq = freqs_block[:, 0]
    dev_freq = freqs_block.gather(1, dev_idx.unsqueeze(1)).squeeze(1)
    return (dev_freq - std_freq).abs().float()


def unpack_batch_with_optional_gap(
    batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
    raise ValueError(f"Unexpected batch structure: {type(batch)}")


def make_gap_weight_tensor(
    gap_hz_bt: Optional[torch.Tensor],
    power: float,
    ref_hz: float,
    max_weight: float,
) -> Optional[torch.Tensor]:
    """Convert per-trial frequency gaps into normalized training weights."""
    if gap_hz_bt is None or float(power) <= 0.0:
        return None

    gap = gap_hz_bt.float().clamp(min=1.0)
    weights = (float(ref_hz) / gap).pow(float(power))
    max_w = max(1.0, float(max_weight))
    min_w = 1.0 / max_w
    weights = torch.clamp(weights, min=min_w, max=max_w)
    return weights / weights.mean().clamp_min(1e-8)


class OnlineRenderDataset(Dataset):
    """
    Render symbolic Hz blocks into token-level inputs online.

    Returns one block per item:
    - x_flat: (T_tokens, D), where T_tokens = 10 * trial_T_tokens
    - y: (10,) labels in {4, 5, 6}
    - gap_hz: (10,) deviant-standard frequency gaps for optional weighting
    """

    def __init__(
        self,
        data_dir: Path,
        seed: int,
        tone_ms: int,
        isi_ms: int,
        ramp_ms: int,
        token_ms: int,
        f_min_hz: float,
        f_max_hz: float,
        n_bins: int,
        add_eos: bool,
        add_bos: bool,
        eos_mode: str,
        sigma_other_noise: float,
        p_other_noise: float,
        sigma_silence_noise: float,
        resample_noise_per_epoch: bool = False,
        quiet: bool = False,
        assert_labels: bool = True,
        encoding_mode: str = "onehot",
        sigma_rf: float = 1.0,
        rf_normalization: str = "peak",
        sigma_rf_noise: float = 0.0,
        rf_noise_per_token: bool = True,
        noise_mode: str = "per_token",
        noise_rho: float = 0.0,
        noise_resample_interval_epochs: int = 1,
    ):
        self.data_dir = data_dir
        self.seed = int(seed)

        self.tone_ms = int(tone_ms)
        self.isi_ms = int(isi_ms)
        self.ramp_ms = int(ramp_ms)
        self.assert_labels = bool(assert_labels)

        self.X, self.Y, self.layout = load_blocks_or_single(data_dir)
        self.B = int(self.X.shape[0])

        self.f_min_hz = float(f_min_hz)
        self.f_max_hz = float(f_max_hz)
        self.n_bins = int(n_bins)
        self.add_eos = bool(add_eos)
        self.add_bos = bool(add_bos)
        self.eos_mode = str(eos_mode)
        self.sigma_other_noise = float(sigma_other_noise)
        self.p_other_noise = float(p_other_noise)
        self.sigma_silence_noise = float(sigma_silence_noise)
        self.encoding_mode = str(encoding_mode)
        self.sigma_rf = float(sigma_rf)
        self.rf_normalization = str(rf_normalization)
        self.sigma_rf_noise = float(sigma_rf_noise)
        self.rf_noise_per_token = bool(rf_noise_per_token)
        self.noise_mode = str(noise_mode)
        self.noise_rho = float(noise_rho)
        self.resample_noise_per_epoch = bool(resample_noise_per_epoch)
        self.noise_resample_interval_epochs = max(1, int(noise_resample_interval_epochs))
        self.current_epoch = 0

        self.encoding_cfg = StimulusEncodingConfig(
            tone_ms=self.tone_ms,
            isi_ms=self.isi_ms,
            token_ms=int(token_ms),
            f_min_hz=self.f_min_hz,
            f_max_hz=self.f_max_hz,
            n_bins=self.n_bins,
            add_eos=self.add_eos,
            add_bos=self.add_bos,
            eos_mode=self.eos_mode,
            sigma_other_noise=self.sigma_other_noise,
            p_other_noise=self.p_other_noise,
            sigma_silence_noise=self.sigma_silence_noise,
            encoding_mode=self.encoding_mode,
            sigma_rf=self.sigma_rf,
            rf_normalization=self.rf_normalization,
            sigma_rf_noise=self.sigma_rf_noise,
            rf_noise_per_token=self.rf_noise_per_token,
            noise_mode=self.noise_mode,
            noise_rho=self.noise_rho,
        )

        self.token_ms = int(self.encoding_cfg.token_ms)
        self.tone_T = int(self.encoding_cfg.tone_T)
        self.isi_T = int(self.encoding_cfg.isi_T)
        self.step_T = self.tone_T + self.isi_T
        self.trial_T_ms = int(self.encoding_cfg.trial_T_ms)
        self.trial_T_tokens = int(self.encoding_cfg.trial_T_tokens)
        self.T = 10 * self.trial_T_tokens
        self.eos_dim = 1 if self.add_eos else 0
        self.bos_dim = 1 if self.add_bos else 0
        self.input_dim = int(self.encoding_cfg.input_dim)
        self.edges_erb = make_erb_edges(self.f_min_hz, self.f_max_hz, self.n_bins)

        if not quiet:
            x_hz = self.X.numpy()
            bins = np.zeros_like(x_hz, dtype=np.int32)
            for b in range(x_hz.shape[0]):
                for t in range(x_hz.shape[1]):
                    for i in range(x_hz.shape[2]):
                        bins[b, t, i] = freq_to_bin_erb(float(x_hz[b, t, i]), self.edges_erb)

            clip_lo = (bins == 0).mean()
            clip_hi = (bins == (self.n_bins - 1)).mean()
            uniq_std = len(np.unique(bins[:, :, 0:3]))
            uniq_456 = len(np.unique(bins[:, :, 3:6]))

            std_bin = bins[:, :, 0]
            pred = np.abs(bins[:, :, 3:6] - std_bin[:, :, None]).argmax(axis=2) + 4
            acc_bin = (pred == self.Y.numpy()).mean()

            print(
                f"[sanity bins] clip_lo={clip_lo:.3f} clip_hi={clip_hi:.3f} "
                f"uniq_std={uniq_std} uniq_456={uniq_456} rule_acc_on_bins={acc_bin:.3f}"
            )

    def __len__(self) -> int:
        return self.B

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _seed_for(self, idx: int, trial: int) -> int:
        noise_epoch = int(self.current_epoch) // int(self.noise_resample_interval_epochs)
        epoch_offset = (noise_epoch * 1_000_003) if self.resample_noise_per_epoch else 0
        return self.seed + epoch_offset + idx * 1000 + trial * 17

    def _render_trial_tokens_onehot(self, freqs_8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return render_trial_tokens_from_freqs(
            freqs_8=freqs_8,
            cfg=self.encoding_cfg,
            rng=rng,
            edges_erb=self.edges_erb,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.Y[idx]
        freqs_block_t = self.X[idx].float()
        freqs_block = freqs_block_t.numpy().astype(np.float32)

        out = np.zeros((10, self.trial_T_tokens, self.input_dim), dtype=np.float32)
        for trial in range(10):
            rng = np.random.default_rng(self._seed_for(idx, trial))
            out[trial] = self._render_trial_tokens_onehot(freqs_block[trial], rng=rng)

        x = torch.from_numpy(out.reshape(10 * self.trial_T_tokens, self.input_dim))
        gap_hz = compute_trial_gap_hz(freqs_block_t, y)
        if getattr(self, "assert_labels", False):
            if not torch.all((y == 4) | (y == 5) | (y == 6)):
                raise RuntimeError(f"[dataset assert] Bad y at idx={idx}: unique={torch.unique(y).tolist()}")
        return x, y, gap_hz


class PreRenderedBlockDataset(Dataset):
    """Use pre-rendered token tensors while preserving labels and gap diagnostics."""

    def __init__(
        self,
        data_dir: Path,
        seed: int,
        tone_ms: int,
        isi_ms: int,
        ramp_ms: int,
        token_ms: int,
        f_min_hz: float,
        f_max_hz: float,
        n_bins: int,
        add_eos: bool,
        add_bos: bool,
        eos_mode: str,
        sigma_other_noise: float,
        p_other_noise: float,
        sigma_silence_noise: float,
        quiet: bool = False,
        assert_labels: bool = True,
        encoding_mode: str = "onehot",
        sigma_rf: float = 1.0,
        rf_normalization: str = "peak",
        sigma_rf_noise: float = 0.0,
        rf_noise_per_token: bool = True,
        noise_mode: str = "per_token",
        noise_rho: float = 0.0,
    ):
        self.data_dir = data_dir
        self.seed = int(seed)
        self.tone_ms = int(tone_ms)
        self.isi_ms = int(isi_ms)
        self.ramp_ms = int(ramp_ms)
        self.assert_labels = bool(assert_labels)

        self.X, self.Y, self.layout = load_blocks_or_single(data_dir)
        self.rendered_X = load_prerendered_input_blocks(data_dir)
        self.B = int(self.X.shape[0])
        if int(self.rendered_X.shape[0]) != self.B:
            raise ValueError(
                f"Prerendered block count mismatch: symbolic={self.B} rendered={int(self.rendered_X.shape[0])}"
            )

        self.f_min_hz = float(f_min_hz)
        self.f_max_hz = float(f_max_hz)
        self.n_bins = int(n_bins)
        self.add_eos = bool(add_eos)
        self.add_bos = bool(add_bos)
        self.eos_mode = str(eos_mode)
        self.sigma_other_noise = float(sigma_other_noise)
        self.p_other_noise = float(p_other_noise)
        self.sigma_silence_noise = float(sigma_silence_noise)
        self.encoding_mode = str(encoding_mode)
        self.sigma_rf = float(sigma_rf)
        self.rf_normalization = str(rf_normalization)
        self.sigma_rf_noise = float(sigma_rf_noise)
        self.rf_noise_per_token = bool(rf_noise_per_token)
        self.noise_mode = str(noise_mode)
        self.noise_rho = float(noise_rho)
        self.current_epoch = 0

        self.encoding_cfg = StimulusEncodingConfig(
            tone_ms=self.tone_ms,
            isi_ms=self.isi_ms,
            token_ms=int(token_ms),
            f_min_hz=self.f_min_hz,
            f_max_hz=self.f_max_hz,
            n_bins=self.n_bins,
            add_eos=self.add_eos,
            add_bos=self.add_bos,
            eos_mode=self.eos_mode,
            sigma_other_noise=self.sigma_other_noise,
            p_other_noise=self.p_other_noise,
            sigma_silence_noise=self.sigma_silence_noise,
            encoding_mode=self.encoding_mode,
            sigma_rf=self.sigma_rf,
            rf_normalization=self.rf_normalization,
            sigma_rf_noise=self.sigma_rf_noise,
            rf_noise_per_token=self.rf_noise_per_token,
            noise_mode=self.noise_mode,
            noise_rho=self.noise_rho,
        )

        self.token_ms = int(self.encoding_cfg.token_ms)
        self.tone_T = int(self.encoding_cfg.tone_T)
        self.isi_T = int(self.encoding_cfg.isi_T)
        self.step_T = self.tone_T + self.isi_T
        self.trial_T_ms = int(self.encoding_cfg.trial_T_ms)
        self.trial_T_tokens = int(self.encoding_cfg.trial_T_tokens)
        self.T = 10 * self.trial_T_tokens
        self.eos_dim = 1 if self.add_eos else 0
        self.bos_dim = 1 if self.add_bos else 0
        self.input_dim = int(self.encoding_cfg.input_dim)
        if tuple(self.rendered_X.shape[1:]) != (self.T, self.input_dim):
            raise ValueError(
                f"Prerendered tensor shape mismatch: expected {(self.T, self.input_dim)} "
                f"got {tuple(self.rendered_X.shape[1:])}"
            )
        self.edges_erb = make_erb_edges(self.f_min_hz, self.f_max_hz, self.n_bins)

        if not quiet:
            print(
                f"[prerendered dataset] loaded rendered_input_blocks.pt shape={tuple(self.rendered_X.shape)} "
                f"trial_T_tokens={self.trial_T_tokens} input_dim={self.input_dim}"
            )

    def __len__(self) -> int:
        return self.B

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.rendered_X[idx]
        y = self.Y[idx]
        gap_hz = compute_trial_gap_hz(self.X[idx].float(), y)
        if getattr(self, "assert_labels", False):
            if not torch.all((y == 4) | (y == 5) | (y == 6)):
                raise RuntimeError(f"[dataset assert] Bad y at idx={idx}: unique={torch.unique(y).tolist()}")
        return x, y, gap_hz


class TrialwiseRenderDataset(Dataset):
    """Expose the same rendered representation as independent trial items."""

    def __init__(self, block_ds: OnlineRenderDataset):
        self.block_ds = block_ds
        self.B = int(block_ds.B)
        self.trials_per_block = 10
        self.N = self.B * self.trials_per_block
        self.trial_T_tokens = int(block_ds.trial_T_tokens)
        self.input_dim = int(block_ds.input_dim)
        self.Y = block_ds.Y

    def __len__(self) -> int:
        return self.N

    def set_epoch(self, epoch: int) -> None:
        self.block_ds.set_epoch(epoch)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bidx = int(idx // self.trials_per_block)
        tidx = int(idx % self.trials_per_block)
        x_block, y_block, _ = self.block_ds[bidx]
        x_trial = x_block.view(self.trials_per_block, self.trial_T_tokens, self.input_dim)[tidx]
        y_trial = y_block[tidx]
        return x_trial, y_trial

