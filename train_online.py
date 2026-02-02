# train_online.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import PredictiveGRU, ModelConfig


# -------------------------
# Utilities: loading + shapes
# -------------------------
def _load_pt(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.load(path, map_location="cpu")


def labels_to_class_index(y_456: torch.Tensor) -> torch.Tensor:
    """
    y_456: (...,) values in {4,5,6} (1-indexed position)
    map to {0,1,2}
    """
    return (y_456 - 4).long()


def infer_end_indices_from_T(T: int, trials_per_block: int = 10) -> torch.Tensor:
    if T % trials_per_block != 0:
        raise ValueError(f"Cannot infer trial length: T={T} not divisible by {trials_per_block}")
    trial_T = T // trials_per_block
    return torch.tensor([(i + 1) * trial_T - 1 for i in range(trials_per_block)], dtype=torch.long)


# -------------------------
# Feature C renderer (Torch, fast)
# -------------------------
def hz_to_erb_rate_torch(f_hz: torch.Tensor) -> torch.Tensor:
    # ERB-rate: 21.4 * log10(1 + 0.00437 f)
    return 21.4 * torch.log10(1.0 + 0.00437 * f_hz)


def make_linear_envelope_ms_torch(tone_ms: int, ramp_ms: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if ramp_ms <= 0:
        return torch.ones((tone_ms,), dtype=torch.float32, device=device)

    ramp_ms = int(ramp_ms)
    if 2 * ramp_ms > tone_ms:
        ramp_ms = tone_ms // 2

    env = torch.ones((tone_ms,), dtype=torch.float32, device=device)
    if ramp_ms > 0:
        up = torch.linspace(0.0, 1.0, steps=ramp_ms, dtype=torch.float32, device=device)
        down = torch.linspace(1.0, 0.0, steps=ramp_ms, dtype=torch.float32, device=device)
        env[:ramp_ms] = up
        env[-ramp_ms:] = down
    return env


def render_trial_feature_C_torch(
    freqs_8_hz: torch.Tensor,   # (8,) float
    tone_ms: int,
    isi_ms: int,
    ramp_ms: int,
    sigma_tone: float,
    sigma_silence: float,
    mu_silence: float,
    env_tone: torch.Tensor,     # (tone_ms,1) precomputed on cpu
    g: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns (trial_T, 1) dtype
      - tone segments: x = env * ERB(f) + N(0, sigma_tone)
      - silence segments: x = mu_silence + N(0, sigma_silence)
    """
    assert tuple(freqs_8_hz.shape) == (8,)
    tone_ms = int(tone_ms)
    isi_ms = int(isi_ms)
    ramp_ms = int(ramp_ms)

    erb = hz_to_erb_rate_torch(freqs_8_hz.float())  # (8,)
    trial_T = 7 * (tone_ms + isi_ms) + tone_ms

    out = (mu_silence + torch.randn((trial_T, 1), generator=g, dtype=torch.float32) * float(sigma_silence)).to(dtype)

    t = 0
    for i in range(8):
        tone_slice = slice(t, t + tone_ms)
        base = env_tone * float(erb[i].item())  # env is float32, base float32
        noise = torch.randn((tone_ms, 1), generator=g, dtype=torch.float32) * float(sigma_tone)
        out[tone_slice] = (base + noise).to(dtype)
        t += tone_ms
        if i < 7:
            t += isi_ms
    return out


# -------------------------
# Gammatone renderer (NumPy+SciPy, slow but faithful)
# -------------------------
def _require_scipy():
    try:
        from scipy.signal import lfilter, hilbert  # noqa: F401
        return True
    except Exception as e:
        raise ImportError(
            "Gammatone frontend requires scipy.\n"
            "Install:\n"
            "  conda install scipy\n"
            "or\n"
            "  pip install scipy\n"
        ) from e


def hz_to_erb_rate_np(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * f_hz)


def erb_rate_to_hz_np(erb: np.ndarray) -> np.ndarray:
    return (10 ** (erb / 21.4) - 1.0) / 0.00437


def make_linear_envelope_samples(tone_ms: int, ramp_ms: int, sr: int) -> np.ndarray:
    n = int(round(tone_ms * sr / 1000.0))
    if ramp_ms <= 0:
        return np.ones(n, dtype=np.float32)

    ramp_n = int(round(ramp_ms * sr / 1000.0))
    ramp_n = min(ramp_n, n // 2)

    env = np.ones(n, dtype=np.float32)
    if ramp_n > 0:
        env[:ramp_n] = np.linspace(0.0, 1.0, ramp_n, dtype=np.float32)
        env[-ramp_n:] = np.linspace(1.0, 0.0, ramp_n, dtype=np.float32)
    return env


def synth_trial_waveform(
    freqs_8_hz: np.ndarray,
    sr: int,
    tone_ms: int,
    isi_ms: int,
    ramp_ms: int,
    tone_amp: float,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    assert freqs_8_hz.shape == (8,)

    tone_n = int(round(tone_ms * sr / 1000.0))
    isi_n = int(round(isi_ms * sr / 1000.0))
    env = make_linear_envelope_samples(tone_ms=tone_ms, ramp_ms=ramp_ms, sr=sr)

    chunks = []
    for i in range(8):
        t = (np.arange(tone_n, dtype=np.float32) / float(sr))
        tone = (tone_amp * np.sin(2.0 * np.pi * float(freqs_8_hz[i]) * t)).astype(np.float32)
        tone = tone * env

        if noise_sigma > 0:
            tone = tone + rng.normal(0.0, noise_sigma, size=tone.shape).astype(np.float32)
        chunks.append(tone)

        if i < 7:
            if noise_sigma > 0:
                isi = rng.normal(0.0, noise_sigma, size=(isi_n,)).astype(np.float32)
            else:
                isi = np.zeros((isi_n,), dtype=np.float32)
            chunks.append(isi)

    return np.concatenate(chunks, axis=0)


def gammatone_impulse_response(sr: int, cf_hz: float, n_samples: int, order: int = 4) -> np.ndarray:
    """
    Lightweight FIR approximation of a gammatone-like filter.
    """
    t = np.arange(n_samples, dtype=np.float32) / float(sr)

    erb_bw = 24.7 * (4.37e-3 * cf_hz + 1.0)
    b = 1.019 * 2.0 * np.pi * erb_bw

    h = (t ** (order - 1)) * np.exp(-b * t) * np.cos(2.0 * np.pi * cf_hz * t)
    h = h.astype(np.float32)
    h = h / (np.sqrt(np.sum(h ** 2)) + 1e-8)
    return h


def gammatone_filterbank_envelope(
    x: np.ndarray,
    sr: int,
    cfs: np.ndarray,
    ir_ms: int = 40,
    order: int = 4,
) -> np.ndarray:
    _require_scipy()
    from scipy.signal import lfilter, hilbert

    ir_n = max(int(round(ir_ms * sr / 1000.0)), 8)
    outs = []
    for cf in cfs:
        h = gammatone_impulse_response(sr=sr, cf_hz=float(cf), n_samples=ir_n, order=order)
        y = lfilter(h, [1.0], x).astype(np.float32)
        env = np.abs(hilbert(y)).astype(np.float32)
        outs.append(env)
    return np.stack(outs, axis=-1)  # (T_samples, n_bands)


def downsample_to_1ms_mean(Y: np.ndarray, sr: int, target_ms: int) -> np.ndarray:
    samples_per_ms = sr / 1000.0
    T_ms = int(round(Y.shape[0] / samples_per_ms))
    out = np.zeros((T_ms, Y.shape[1]), dtype=np.float32)

    for m in range(T_ms):
        a = int(round(m * samples_per_ms))
        b = int(round((m + 1) * samples_per_ms))
        b = min(b, Y.shape[0])
        if b <= a:
            b = min(a + 1, Y.shape[0])
        out[m] = Y[a:b].mean(axis=0)

    if out.shape[0] > target_ms:
        return out[:target_ms]
    if out.shape[0] < target_ms:
        pad = np.zeros((target_ms - out.shape[0], out.shape[1]), dtype=np.float32)
        return np.concatenate([out, pad], axis=0)
    return out


def moving_average_ms(X: np.ndarray, win_ms: int) -> np.ndarray:
    win_ms = int(win_ms)
    if win_ms <= 1:
        return X
    kernel = np.ones((win_ms,), dtype=np.float32) / float(win_ms)
    Y = np.empty_like(X)
    for k in range(X.shape[1]):
        Y[:, k] = np.convolve(X[:, k], kernel, mode="same").astype(np.float32)
    return Y


# -------------------------
# Load compact blocks
# -------------------------
def load_blocks_or_single(in_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Supports:
      - input_blocks.pt (B,10,8), labels_blocks.pt (B,10)
      - input_tensor.pt (1,10,8), labels_tensor.pt (1,10)
    Returns X (B,10,8), Y (B,10), layout
    """
    xb = in_dir / "input_blocks.pt"
    yb = in_dir / "labels_blocks.pt"
    xs = in_dir / "input_tensor.pt"
    ys = in_dir / "labels_tensor.pt"

    if xb.exists() and yb.exists():
        X = torch.load(xb, map_location="cpu").float()
        Y = torch.load(yb, map_location="cpu").long()
        if X.ndim != 3 or tuple(X.shape[1:]) != (10, 8):
            raise ValueError(f"Expected input_blocks (B,10,8), got {tuple(X.shape)}")
        if Y.ndim != 2 or tuple(Y.shape[1:]) != (10,):
            raise ValueError(f"Expected labels_blocks (B,10), got {tuple(Y.shape)}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Block count mismatch: X={X.shape[0]} vs Y={Y.shape[0]}")
        return X, Y, "blocks"

    if xs.exists() and ys.exists():
        X = torch.load(xs, map_location="cpu").float()
        Y = torch.load(ys, map_location="cpu").long()
        if X.ndim != 3 or tuple(X.shape) != (1, 10, 8):
            raise ValueError(f"Expected input_tensor (1,10,8), got {tuple(X.shape)}")
        if Y.ndim != 2 or tuple(Y.shape) != (1, 10):
            raise ValueError(f"Expected labels_tensor (1,10), got {tuple(Y.shape)}")
        return X, Y, "single"

    raise FileNotFoundError("Need input_blocks.pt/labels_blocks.pt OR input_tensor.pt/labels_tensor.pt in --data_dir")


# -------------------------
# Dataset: on-the-fly rendering
# -------------------------
Frontend = Literal["feature_c", "gammatone"]


class OnlineRenderDataset(Dataset):
    """
    Returns one block per item:
      x_flat: (T, D) where T = 10 * trial_T_ms
      y:      (10,) labels in {4,5,6}

    Rendering is deterministic given (seed, idx).
    """

    def __init__(
        self,
        data_dir: Path,
        frontend: Frontend,
        seed: int,
        # shared trial timing
        tone_ms: int,
        isi_ms: int,
        ramp_ms: int,
        # feature_c params
        sigma_tone: float = 0.05,
        sigma_silence: float = 0.05,
        mu_silence: float = 0.0,
        feature_dtype: str = "float32",
        # gammatone params
        sr: int = 16000,
        tone_amp: float = 0.1,
        noise_sigma: float = 0.02,
        n_bands: int = 32,
        cf_min: float = 300.0,
        cf_max: float = 4000.0,
        ir_ms: int = 40,
        order: int = 4,
        log_compress: bool = False,
        log_eps: float = 1e-4,
        smooth_ms: int = 7,
    ):
        self.data_dir = data_dir
        self.frontend = frontend
        self.seed = int(seed)

        self.tone_ms = int(tone_ms)
        self.isi_ms = int(isi_ms)
        self.ramp_ms = int(ramp_ms)

        self.X, self.Y, self.layout = load_blocks_or_single(data_dir)
        self.B = int(self.X.shape[0])

        self.trial_T_ms = 7 * (self.tone_ms + self.isi_ms) + self.tone_ms
        if self.trial_T_ms <= 0:
            raise ValueError("Invalid tone_ms/isi_ms leading to non-positive trial length.")

        # feature_c setup
        self.sigma_tone = float(sigma_tone)
        self.sigma_silence = float(sigma_silence)
        self.mu_silence = float(mu_silence)

        if feature_dtype == "float16":
            self.feature_dtype = torch.float16
        else:
            self.feature_dtype = torch.float32

        env = make_linear_envelope_ms_torch(self.tone_ms, self.ramp_ms, device=torch.device("cpu")).view(self.tone_ms, 1)
        self.env_tone = env  # float32 on cpu

        # gammatone setup
        self.sr = int(sr)
        self.tone_amp = float(tone_amp)
        self.noise_sigma = float(noise_sigma)
        self.n_bands = int(n_bands)
        self.cf_min = float(cf_min)
        self.cf_max = float(cf_max)
        self.ir_ms = int(ir_ms)
        self.order = int(order)
        self.log_compress = bool(log_compress)
        self.log_eps = float(log_eps)
        self.smooth_ms = int(smooth_ms)

        if self.frontend == "gammatone":
            # precompute center frequencies on ERB-rate scale
            erb_min = hz_to_erb_rate_np(np.array([self.cf_min], dtype=np.float32))[0]
            erb_max = hz_to_erb_rate_np(np.array([self.cf_max], dtype=np.float32))[0]
            erb_grid = np.linspace(erb_min, erb_max, self.n_bands, dtype=np.float32)
            self.cfs = erb_rate_to_hz_np(erb_grid).astype(np.float32)

        # inferred dimensions
        if self.frontend == "feature_c":
            self.input_dim = 1
        elif self.frontend == "gammatone":
            self.input_dim = self.n_bands
        else:
            raise ValueError(f"Unknown frontend: {self.frontend}")

        self.T = 10 * self.trial_T_ms

    def __len__(self) -> int:
        return self.B

    def _seed_for(self, idx: int, trial: int) -> int:
        # stable per (block, trial)
        return self.seed + idx * 1000 + trial * 17

    def _render_block_feature_c(self, idx: int) -> torch.Tensor:
        out = torch.empty((10, self.trial_T_ms, 1), dtype=self.feature_dtype)
        freqs_block = self.X[idx]  # (10,8) float
        for t in range(10):
            g = torch.Generator().manual_seed(self._seed_for(idx, t))
            out[t] = render_trial_feature_C_torch(
                freqs_8_hz=freqs_block[t],
                tone_ms=self.tone_ms,
                isi_ms=self.isi_ms,
                ramp_ms=self.ramp_ms,
                sigma_tone=self.sigma_tone,
                sigma_silence=self.sigma_silence,
                mu_silence=self.mu_silence,
                env_tone=self.env_tone,
                g=g,
                dtype=self.feature_dtype,
            )
        return out.reshape(10 * self.trial_T_ms, 1)

    def _render_block_gammatone(self, idx: int) -> torch.Tensor:
        # render on CPU with NumPy
        out = np.zeros((10, self.trial_T_ms, self.n_bands), dtype=np.float32)
        freqs_block = self.X[idx].numpy().astype(np.float32)  # (10,8)

        for t in range(10):
            rng = np.random.default_rng(self._seed_for(idx, t))
            wav = synth_trial_waveform(
                freqs_8_hz=freqs_block[t],
                sr=self.sr,
                tone_ms=self.tone_ms,
                isi_ms=self.isi_ms,
                ramp_ms=self.ramp_ms,
                tone_amp=self.tone_amp,
                noise_sigma=self.noise_sigma,
                rng=rng,
            )
            env_fb = gammatone_filterbank_envelope(
                x=wav,
                sr=self.sr,
                cfs=self.cfs,
                ir_ms=self.ir_ms,
                order=self.order,
            )
            env_ms = downsample_to_1ms_mean(env_fb, sr=self.sr, target_ms=self.trial_T_ms)

            if self.log_compress:
                env_ms = np.log(env_ms + self.log_eps).astype(np.float32)

            env_ms = moving_average_ms(env_ms, win_ms=self.smooth_ms)
            out[t] = env_ms

        out_t = torch.from_numpy(out)  # (10,trial_T_ms,n_bands)
        return out_t.reshape(10 * self.trial_T_ms, self.n_bands)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.Y[idx]  # (10,) long in {4,5,6}

        if self.frontend == "feature_c":
            x = self._render_block_feature_c(idx)
        elif self.frontend == "gammatone":
            x = self._render_block_gammatone(idx)
        else:
            raise ValueError(f"Unknown frontend: {self.frontend}")

        return x, y


# -------------------------
# Train / Eval with TBPTT
# -------------------------
def _run_block_through_tbptt(
    model: PredictiveGRU,
    x: torch.Tensor,                 # (B,T,D)
    end_idx: torch.Tensor,           # (10,) on device
    chunk_len: int,
    compute_pred_loss: bool,
    huber: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      h_end: (B,10,H) trial-end hidden states
      pred_loss_accum: scalar tensor (0 if compute_pred_loss=False)
    """
    B, T, D = x.shape
    h = None
    collected = []
    pred_loss_accum = x.new_tensor(0.0)

    for s in range(0, T, chunk_len):
        e = min(s + chunk_len, T)   # exclusive
        x_in = x[:, s:e, :]
        h_seq, h, x_hat = model.forward_chunk(x_in, h0=h)  # (B,L,H), (B,L,D)

        L = e - s
        if compute_pred_loss and L >= 2:
            # robust to non-contiguous tensors: reshape instead of view
            pred_in = x_hat[:, :-1, :].reshape(-1, D)
            pred_tg = x[:, s+1:e, :].reshape(-1, D)
            pred_loss_accum = pred_loss_accum + huber(pred_in, pred_tg)

        mask = (end_idx >= s) & (end_idx < e)
        if mask.any():
            rel = (end_idx[mask] - s).long()
            hs = h_seq.index_select(dim=1, index=rel)
            collected.append(hs)

        h = h.detach()

    if len(collected) == 0:
        raise RuntimeError("No trial-end states collected. Check end_idx and T.")

    h_end = torch.cat(collected, dim=1)  # (B,10,H)
    return h_end, pred_loss_accum


def train_one_epoch(
    model: PredictiveGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    chunk_len: int,
    lambda_pred: float,
    grad_clip: float,
) -> dict:
    model.train()
    huber = nn.SmoothL1Loss(reduction="mean")
    ce = nn.CrossEntropyLoss(reduction="mean")

    total_cls = 0.0
    total_pred = 0.0
    total = 0
    correct = 0

    compute_pred = lambda_pred > 0

    for x, y in loader:
        x = x.to(device)  # (B,T,D)
        y = y.to(device)  # (B,10)
        B, T, D = x.shape

        end_idx = infer_end_indices_from_T(T, trials_per_block=10).to(device)

        optimizer.zero_grad(set_to_none=True)

        h_end, pred_loss_accum = _run_block_through_tbptt(
            model=model,
            x=x,
            end_idx=end_idx,
            chunk_len=chunk_len,
            compute_pred_loss=compute_pred,
            huber=huber,
        )

        logits = model.classify_from_states(h_end)  # (B,10,3)
        y_cls = labels_to_class_index(y)

        cls_loss = ce(logits.reshape(-1, 3), y_cls.reshape(-1))
        total_loss = cls_loss + float(lambda_pred) * pred_loss_accum

        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_cls += float(cls_loss.item()) * B
        total_pred += float(pred_loss_accum.item()) * B

        pred = logits.argmax(dim=-1)
        correct += int((pred == y_cls).sum().item())
        total += int(y_cls.numel())

    denom = max(1, len(loader.dataset))
    return {
        "cls_loss": total_cls / denom,
        "pred_loss": total_pred / denom,
        "total_loss": (total_cls + float(lambda_pred) * total_pred) / denom,
        "acc": correct / max(1, total),
    }


@torch.no_grad()
def evaluate(
    model: PredictiveGRU,
    loader: DataLoader,
    device: torch.device,
    chunk_len: int,
    lambda_pred: float,
) -> dict:
    model.eval()
    huber = nn.SmoothL1Loss(reduction="mean")
    ce = nn.CrossEntropyLoss(reduction="mean")

    total_cls = 0.0
    total_pred = 0.0
    total = 0
    correct = 0

    compute_pred = lambda_pred > 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        B, T, D = x.shape

        end_idx = infer_end_indices_from_T(T, trials_per_block=10).to(device)

        h_end, pred_loss_accum = _run_block_through_tbptt(
            model=model,
            x=x,
            end_idx=end_idx,
            chunk_len=chunk_len,
            compute_pred_loss=compute_pred,
            huber=huber,
        )

        logits = model.classify_from_states(h_end)
        y_cls = labels_to_class_index(y)

        cls_loss = ce(logits.reshape(-1, 3), y_cls.reshape(-1))
        total_cls += float(cls_loss.item()) * B
        total_pred += float(pred_loss_accum.item()) * B

        pred = logits.argmax(dim=-1)
        correct += int((pred == y_cls).sum().item())
        total += int(y_cls.numel())

    denom = max(1, len(loader.dataset))
    return {
        "cls_loss": total_cls / denom,
        "pred_loss": total_pred / denom,
        "total_loss": (total_cls + float(lambda_pred) * total_pred) / denom,
        "acc": correct / max(1, total),
    }


# -------------------------
# Main
# -------------------------
def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--frontend", type=str, default="feature_c", choices=["feature_c", "gammatone"])

    # reproducibility
    p.add_argument("--seed", type=int, default=42)

    # device
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")

    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)

    # model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_norm", action="store_true")

    # tbptt + losses
    p.add_argument("--chunk_len", type=int, default=1000)
    p.add_argument("--lambda_pred", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # shared trial timing
    p.add_argument("--tone_ms", type=int, default=50)
    p.add_argument("--isi_ms", type=int, default=700)
    p.add_argument("--ramp_ms", type=int, default=5)

    # feature_c params
    p.add_argument("--sigma_tone", type=float, default=0.05)
    p.add_argument("--sigma_silence", type=float, default=0.05)
    p.add_argument("--mu_silence", type=float, default=0.0)
    p.add_argument("--feature_dtype", type=str, default="float32", choices=["float32", "float16"])

    # gammatone params
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--tone_amp", type=float, default=0.1)
    p.add_argument("--noise_sigma", type=float, default=0.02)
    p.add_argument("--n_bands", type=int, default=32)
    p.add_argument("--cf_min", type=float, default=300.0)
    p.add_argument("--cf_max", type=float, default=4000.0)
    p.add_argument("--ir_ms", type=int, default=40)
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--log_compress", action="store_true")
    p.add_argument("--log_eps", type=float, default=1e-4)
    p.add_argument("--smooth_ms", type=int, default=7)

    # resume
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt (best.pt/last.pt)")

    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[device] using: {device}")
    if device.type == "cuda":
        print(f"[cuda] name: {torch.cuda.get_device_name(0)}")
        print(f"[cuda] capability: {torch.cuda.get_device_capability(0)}")

    # dataset
    ds = OnlineRenderDataset(
        data_dir=data_dir,
        frontend=args.frontend,
        seed=args.seed,
        tone_ms=args.tone_ms,
        isi_ms=args.isi_ms,
        ramp_ms=args.ramp_ms,
        sigma_tone=args.sigma_tone,
        sigma_silence=args.sigma_silence,
        mu_silence=args.mu_silence,
        feature_dtype=args.feature_dtype,
        sr=args.sr,
        tone_amp=args.tone_amp,
        noise_sigma=args.noise_sigma,
        n_bands=args.n_bands,
        cf_min=args.cf_min,
        cf_max=args.cf_max,
        ir_ms=args.ir_ms,
        order=args.order,
        log_compress=args.log_compress,
        log_eps=args.log_eps,
        smooth_ms=args.smooth_ms,
    )

    n = len(ds)
    n_val = max(1, int(round(n * args.val_split)))
    n_train = n - n_val
    if n_train <= 0:
        raise ValueError("val_split too large for dataset size.")

    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # model
    cfg = ModelConfig(
        input_dim=int(ds.input_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        layer_norm=bool(args.layer_norm),
    )
    model = PredictiveGRU(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "optim_state" in ckpt:
            optim.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] loaded: {args.resume} start_epoch={start_epoch}")

    # save config
    (save_dir / "config.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "frontend": args.frontend,
                "dataset": {
                    "layout": ds.layout,
                    "n_blocks": n,
                    "tone_ms": ds.tone_ms,
                    "isi_ms": ds.isi_ms,
                    "ramp_ms": ds.ramp_ms,
                    "trial_T_ms": ds.trial_T_ms,
                    "T": ds.T,
                    "input_dim": ds.input_dim,
                },
                "model_cfg": asdict(cfg),
                "train_args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_val = float("inf")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optim,
            device=device,
            chunk_len=int(args.chunk_len),
            lambda_pred=float(args.lambda_pred),
            grad_clip=float(args.grad_clip),
        )
        va = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            chunk_len=int(args.chunk_len),
            lambda_pred=float(args.lambda_pred),
        )

        print(
            f"[epoch {epoch:03d}] "
            f"train: loss={tr['total_loss']:.4f} cls={tr['cls_loss']:.4f} pred={tr['pred_loss']:.4f} acc={tr['acc']:.4f} | "
            f"val: loss={va['total_loss']:.4f} cls={va['cls_loss']:.4f} pred={va['pred_loss']:.4f} acc={va['acc']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "cfg": asdict(cfg),
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")
        if va["total_loss"] < best_val:
            best_val = va["total_loss"]
            torch.save(ckpt, save_dir / "best.pt")

    print("Saved checkpoints to:", save_dir.resolve())
    print(" - best.pt")
    print(" - last.pt")
    print(" - config.json")


if __name__ == "__main__":
    main()
