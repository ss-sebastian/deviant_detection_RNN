# train_online.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Optional, Literal
# import torch_xla.core.xla_model as xm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import torch_xla.distributed.parallel_loader as pl
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
def make_rt_targets_and_mask(
    abs_t: torch.Tensor,         # (L,) absolute time indices in [0, T)
    y_pos_456: torch.Tensor,     # (B,10) in {4,5,6}
    trial_T: int,
    tone_ms: int,
    isi_ms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-ms targets for rt_head and a mask to ignore silence.

    Returns:
      target: (B,L) long in {0,1}    1 means "deviant tone" at this ms
      mask:   (B,L) bool            True only for tone segments
    """
    # which trial does each ms belong to?
    trial_id = (abs_t // trial_T).long()     # (L,) values 0..9
    within = (abs_t % trial_T).long()        # (L,) within-trial time

    step = tone_ms + isi_ms

    tone_id = (within // step).clamp_max(7)  # (L,) 0..7 (8 tones)
    phase = (within % step)                  # (L,)
    is_tone = (phase < tone_ms)              # (L,) tone vs silence

    # deviant tone index in 0-based: y in {4,5,6} -> {3,4,5}
    dev_idx = (y_pos_456 - 1).long()         # (B,10)

    # expand to (B,L)
    dev_for_t = dev_idx[:, trial_id]         # (B,L)
    tone_for_t = tone_id.unsqueeze(0).expand_as(dev_for_t)

    target = (tone_for_t == dev_for_t).long()    # (B,L)
    mask = is_tone.unsqueeze(0).expand_as(target) # (B,L)
    return target, mask

def _run_block_through_tbptt(
    model: PredictiveGRU,
    x: torch.Tensor,                 # (B,T,D)
    y_pos_456: torch.Tensor,         # (B,10) values in {4,5,6}  (for RT supervision)
    end_idx: torch.Tensor,           # (10,) on device
    chunk_len: int,
    compute_pred_loss: bool,
    huber: nn.Module,
    # --- RT head supervision ---
    rt_ce: nn.Module,                # CrossEntropyLoss
    tone_ms: int,
    isi_ms: int,
    trial_T_ms: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      h_end: (B,10,H) trial-end hidden states
      pred_loss_accum: scalar tensor (0 if compute_pred_loss=False)
      rt_loss_accum: scalar tensor (tone-only CE accumulated across chunks)
    """
    B, T, D = x.shape
    h = None
    collected = []
    pred_loss_accum = x.new_tensor(0.0)
    rt_loss_accum = x.new_tensor(0.0)

    for s in range(0, T, chunk_len):
        e = min(s + chunk_len, T)   # exclusive
        x_in = x[:, s:e, :]
        h_seq, h, x_hat = model.forward_chunk(x_in, h0=h)  # h_seq: (B,L,H)
        L = e - s

        # ---- next-step prediction loss (optional)
        if compute_pred_loss and L >= 2:
            pred_in = x_hat[:, :-1, :].reshape(-1, D)
            pred_tg = x[:, s+1:e, :].reshape(-1, D)
            pred_loss_accum = pred_loss_accum + huber(pred_in, pred_tg)

        # ---- RT head loss (tone-only)
        # logits: (B,L,2)
        rt_logits = model.classify_rt_from_seq(h_seq)

        abs_t = torch.arange(s, e, device=x.device)  # (L,)
        target, mask = make_rt_targets_and_mask(
            abs_t=abs_t,
            y_pos_456=y_pos_456,
            trial_T=trial_T_ms,
            tone_ms=int(tone_ms),
            isi_ms=int(isi_ms),
        )
        target = target.to(device=x.device)
        mask = mask.to(device=x.device)
        rt_loss_sum = x.new_tensor(0.0)
        rt_count = x.new_tensor(0.0)
        if mask.any():
            # rt_ce expects (N,2) and (N,)
            per = rt_ce(rt_logits[mask], target[mask])
            rt_loss_sum = rt_loss_sum + per.sum()
            rt_count = rt_count + per.numel()
        
        rt_loss_accum = rt_loss_sum / (rt_count + 1e-8)
        
        # ---- collect trial-end states (unchanged)
        m_end = (end_idx >= s) & (end_idx < e)
        if m_end.any():
            rel = (end_idx[m_end] - s).long()
            hs = h_seq.index_select(dim=1, index=rel)
            collected.append(hs)

        # ---- TBPTT truncate
        h = h.detach()

    if len(collected) == 0:
        raise RuntimeError("No trial-end states collected. Check end_idx and T.")

    h_end = torch.cat(collected, dim=1)  # (B,10,H)
    return h_end, pred_loss_accum, rt_loss_accum

def train_one_epoch(
    model: PredictiveGRU,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    chunk_len: int,
    lambda_pred: float,
    lambda_rt: float,
    grad_clip: float,
    tone_ms: int,
    isi_ms: int,
    trial_T_ms: int,
    debug: bool = False,
    debug_steps: int = 0,
) -> dict:
    model.train()
    huber = nn.SmoothL1Loss(reduction="mean")
    ce = nn.CrossEntropyLoss(reduction="mean")
    rt_ce = nn.CrossEntropyLoss(reduction="mean")

    total_cls = 0.0
    total_pred = 0.0
    total_rt = 0.0
    total = 0
    correct = 0
    n_examples = 0

    compute_pred = lambda_pred > 0
    compute_rt = lambda_rt > 0

    step = 0
    for x, y in loader:
        step += 1

        x = x.to(device, non_blocking=True)  # (B,T,D)
        y = y.to(device, non_blocking=True)  # (B,10) in {4,5,6}
        B, T, D = x.shape
        n_examples += B

        end_idx = infer_end_indices_from_T(T, trials_per_block=10).to(device)

        optimizer.zero_grad(set_to_none=True)

        # --- debug: snapshot a parameter before update
        w0 = None
        if debug and step <= int(debug_steps):
            w = next(model.parameters())
            w0 = w.detach().float().cpu().clone()

        # --- TBPTT run (now also returns rt_loss_accum)
        h_end, pred_loss_accum, rt_loss_accum = _run_block_through_tbptt(
            model=model,
            x=x,
            y_pos_456=y,
            end_idx=end_idx,
            chunk_len=chunk_len,
            compute_pred_loss=compute_pred,
            huber=huber,
            rt_ce=rt_ce,
            tone_ms=int(tone_ms),
            isi_ms=int(isi_ms),
            trial_T_ms=int(trial_T_ms),
        )

        logits = model.classify_from_states(h_end)  # (B,10,3)
        y_cls = labels_to_class_index(y)            # (B,10) -> {0,1,2}

        cls_loss = ce(logits.reshape(-1, 3), y_cls.reshape(-1))

        total_loss = cls_loss
        if compute_pred:
            total_loss = total_loss + float(lambda_pred) * pred_loss_accum
        if compute_rt:
            total_loss = total_loss + float(lambda_rt) * rt_loss_accum

        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # --- step
        if device.type == "xla":
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()

        total_cls += float(cls_loss.item()) * B
        total_pred += float(pred_loss_accum.item()) * B
        total_rt += float(rt_loss_accum.item()) * B

        pred = logits.argmax(dim=-1)
        correct += int((pred == y_cls).sum().item())
        total += int(y_cls.numel())

        # --- debug: check parameter actually changed
        if debug and step <= int(debug_steps):
            w1 = next(model.parameters()).detach().float().cpu()
            delta = (w1 - w0).abs().mean().item() if w0 is not None else float("nan")
            batch_acc = (pred == y_cls).float().mean().item()
            print(
                f"[debug step {step}] mean|Δparam|={delta:.6e} "
                f"cls={cls_loss.item():.4f} pred={float(pred_loss_accum.item()):.4f} rt={float(rt_loss_accum.item()):.4f} "
                f"acc={batch_acc:.3f} logits_mean={logits.detach().float().mean().item():.4f}"
            )

    denom = max(1, n_examples)
    # keep your old total_loss definition style, but now includes rt
    avg_total = total_cls
    if compute_pred:
        avg_total += float(lambda_pred) * total_pred
    if compute_rt:
        avg_total += float(lambda_rt) * total_rt

    return {
        "cls_loss": total_cls / denom,
        "pred_loss": total_pred / denom,
        "rt_loss": total_rt / denom,
        "total_loss": avg_total / denom,
        "acc": correct / max(1, total),
    }


@torch.no_grad()
def evaluate(
    model: PredictiveGRU,
    loader,
    device: torch.device,
    chunk_len: int,
    lambda_pred: float,
    lambda_rt: float,
    tone_ms: int,
    isi_ms: int,
    trial_T_ms: int,
) -> dict:
    model.eval()
    huber = nn.SmoothL1Loss(reduction="mean")
    ce = nn.CrossEntropyLoss(reduction="mean")
    rt_ce = nn.CrossEntropyLoss(reduction="mean")

    total_cls = 0.0
    total_pred = 0.0
    total_rt = 0.0
    total = 0
    correct = 0
    n_examples = 0

    compute_pred = lambda_pred > 0
    compute_rt = lambda_rt > 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B, T, D = x.shape
        n_examples += B

        end_idx = infer_end_indices_from_T(T, trials_per_block=10).to(device)

        h_end, pred_loss_accum, rt_loss_accum = _run_block_through_tbptt(
            model=model,
            x=x,
            y_pos_456=y,
            end_idx=end_idx,
            chunk_len=chunk_len,
            compute_pred_loss=compute_pred,
            huber=huber,
            rt_ce=rt_ce,
            tone_ms=int(tone_ms),
            isi_ms=int(isi_ms),
            trial_T_ms=int(trial_T_ms),
        )

        logits = model.classify_from_states(h_end)
        y_cls = labels_to_class_index(y)

        cls_loss = ce(logits.reshape(-1, 3), y_cls.reshape(-1))

        total_cls += float(cls_loss.item()) * B
        total_pred += float(pred_loss_accum.item()) * B
        total_rt += float(rt_loss_accum.item()) * B

        pred = logits.argmax(dim=-1)
        correct += int((pred == y_cls).sum().item())
        total += int(y_cls.numel())

    denom = max(1, n_examples)
    avg_total = total_cls
    if compute_pred:
        avg_total += float(lambda_pred) * total_pred
    if compute_rt:
        avg_total += float(lambda_rt) * total_rt

    return {
        "cls_loss": total_cls / denom,
        "pred_loss": total_pred / denom,
        "rt_loss": total_rt / denom,
        "total_loss": avg_total / denom,
        "acc": correct / max(1, total),
    }


# -------------------------
# Main
# -------------------------
def resolve_device(device_str: str) -> torch.device:
    s = device_str.lower()
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        # TPU 在 auto 不会自动选，避免误用；想用 TPU 请显式 --device xla
        return torch.device("cpu")

    if s in ["xla", "tpu"]:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()

    return torch.device(s)


def maybe_wrap_xla_loader(loader, device: torch.device):
    """
    Wrap a PyTorch DataLoader with MpDeviceLoader when using XLA/TPU.
    This improves host->device transfer and is the recommended pattern in torch_xla.
    """
    if device.type == "xla":
        import torch_xla.distributed.parallel_loader as pl
        return pl.MpDeviceLoader(loader, device)
    return loader



def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--frontend", type=str, default="feature_c", choices=["feature_c", "gammatone"])

    # reproducibility
    p.add_argument("--seed", type=int, default=42)

    # device
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu | xla(tpu)")

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
    p.add_argument("--log_every", type=int, default=10)
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

    # debug / sanity
    p.add_argument("--debug", action="store_true", help="Enable debug prints/checks (slower).")
    p.add_argument("--debug_steps", type=int, default=3, help="How many first steps to run debug prints.")
    p.add_argument("--debug_xy", action="store_true", help="Check X-Y alignment (deviant pos) on a sample.")
    p.add_argument("--debug_xy_n", type=int, default=200, help="How many (block,trial) to sample for X-Y check.")
    p.add_argument("--max_blocks", type=int, default=0, help="If >0, restrict dataset to first N blocks (overfit sanity).")

    args = p.parse_args()

    # Seeds
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

    ordinal = None  # for xla master fallback
    world_size = 1
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        if hasattr(xm, "get_ordinal"):
            ordinal = xm.get_ordinal()
        elif hasattr(xm, "ordinal"):
            ordinal = xm.ordinal()
        elif hasattr(xm, "get_local_ordinal"):
            ordinal = xm.get_local_ordinal()
        else:
            ordinal = -1

        if hasattr(xm, "xrt_world_size"):
            world_size = xm.xrt_world_size()
        elif hasattr(xm, "world_size"):
            world_size = xm.world_size()
        else:
            world_size = 1

        print(f"[xla] ordinal={ordinal} world_size={world_size}")

    # dataset (compact blocks -> on-the-fly ms rendering)
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

    # -------------------------
    # Debug / sanity: dataset checks
    # -------------------------
    if args.max_blocks and args.max_blocks > 0:
        m = min(int(args.max_blocks), int(ds.B))
        ds.X = ds.X[:m]
        ds.Y = ds.Y[:m]
        ds.B = m
        print(f"[debug] max_blocks applied: B={ds.B}")

    vals, counts = torch.unique(ds.Y, return_counts=True)
    print("[data] unique Y values:", list(zip(vals.tolist(), counts.tolist())))
    if not set(vals.tolist()).issubset({4, 5, 6}):
        print("[WARN] Y contains values outside {4,5,6}. labels_to_class_index(y-4) will be wrong.")

    try:
        end_preview = infer_end_indices_from_T(int(ds.T), trials_per_block=10)
        print(f"[data] T={ds.T} trial_T={ds.trial_T_ms} end_idx preview: "
              f"{end_preview[:3].tolist()} ... {end_preview[-3:].tolist()}")
    except Exception as e:
        print("[WARN] cannot infer end indices from ds.T:", repr(e))

    def _deviant_pos_from_freqs(freqs8: torch.Tensor):
        uniq, cnt = torch.unique(freqs8.cpu(), return_counts=True)
        if uniq.numel() != 2:
            return None
        dev_freq = uniq[cnt.argmin()]
        idx = (freqs8.cpu() == dev_freq).nonzero(as_tuple=False).view(-1)
        if idx.numel() != 1:
            return None
        return int(idx.item()) + 1  # 1-based

    if args.debug_xy:
        ncheck = min(int(args.debug_xy_n), int(ds.B) * 10)
        ok = 0
        mismatch = 0
        bad_pattern = 0
        checked = 0

        n_blocks_to_scan = min(ds.B, int(math.ceil(ncheck / 10)))
        for b in range(n_blocks_to_scan):
            for t in range(10):
                if checked >= ncheck:
                    break
                pos = _deviant_pos_from_freqs(ds.X[b, t])
                y = int(ds.Y[b, t].item())
                checked += 1
                if pos is None:
                    bad_pattern += 1
                else:
                    if pos == y:
                        ok += 1
                    else:
                        mismatch += 1
            if checked >= ncheck:
                break

        print(f"[debug_xy] checked={checked} ok={ok} mismatch={mismatch} bad_pattern={bad_pattern}")
        if mismatch > 0:
            print("[WARN] X-Y mismatch: label may not match deviant position encoded in X.")
        if bad_pattern > 0:
            print("[WARN] Some trials are not 7+1 pattern (unique!=2 or dev not unique).")

    # -------------------------
    # Split
    # -------------------------
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

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # Wrap loaders for XLA
    train_loader = maybe_wrap_xla_loader(train_loader, device)
    val_loader = maybe_wrap_xla_loader(val_loader, device)

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
                    "layout": getattr(ds, "layout", "unknown"),
                    "n_blocks": n,
                    "tone_ms": ds.tone_ms,
                    "isi_ms": ds.isi_ms,
                    "ramp_ms": ds.ramp_ms,
                    "trial_T_ms": ds.trial_T_ms,
                    "T": ds.T,
                    "input_dim": ds.input_dim,
                },
                "device": str(device),
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
            debug=bool(args.debug),
            debug_steps=int(args.debug_steps),
            lambda_rt=0.1,
            tone_ms=int(args.tone_ms),
            isi_ms=int(args.isi_ms),
            trial_T_ms=int(ds.trial_T_ms),
        )
        va = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            chunk_len=int(args.chunk_len),
            lambda_pred=float(args.lambda_pred),
            lambda_rt=0.1,
            tone_ms=int(args.tone_ms),
            isi_ms=int(args.isi_ms),
            trial_T_ms=int(ds.trial_T_ms),
        )

        print(
            f"[epoch {epoch:03d}] "
            f"train: loss={tr['total_loss']:.4f} cls={tr['cls_loss']:.4f} pred={tr['pred_loss']:.4f} rt={tr['rt_loss']:.4f} acc={tr['acc']:.4f} | "
            f"val: loss={va['total_loss']:.4f} cls={va['cls_loss']:.4f} pred={va['pred_loss']:.4f} rt={va['rt_loss']:.4f} acc={va['acc']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "cfg": asdict(cfg),
            "args": vars(args),
        }

        # Saving on XLA: only master should write to disk (compat)
        if device.type == "xla":
            import torch_xla.core.xla_model as xm
            is_master = True
            if hasattr(xm, "is_master_ordinal"):
                is_master = xm.is_master_ordinal()
            elif ordinal is not None:
                is_master = (int(ordinal) == 0)
            if is_master:
                torch.save(ckpt, save_dir / "last.pt")
                if va["total_loss"] < best_val:
                    best_val = va["total_loss"]
                    torch.save(ckpt, save_dir / "best.pt")
        else:
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
