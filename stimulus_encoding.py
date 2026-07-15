from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


_RF_VECTOR_CACHE: dict[tuple[int, float, str, int], np.ndarray] = {}


def hz_to_erb_rate_np(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * f_hz)


def make_erb_edges(f_min_hz: float, f_max_hz: float, n_bins: int) -> np.ndarray:
    erb_min = float(hz_to_erb_rate_np(np.array([f_min_hz], dtype=np.float32))[0])
    erb_max = float(hz_to_erb_rate_np(np.array([f_max_hz], dtype=np.float32))[0])
    return np.linspace(erb_min, erb_max, int(n_bins) + 1, dtype=np.float32)


def freq_to_bin_erb(f_hz: float, edges_erb: np.ndarray) -> int:
    erb = float(hz_to_erb_rate_np(np.array([f_hz], dtype=np.float32))[0])
    j = int(np.searchsorted(edges_erb, erb, side="right") - 1)
    return max(0, min(j, int(edges_erb.shape[0] - 2)))


def normalize_rf_vector(x: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode)
    if mode == "peak":
        denom = float(np.max(x))
        return x if denom <= 0.0 else (x / denom).astype(np.float32)
    if mode == "sum":
        denom = float(np.sum(x))
        return x if denom <= 0.0 else (x / denom).astype(np.float32)
    if mode == "none":
        return x.astype(np.float32)
    raise ValueError(f"Unknown rf_normalization: {mode}")


def gaussian_rf_vector(
    center_bin: int,
    n_bins: int,
    sigma_rf: float,
    rf_normalization: str = "peak",
) -> np.ndarray:
    sigma_rf = float(sigma_rf)
    if sigma_rf <= 0.0:
        raise ValueError("sigma_rf must be > 0")
    key = (int(n_bins), sigma_rf, str(rf_normalization), int(center_bin))
    cached = _RF_VECTOR_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    idx = np.arange(int(n_bins), dtype=np.float32)
    x = np.exp(-0.5 * ((idx - float(center_bin)) / sigma_rf) ** 2).astype(np.float32)
    out = normalize_rf_vector(x, rf_normalization)
    _RF_VECTOR_CACHE[key] = out.copy()
    return out.copy()


@dataclass
class StimulusEncodingConfig:
    tone_ms: int
    isi_ms: int
    token_ms: int
    f_min_hz: float
    f_max_hz: float
    n_bins: int
    add_eos: bool = False
    add_bos: bool = False
    eos_mode: str = "separate"
    sigma_other_noise: float = 0.0
    p_other_noise: float = 0.0
    sigma_silence_noise: float = 0.0
    encoding_mode: str = "onehot"
    sigma_rf: float = 1.0
    rf_normalization: str = "peak"
    sigma_rf_noise: float = 0.0
    rf_noise_per_token: bool = True
    noise_mode: str = "per_token"
    noise_rho: float = 0.0

    def __post_init__(self) -> None:
        self.tone_ms = int(self.tone_ms)
        self.isi_ms = int(self.isi_ms)
        self.token_ms = int(self.token_ms)
        self.n_bins = int(self.n_bins)
        self.add_eos = bool(self.add_eos)
        self.add_bos = bool(self.add_bos)
        self.eos_mode = str(self.eos_mode)
        self.sigma_other_noise = float(self.sigma_other_noise)
        self.p_other_noise = float(self.p_other_noise)
        self.sigma_silence_noise = float(self.sigma_silence_noise)
        self.encoding_mode = str(self.encoding_mode)
        self.sigma_rf = float(self.sigma_rf)
        self.rf_normalization = str(self.rf_normalization)
        self.sigma_rf_noise = float(self.sigma_rf_noise)
        self.rf_noise_per_token = bool(self.rf_noise_per_token)
        self.noise_mode = str(self.noise_mode)
        self.noise_rho = float(self.noise_rho)

        if self.token_ms <= 0:
            raise ValueError("token_ms must be positive.")
        if self.tone_ms % self.token_ms != 0:
            raise ValueError(f"tone_ms={self.tone_ms} not divisible by token_ms={self.token_ms}")
        if self.isi_ms % self.token_ms != 0:
            raise ValueError(f"isi_ms={self.isi_ms} not divisible by token_ms={self.token_ms}")
        if self.n_bins <= 2:
            raise ValueError("n_bins must be >2")
        if self.eos_mode not in ("separate", "mixed"):
            raise ValueError("eos_mode must be one of: separate, mixed")
        if self.encoding_mode not in ("onehot", "gaussian_rf"):
            raise ValueError("encoding_mode must be one of: onehot, gaussian_rf")
        if self.rf_normalization not in ("peak", "sum", "none"):
            raise ValueError("rf_normalization must be one of: peak, sum, none")
        if self.encoding_mode == "gaussian_rf" and self.sigma_rf <= 0.0:
            raise ValueError("sigma_rf must be > 0 for gaussian_rf encoding.")
        if self.noise_mode not in ("per_token", "smoothed", "fixed", "per_tone"):
            raise ValueError("noise_mode must be one of: per_token, smoothed, fixed, per_tone")
        if not (-1.0 < self.noise_rho < 1.0 or np.isclose(self.noise_rho, 1.0)):
            raise ValueError("noise_rho must be in (-1,1] for AR(1) smoothing.")

    @property
    def tone_T(self) -> int:
        return self.tone_ms // self.token_ms

    @property
    def isi_T(self) -> int:
        return self.isi_ms // self.token_ms

    @property
    def boundary_T(self) -> int:
        eos_T = 1 if bool(self.add_eos) and str(self.eos_mode) == "separate" else 0
        bos_T = 1 if bool(self.add_bos) else 0
        return int(eos_T + bos_T)

    @property
    def trial_T_ms(self) -> int:
        base_ms = 7 * (self.tone_ms + self.isi_ms) + self.tone_ms
        return base_ms + int(self.boundary_T) * self.token_ms

    @property
    def trial_T_tokens(self) -> int:
        if self.trial_T_ms % self.token_ms != 0:
            raise ValueError(f"trial_T_ms={self.trial_T_ms} not divisible by token_ms={self.token_ms}")
        trial_T_tokens = self.trial_T_ms // self.token_ms
        expected = 8 * self.tone_T + 7 * self.isi_T + int(self.boundary_T)
        if expected != trial_T_tokens:
            raise RuntimeError(f"Token length mismatch: expected={expected} got={trial_T_tokens}")
        return trial_T_tokens

    @property
    def input_dim(self) -> int:
        return int(self.n_bins) + (1 if self.add_eos else 0) + (1 if self.add_bos else 0)

    @property
    def eos_index(self) -> Optional[int]:
        return int(self.n_bins) if bool(self.add_eos) else None

    @property
    def bos_index(self) -> Optional[int]:
        if not bool(self.add_bos):
            return None
        return int(self.n_bins) + (1 if self.add_eos else 0)


def build_tone_encoding_vector(
    freq_hz: float,
    edges_erb: np.ndarray,
    cfg: StimulusEncodingConfig,
) -> Tuple[np.ndarray, int]:
    center_bin = freq_to_bin_erb(float(freq_hz), edges_erb)
    if str(cfg.encoding_mode) == "onehot":
        x = np.zeros((int(cfg.n_bins),), dtype=np.float32)
        x[center_bin] = 1.0
        return x, center_bin
    x = gaussian_rf_vector(
        center_bin=center_bin,
        n_bins=int(cfg.n_bins),
        sigma_rf=float(cfg.sigma_rf),
        rf_normalization=str(cfg.rf_normalization),
    )
    return x, center_bin


def apply_gaussian_rf_noise(
    rf_vec: np.ndarray,
    sigma_rf_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if float(sigma_rf_noise) <= 0.0:
        return rf_vec.astype(np.float32, copy=True)
    noise = rng.normal(0.0, float(sigma_rf_noise), size=rf_vec.shape).astype(np.float32)
    return np.clip(rf_vec.astype(np.float32) + noise, 0.0, 1.0).astype(np.float32)


def _apply_gaussian_rf_noise_with_sample(
    rf_vec: np.ndarray,
    noise_sample: np.ndarray,
) -> np.ndarray:
    return np.clip(rf_vec.astype(np.float32) + noise_sample.astype(np.float32), 0.0, 1.0).astype(np.float32)


def _sample_smoothed_noise_sequence(
    n_steps: int,
    n_bins: int,
    sigma: float,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.zeros((int(n_steps), int(n_bins)), dtype=np.float32)
    if int(n_steps) <= 0 or float(sigma) <= 0.0:
        return out
    rho = float(rho)
    if np.isclose(rho, 1.0):
        base = rng.normal(0.0, float(sigma), size=(int(n_bins),)).astype(np.float32)
        out[:] = base
        return out
    scale = np.sqrt(max(0.0, 1.0 - rho * rho)) * float(sigma)
    prev = rng.normal(0.0, float(sigma), size=(int(n_bins),)).astype(np.float32)
    out[0] = prev
    for t in range(1, int(n_steps)):
        eps_t = rng.normal(0.0, 1.0, size=(int(n_bins),)).astype(np.float32)
        prev = (rho * prev + scale * eps_t).astype(np.float32)
        out[t] = prev
    return out


def render_trial_tokens_from_freqs(
    freqs_8: np.ndarray,
    cfg: StimulusEncodingConfig,
    rng: np.random.Generator,
    edges_erb: Optional[np.ndarray] = None,
) -> np.ndarray:
    if edges_erb is None:
        edges_erb = make_erb_edges(float(cfg.f_min_hz), float(cfg.f_max_hz), int(cfg.n_bins))
    X = np.zeros((int(cfg.trial_T_tokens), int(cfg.input_dim)), dtype=np.float32)
    noise_mode = str(cfg.noise_mode)
    if noise_mode == "per_tone":
        noise_mode = "fixed"

    t = 0
    for tone_i in range(8):
        base_vec, center_bin = build_tone_encoding_vector(float(freqs_8[tone_i]), edges_erb, cfg)
        shared_rf_vec = None
        tone_noise_seq = None
        if str(cfg.encoding_mode) == "gaussian_rf" and float(cfg.sigma_rf_noise) > 0.0:
            if noise_mode == "fixed" or (noise_mode == "per_token" and (not bool(cfg.rf_noise_per_token))):
                shared_rf_vec = apply_gaussian_rf_noise(base_vec, float(cfg.sigma_rf_noise), rng)
            elif noise_mode == "smoothed":
                tone_noise_seq = _sample_smoothed_noise_sequence(
                    n_steps=int(cfg.tone_T),
                    n_bins=int(cfg.n_bins),
                    sigma=float(cfg.sigma_rf_noise),
                    rho=float(cfg.noise_rho),
                    rng=rng,
                )

        for tone_tok in range(int(cfg.tone_T)):
            if str(cfg.encoding_mode) == "onehot":
                X[t, : int(cfg.n_bins)] = base_vec
                if float(cfg.sigma_other_noise) > 0.0 and float(cfg.p_other_noise) > 0.0:
                    if float(rng.random()) < float(cfg.p_other_noise):
                        noise = rng.normal(0.0, float(cfg.sigma_other_noise), size=(int(cfg.n_bins),)).astype(np.float32)
                        noise[center_bin] = 0.0
                        X[t, : int(cfg.n_bins)] += noise
            else:
                if shared_rf_vec is not None:
                    X[t, : int(cfg.n_bins)] = shared_rf_vec
                elif tone_noise_seq is not None:
                    X[t, : int(cfg.n_bins)] = _apply_gaussian_rf_noise_with_sample(base_vec, tone_noise_seq[tone_tok])
                elif float(cfg.sigma_rf_noise) > 0.0:
                    X[t, : int(cfg.n_bins)] = apply_gaussian_rf_noise(base_vec, float(cfg.sigma_rf_noise), rng)
                else:
                    X[t, : int(cfg.n_bins)] = base_vec
            t += 1

        if tone_i < 7:
            if float(cfg.sigma_silence_noise) > 0.0:
                for _ in range(int(cfg.isi_T)):
                    X[t, : int(cfg.n_bins)] = rng.normal(
                        0.0, float(cfg.sigma_silence_noise), size=(int(cfg.n_bins),)
                    ).astype(np.float32)
                    t += 1
            else:
                t += int(cfg.isi_T)

    if bool(cfg.add_eos):
        if str(cfg.eos_mode) == "mixed":
            # Legacy behavior: mark the final real tone token with an EOS flag.
            X[t - 1, int(cfg.eos_index)] = 1.0
        else:
            # Preferred behavior: append a dedicated boundary token.
            X[t, int(cfg.eos_index)] = 1.0
            t += 1
    if bool(cfg.add_bos):
        if cfg.bos_index is None:
            raise RuntimeError("BOS index is unexpectedly None")
        X[t, int(cfg.bos_index)] = 1.0
        t += 1

    if t != int(cfg.trial_T_tokens):
        raise RuntimeError(f"Trial token fill mismatch: t={t} trial_T_tokens={cfg.trial_T_tokens}")
    return X


def render_blocks_hz_to_tokens(
    X_hz,
    cfg: StimulusEncodingConfig,
    seed: int = 42,
) :
    import torch

    if X_hz.ndim != 3 or tuple(X_hz.shape[1:]) != (10, 8):
        raise ValueError(f"Expected X_hz (B,10,8), got {tuple(X_hz.shape)}")

    X_hz = X_hz.detach().cpu().float()
    B = int(X_hz.shape[0])
    T_tokens = 10 * int(cfg.trial_T_tokens)
    out = torch.zeros((B, T_tokens, int(cfg.input_dim)), dtype=torch.float32)
    edges_erb = make_erb_edges(float(cfg.f_min_hz), float(cfg.f_max_hz), int(cfg.n_bins))

    for b in range(B):
        t_abs = 0
        for tr in range(10):
            rng = np.random.default_rng(int(seed) + b * 1000 + tr * 17)
            trial_tokens = render_trial_tokens_from_freqs(
                freqs_8=X_hz[b, tr].numpy().astype(np.float32),
                cfg=cfg,
                rng=rng,
                edges_erb=edges_erb,
            )
            trial_len = int(cfg.trial_T_tokens)
            out[b, t_abs:t_abs + trial_len] = torch.from_numpy(trial_tokens)
            t_abs += trial_len
        if t_abs != T_tokens:
            raise RuntimeError(f"Token fill mismatch: ended at {t_abs}, expected {T_tokens}")
    return out
