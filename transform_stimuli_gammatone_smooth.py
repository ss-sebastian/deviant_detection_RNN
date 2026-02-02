# transform_stimuli_gammatone_smooth.py
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

try:
    from scipy.signal import lfilter, hilbert
except Exception as e:
    raise ImportError(
        "This script requires scipy.\n"
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
    Not a perfect cochlear model, but good enough as a perceptual frontend proxy.
    """
    t = np.arange(n_samples, dtype=np.float32) / float(sr)

    # ERB bandwidth (Hz)
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
    ir_n = max(int(round(ir_ms * sr / 1000.0)), 8)
    outs = []
    for cf in cfs:
        h = gammatone_impulse_response(sr=sr, cf_hz=float(cf), n_samples=ir_n, order=order)
        y = lfilter(h, [1.0], x).astype(np.float32)
        env = np.abs(hilbert(y)).astype(np.float32)
        outs.append(env)
    return np.stack(outs, axis=-1)  # (T_samples, n_bands)


def downsample_to_1ms_mean(Y: np.ndarray, sr: int, target_ms: int) -> np.ndarray:
    """
    Average pooling to 1ms bins, then crop/pad to target_ms.
    """
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
    """
    X: (T_ms, n_bands)
    moving average over time with window win_ms, centered-ish using 'same' via convolution.
    """
    win_ms = int(win_ms)
    if win_ms <= 1:
        return X
    kernel = np.ones((win_ms,), dtype=np.float32) / float(win_ms)
    # apply per band
    Y = np.empty_like(X)
    for k in range(X.shape[1]):
        Y[:, k] = np.convolve(X[:, k], kernel, mode="same").astype(np.float32)
    return Y


def load_blocks_or_single(in_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, str]:
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

    raise FileNotFoundError("Need input_blocks.pt/labels_blocks.pt OR input_tensor.pt/labels_tensor.pt in --in_dir")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True)

    # trial timing
    p.add_argument("--tone_ms", type=int, default=50)
    p.add_argument("--isi_ms", type=int, default=700)
    p.add_argument("--ramp_ms", type=int, default=5)

    # waveform
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--tone_amp", type=float, default=0.1)
    p.add_argument("--noise_sigma", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)

    # gammatone
    p.add_argument("--n_bands", type=int, default=32)
    p.add_argument("--cf_min", type=float, default=300.0)
    p.add_argument("--cf_max", type=float, default=4000.0)
    p.add_argument("--ir_ms", type=int, default=40)
    p.add_argument("--order", type=int, default=4)

    # postproc
    p.add_argument("--log_compress", action="store_true")
    p.add_argument("--log_eps", type=float, default=1e-4)
    p.add_argument("--smooth_ms", type=int, default=7, help="moving average window in ms (e.g., 5-10)")

    p.add_argument("--save_name", type=str, default="gt_input_tensor.pt")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    X, Y, layout = load_blocks_or_single(in_dir)
    B = X.shape[0]

    trial_T_ms = 7 * (args.tone_ms + args.isi_ms) + args.tone_ms  # 5300
    rng = np.random.default_rng(int(args.seed))

    # Center frequencies evenly spaced on ERB-rate scale
    erb_min = hz_to_erb_rate_np(np.array([args.cf_min], dtype=np.float32))[0]
    erb_max = hz_to_erb_rate_np(np.array([args.cf_max], dtype=np.float32))[0]
    erb_grid = np.linspace(erb_min, erb_max, int(args.n_bands), dtype=np.float32)
    cfs = erb_rate_to_hz_np(erb_grid).astype(np.float32)

    out = np.zeros((B, 10, trial_T_ms, int(args.n_bands)), dtype=np.float32)

    for b in range(B):
        for t in range(10):
            freqs = X[b, t].numpy().astype(np.float32)  # (8,)
            wav = synth_trial_waveform(
                freqs_8_hz=freqs,
                sr=int(args.sr),
                tone_ms=int(args.tone_ms),
                isi_ms=int(args.isi_ms),
                ramp_ms=int(args.ramp_ms),
                tone_amp=float(args.tone_amp),
                noise_sigma=float(args.noise_sigma),
                rng=rng,
            )

            env_fb = gammatone_filterbank_envelope(
                x=wav,
                sr=int(args.sr),
                cfs=cfs,
                ir_ms=int(args.ir_ms),
                order=int(args.order),
            )  # (T_samples, n_bands)

            env_ms = downsample_to_1ms_mean(env_fb, sr=int(args.sr), target_ms=int(trial_T_ms))  # (5300, n_bands)

            if args.log_compress:
                env_ms = np.log(env_ms + float(args.log_eps)).astype(np.float32)

            # time smoothing (neural integration proxy)
            env_ms = moving_average_ms(env_ms, win_ms=int(args.smooth_ms))

            out[b, t] = env_ms

    out_t = torch.from_numpy(out)
    torch.save(out_t, in_dir / args.save_name)
    torch.save(Y, in_dir / "gt_labels_tensor.pt")

    end_idx = torch.tensor([trial_T_ms - 1] * 10, dtype=torch.long)
    torch.save(end_idx, in_dir / "trial_end_indices.pt")

    meta = {
        "mode": "gammatone_envelope_smoothed",
        "layout": layout,
        "n_blocks": int(B),
        "tone_ms": int(args.tone_ms),
        "isi_ms": int(args.isi_ms),
        "ramp_ms": int(args.ramp_ms),
        "ms_per_trial": int(trial_T_ms),
        "trials_per_block": 10,
        "sr": int(args.sr),
        "tone_amp": float(args.tone_amp),
        "noise_sigma": float(args.noise_sigma),
        "n_bands": int(args.n_bands),
        "cf_min": float(args.cf_min),
        "cf_max": float(args.cf_max),
        "cfs_hz": [float(v) for v in cfs.tolist()],
        "ir_ms": int(args.ir_ms),
        "order": int(args.order),
        "log_compress": bool(args.log_compress),
        "log_eps": float(args.log_eps),
        "smooth_ms": int(args.smooth_ms),
        "output_shape": list(out_t.shape),
        "labels": "deviant position per trial (1-indexed in {4,5,6})",
        "trial_end_indices_file": "trial_end_indices.pt",
    }
    (in_dir / "gt_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:")
    print(" -", (in_dir / args.save_name).resolve(), "shape=", tuple(out_t.shape))
    print(" -", (in_dir / "gt_labels_tensor.pt").resolve(), "shape=", tuple(Y.shape))
    print(" -", (in_dir / "gt_meta.json").resolve())


if __name__ == "__main__":
    main()
