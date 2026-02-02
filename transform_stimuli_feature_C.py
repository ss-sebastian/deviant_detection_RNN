# transform_stimuli_feature_C.py
import argparse
import json
from pathlib import Path
from typing import Tuple

import torch


def hz_to_erb_rate(f_hz: torch.Tensor) -> torch.Tensor:
    # ERB-rate: 21.4 * log10(1 + 0.00437 f)
    return 21.4 * torch.log10(1.0 + 0.00437 * f_hz)


def make_linear_envelope_ms(tone_ms: int, ramp_ms: int) -> torch.Tensor:
    """
    Envelope length = tone_ms.
    Linear ramp-in for ramp_ms, linear ramp-out for ramp_ms.
    """
    if ramp_ms <= 0:
        return torch.ones((tone_ms,), dtype=torch.float32)

    ramp_ms = int(ramp_ms)
    if 2 * ramp_ms > tone_ms:
        ramp_ms = tone_ms // 2

    env = torch.ones((tone_ms,), dtype=torch.float32)
    if ramp_ms > 0:
        up = torch.linspace(0.0, 1.0, steps=ramp_ms, dtype=torch.float32)
        down = torch.linspace(1.0, 0.0, steps=ramp_ms, dtype=torch.float32)
        env[:ramp_ms] = up
        env[-ramp_ms:] = down
    return env


def render_trial_feature_C(
    freqs_8_hz: torch.Tensor,
    tone_ms: int,
    isi_ms: int,
    ramp_ms: int,
    sigma_tone: float,
    sigma_silence: float,
    mu_silence: float,
    g: torch.Generator,
) -> torch.Tensor:
    """
    Returns (trial_T, 1) float32
      - tone segments: x = env * ERB(f) + N(0, sigma_tone)
      - silence segments: x = mu_silence + N(0, sigma_silence)
    """
    assert tuple(freqs_8_hz.shape) == (8,)
    erb = hz_to_erb_rate(freqs_8_hz.float())  # (8,)

    trial_T = 7 * (tone_ms + isi_ms) + tone_ms  # should be 5300 with defaults
    out = mu_silence + torch.randn((trial_T, 1), generator=g, dtype=torch.float32) * float(sigma_silence)

    env = make_linear_envelope_ms(tone_ms=tone_ms, ramp_ms=ramp_ms).view(tone_ms, 1)  # (tone_ms,1)

    t = 0
    for i in range(8):
        tone_slice = slice(t, t + tone_ms)
        base = env * float(erb[i].item())
        noise = torch.randn((tone_ms, 1), generator=g, dtype=torch.float32) * float(sigma_tone)
        out[tone_slice] = base + noise
        t += tone_ms
        if i < 7:
            t += isi_ms  # silence remains background noise

    return out


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

    raise FileNotFoundError("Need input_blocks.pt/labels_blocks.pt OR input_tensor.pt/labels_tensor.pt in --in_dir")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True)
    p.add_argument("--tone_ms", type=int, default=50)
    p.add_argument("--isi_ms", type=int, default=700)
    p.add_argument("--ramp_ms", type=int, default=5)

    p.add_argument("--sigma_tone", type=float, default=0.05)
    p.add_argument("--sigma_silence", type=float, default=0.05)
    p.add_argument("--mu_silence", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--save_name", type=str, default="ms_input_tensor.pt")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    X, Y, layout = load_blocks_or_single(in_dir)
    B = X.shape[0]

    trial_T = 7 * (args.tone_ms + args.isi_ms) + args.tone_ms
    if trial_T <= 0:
        raise ValueError("Invalid tone_ms/isi_ms")

    out = torch.empty((B, 10, trial_T, 1), dtype=torch.float32)
    g = torch.Generator().manual_seed(int(args.seed))

    for b in range(B):
        for t in range(10):
            out[b, t] = render_trial_feature_C(
                freqs_8_hz=X[b, t],
                tone_ms=int(args.tone_ms),
                isi_ms=int(args.isi_ms),
                ramp_ms=int(args.ramp_ms),
                sigma_tone=float(args.sigma_tone),
                sigma_silence=float(args.sigma_silence),
                mu_silence=float(args.mu_silence),
                g=g,
            )

    torch.save(out, in_dir / args.save_name)
    torch.save(Y, in_dir / "ms_labels_tensor.pt")

    end_idx = torch.tensor([trial_T - 1] * 10, dtype=torch.long)
    torch.save(end_idx, in_dir / "trial_end_indices.pt")

    meta = {
        "mode": "feature_C_noise_ramps_1ch",
        "layout": layout,
        "n_blocks": int(B),
        "tone_ms": int(args.tone_ms),
        "isi_ms": int(args.isi_ms),
        "ramp_ms": int(args.ramp_ms),
        "ms_per_trial": int(trial_T),
        "trials_per_block": 10,
        "channels": ["x"],
        "x_definition": "tone: env*ERB(f)+N(0,sigma_tone); silence: mu_silence+N(0,sigma_silence)",
        "sigma_tone": float(args.sigma_tone),
        "sigma_silence": float(args.sigma_silence),
        "mu_silence": float(args.mu_silence),
        "labels": "deviant position per trial (1-indexed in {4,5,6})",
        "output_shape": list(out.shape),
    }
    (in_dir / "ms_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:")
    print(" -", (in_dir / args.save_name).resolve(), "shape=", tuple(out.shape))
    print(" -", (in_dir / "ms_labels_tensor.pt").resolve(), "shape=", tuple(Y.shape))
    print(" -", (in_dir / "trial_end_indices.pt").resolve(), "shape=", tuple(end_idx.shape))
    print(" -", (in_dir / "ms_meta.json").resolve())


if __name__ == "__main__":
    main()
