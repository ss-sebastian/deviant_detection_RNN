# transform_stimuli_ms.py
import argparse
import json
from pathlib import Path

import torch


def hz_to_erb_rate(f_hz: torch.Tensor) -> torch.Tensor:
    # ERB-rate (a common mapping):
    # ERB(f) = 21.4 * log10(1 + 0.00437 f)
    return 21.4 * torch.log10(1.0 + 0.00437 * f_hz)


def render_trial_ms(freqs_8_hz: torch.Tensor, tone_ms: int, isi_ms: int) -> torch.Tensor:
    """
    freqs_8_hz: (8,) Hz
    returns: (trial_T, 2) -> [erb_value, is_tone]
    trial_T = 7*(tone_ms+isi_ms) + tone_ms
    """
    assert freqs_8_hz.shape == (8,)
    erb = hz_to_erb_rate(freqs_8_hz.float())  # (8,)

    trial_T = 7 * (tone_ms + isi_ms) + tone_ms
    out = torch.zeros((trial_T, 2), dtype=torch.float32)

    t = 0
    for i in range(8):
        out[t:t + tone_ms, 0] = erb[i]
        out[t:t + tone_ms, 1] = 1.0
        t += tone_ms
        if i < 7:
            t += isi_ms  # silence, already zeros

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True)
    p.add_argument("--tone_ms", type=int, default=50)
    p.add_argument("--isi_ms", type=int, default=700)
    p.add_argument("--add_trial_start", action="store_true", help="Add trial_start channel (recommended)")
    p.add_argument("--save_name", type=str, default="ms_input_tensor.pt")
    args = p.parse_args()

    in_dir = Path(args.in_dir)

    x = torch.load(in_dir / "input_tensor.pt", map_location="cpu")   # (1,10,8) Hz
    y = torch.load(in_dir / "labels_tensor.pt", map_location="cpu")  # (1,10) in {4,5,6}

    if x.ndim != 3 or tuple(x.shape[1:]) != (10, 8):
        raise ValueError(f"Expected input_tensor shape (1,10,8). Got {tuple(x.shape)}")
    if y.ndim != 2 or tuple(y.shape) != (1, 10):
        raise ValueError(f"Expected labels_tensor shape (1,10). Got {tuple(y.shape)}")

    trials = x[0].float()  # (10,8)
    rendered = [render_trial_ms(trials[i], args.tone_ms, args.isi_ms) for i in range(10)]

    trial_T = rendered[0].shape[0]
    expected_T = 7 * (args.tone_ms + args.isi_ms) + args.tone_ms
    if trial_T != expected_T:
        raise RuntimeError(f"Trial length mismatch: got {trial_T}, expected {expected_T}")

    # ✅ trialwise tensor: (10, trial_T, 2)
    trial_ms = torch.stack(rendered, dim=0)

    if args.add_trial_start:
        # output: (10, trial_T, 3) -> [erb_value, is_tone, trial_start]
        out = torch.zeros((trial_ms.shape[0], trial_ms.shape[1], 3), dtype=torch.float32)
        out[:, :, :2] = trial_ms
        out[:, 0, 2] = 1.0  # start of each trial
        features = ["erb_value", "is_tone", "trial_start"]
    else:
        out = trial_ms  # (10, trial_T, 2)
        features = ["erb_value", "is_tone"]

    # add batch dim so it is consistent with your previous style
    # final: (1, 10, trial_T, D)
    out = out.unsqueeze(0)

    # Save
    torch.save(out, in_dir / args.save_name)
    torch.save(y, in_dir / "ms_labels_tensor.pt")

    # trial end index within each trial (0-indexed)
    # Here every trial has same length, so end index is constant.
    # Save as shape (10,) for convenience
    end_idx = torch.tensor([trial_T - 1] * 10, dtype=torch.long)
    torch.save(end_idx, in_dir / "trial_end_indices.pt")

    meta = {
        "mode": "trialwise",
        "tone_ms": args.tone_ms,
        "isi_ms": args.isi_ms,
        "ms_per_trial": int(trial_T),
        "trials_per_block": 10,
        "output_shape": list(out.shape),  # [1,10,trial_T,D]
        "features": features,
        "labels": "deviant position per trial (1-indexed in {4,5,6})",
        "trial_end_indices_file": "trial_end_indices.pt",
        "note": "Each trial is one input sample; concatenate in training loop if you want stateful trial-to-trial learning.",
    }
    (in_dir / "ms_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:")
    print(" -", (in_dir / args.save_name).resolve(), "shape=", tuple(out.shape))
    print(" -", (in_dir / "ms_labels_tensor.pt").resolve(), "shape=", tuple(y.shape))
    print(" -", (in_dir / "trial_end_indices.pt").resolve(), "shape=", tuple(end_idx.shape))
    print(" -", (in_dir / "ms_meta.json").resolve())


if __name__ == "__main__":
    main()
