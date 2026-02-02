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


def _load_block_tensors(in_dir: Path):
    """
    Supports two layouts:
      A) single-block:
         - input_tensor.pt:  (1,10,8)
         - labels_tensor.pt: (1,10)
      B) multi-block:
         - input_blocks.pt:  (B,10,8)
         - labels_blocks.pt: (B,10)
    Returns:
      X: (B,10,8)
      Y: (B,10)
      mode: "single" or "blocks"
    """
    x_single = in_dir / "input_tensor.pt"
    y_single = in_dir / "labels_tensor.pt"
    x_blocks = in_dir / "input_blocks.pt"
    y_blocks = in_dir / "labels_blocks.pt"

    if x_blocks.exists() and y_blocks.exists():
        X = torch.load(x_blocks, map_location="cpu")
        Y = torch.load(y_blocks, map_location="cpu")
        if X.ndim != 3 or tuple(X.shape[1:]) != (10, 8):
            raise ValueError(f"Expected input_blocks shape (B,10,8). Got {tuple(X.shape)}")
        if Y.ndim != 2 or tuple(Y.shape[1:]) != (10,):
            raise ValueError(f"Expected labels_blocks shape (B,10). Got {tuple(Y.shape)}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Block count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}")
        return X.float(), Y.long(), "blocks"

    if x_single.exists() and y_single.exists():
        X = torch.load(x_single, map_location="cpu")
        Y = torch.load(y_single, map_location="cpu")
        if X.ndim != 3 or tuple(X.shape) != (1, 10, 8):
            raise ValueError(f"Expected input_tensor shape (1,10,8). Got {tuple(X.shape)}")
        if Y.ndim != 2 or tuple(Y.shape) != (1, 10):
            raise ValueError(f"Expected labels_tensor shape (1,10). Got {tuple(Y.shape)}")
        return X[0].unsqueeze(0).float(), Y[0].unsqueeze(0).long(), "single"

    raise FileNotFoundError(
        "Could not find either (input_blocks.pt & labels_blocks.pt) or "
        "(input_tensor.pt & labels_tensor.pt) in the provided --in_dir."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True)
    p.add_argument("--tone_ms", type=int, default=50)
    p.add_argument("--isi_ms", type=int, default=700)
    p.add_argument("--add_trial_start", action="store_true", help="Add trial_start channel")
    p.add_argument("--save_name", type=str, default="ms_input_tensor.pt")
    args = p.parse_args()

    in_dir = Path(args.in_dir)

    # Load X,Y as (B,10,8) and (B,10)
    X, Y, mode = _load_block_tensors(in_dir)
    B = X.shape[0]

    # Precompute one trial length for validation
    expected_T = 7 * (args.tone_ms + args.isi_ms) + args.tone_ms

    # Render all blocks trialwise
    # Output target: (B,10,trial_T,D)
    # D = 2 or 3 depending on add_trial_start
    D = 3 if args.add_trial_start else 2
    out = torch.zeros((B, 10, expected_T, D), dtype=torch.float32)

    for b in range(B):
        for t in range(10):
            trial = X[b, t]  # (8,)
            trial_ms = render_trial_ms(trial, args.tone_ms, args.isi_ms)  # (T,2)
            if trial_ms.shape[0] != expected_T:
                raise RuntimeError(f"Trial length mismatch: got {trial_ms.shape[0]}, expected {expected_T}")
            out[b, t, :, :2] = trial_ms
            if args.add_trial_start:
                out[b, t, 0, 2] = 1.0

    # Save tensors
    torch.save(out, in_dir / args.save_name)
    torch.save(Y, in_dir / "ms_labels_tensor.pt")

    # trial end index within each trial (0-indexed), constant
    end_idx = torch.tensor([expected_T - 1] * 10, dtype=torch.long)
    torch.save(end_idx, in_dir / "trial_end_indices.pt")

    meta = {
        "mode": "trialwise",
        "source_layout": mode,
        "n_blocks": int(B),
        "tone_ms": args.tone_ms,
        "isi_ms": args.isi_ms,
        "ms_per_trial": int(expected_T),
        "trials_per_block": 10,
        "output_shape": list(out.shape),  # [B,10,T,D]
        "features": ["erb_value", "is_tone"] + (["trial_start"] if args.add_trial_start else []),
        "labels": "deviant position per trial (1-indexed in {4,5,6})",
        "trial_end_indices_file": "trial_end_indices.pt",
    }
    (in_dir / "ms_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:")
    print(" -", (in_dir / args.save_name).resolve(), "shape=", tuple(out.shape))
    print(" -", (in_dir / "ms_labels_tensor.pt").resolve(), "shape=", tuple(Y.shape))
    print(" -", (in_dir / "trial_end_indices.pt").resolve(), "shape=", tuple(end_idx.shape))
    print(" -", (in_dir / "ms_meta.json").resolve())


if __name__ == "__main__":
    main()
