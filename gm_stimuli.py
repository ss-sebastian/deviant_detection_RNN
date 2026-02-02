# gm_stimuli.py
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Set

import torch


@dataclass
class GMConfig:
    n_blocks: int = 100
    block_index: int = 2                 # 1-indexed
    trials_per_block: int = 10
    seq_len: int = 8

    # deviant can only be at 4/5/6 (1-indexed within trial)
    deviant_positions: Tuple[int, ...] = (4, 5, 6)

    # frequency sampling
    f_min: float = 1000.0
    f_max: float = 2000.0
    f_step: float = 5.0
    min_diff: float = 5.0               # |f_dev - f_std| >= min_diff

    # exclude these exact frequencies (Hz)
    exclude_freqs: Tuple[float, ...] = (1455.0, 1500.0, 1600.0)

    seed: int = 42


def make_freq_grid(cfg: GMConfig) -> torch.Tensor:
    if cfg.f_step <= 0:
        raise ValueError("f_step must be > 0")
    if cfg.f_max < cfg.f_min:
        raise ValueError("f_max must be >= f_min")

    n = int(torch.floor(torch.tensor((cfg.f_max - cfg.f_min) / cfg.f_step)).item()) + 1
    grid = cfg.f_min + torch.arange(n, dtype=torch.float32) * cfg.f_step
    grid = grid[grid <= (cfg.f_max + 1e-6)]

    # exclude exact frequencies
    if cfg.exclude_freqs:
        excl = torch.tensor(list(cfg.exclude_freqs), dtype=torch.float32)
        mask = torch.ones_like(grid, dtype=torch.bool)
        for v in excl:
            mask &= (torch.abs(grid - v) > 1e-6)
        grid = grid[mask]

    if grid.numel() < 2:
        raise ValueError("Frequency grid must contain at least 2 values after exclusions.")
    return grid


def sample_from_grid(grid: torch.Tensor, g: torch.Generator) -> float:
    idx = torch.randint(0, grid.numel(), (1,), generator=g).item()
    return float(grid[idx].item())


def sample_deviant_freq(grid: torch.Tensor, f_std: float, min_diff: float, g: torch.Generator) -> float:
    diffs = torch.abs(grid - float(f_std))
    candidates = grid[diffs >= float(min_diff)]
    if candidates.numel() == 0:
        raise ValueError(
            f"No deviant candidates satisfy min_diff={min_diff}. "
            f"Try smaller min_diff or enlarge frequency range/step."
        )
    idx = torch.randint(0, candidates.numel(), (1,), generator=g).item()
    return float(candidates[idx].item())


def make_block_position_schedule(
    block_idx_0: int,
    trials_per_block: int,
    dev_positions: Tuple[int, ...],
    g: torch.Generator
) -> List[int]:
    """
    For trials_per_block=10 and dev_positions=(4,5,6),
    we want within-block distribution as even as possible: 4/3/3.
    Rotate which position gets the extra +1 across blocks to make
    global counts close to 1/3 each over many blocks.
    Then shuffle within the block.
    """
    if len(dev_positions) != 3:
        raise ValueError("This schedule assumes exactly 3 deviant positions (e.g., 4/5/6).")
    if trials_per_block != 10:
        # generic fallback: distribute as evenly as possible
        base = trials_per_block // 3
        rem = trials_per_block % 3
        counts = [base, base, base]
        # rotate who gets the remainder
        for k in range(rem):
            counts[(block_idx_0 + k) % 3] += 1
    else:
        # exactly 10 -> 4,3,3 with rotation
        counts = [3, 3, 3]
        counts[block_idx_0 % 3] += 1  # one of them becomes 4

    pos_list: List[int] = []
    for p, c in zip(dev_positions, counts):
        pos_list.extend([int(p)] * int(c))

    # shuffle
    perm = torch.randperm(len(pos_list), generator=g).tolist()
    pos_list = [pos_list[i] for i in perm]
    return pos_list


def generate_one_block(cfg: GMConfig, grid: torch.Tensor, g: torch.Generator, block_idx_0: int,
                       prev_pair: Tuple[float, float] | None,
                       seen_pairs: Set[Tuple[float, float]] | None) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    One block:
      - sample one f_std and one f_dev (fixed for all trials in the block)
      - deviant position varies across the 10 trials using a balanced schedule
    Returns:
      freqs_hz: (10,8) float32
      labels:   (10,)  int64 values in {4,5,6}
      f_std, f_dev
    """
    # Choose a (std, dev) pair. Guarantee not identical to previous block pair.
    # Also try to avoid repeats globally if seen_pairs provided, but don't get stuck.
    max_tries = 2000
    for _ in range(max_tries):
        f_std = sample_from_grid(grid, g)
        f_dev = sample_deviant_freq(grid, f_std, cfg.min_diff, g)

        pair = (float(f_std), float(f_dev))
        if prev_pair is not None and pair == prev_pair:
            continue
        if seen_pairs is not None and pair in seen_pairs:
            continue
        # ok
        if seen_pairs is not None:
            seen_pairs.add(pair)
        break
    else:
        # fallback: only avoid previous
        while True:
            f_std = sample_from_grid(grid, g)
            f_dev = sample_deviant_freq(grid, f_std, cfg.min_diff, g)
            pair = (float(f_std), float(f_dev))
            if prev_pair is None or pair != prev_pair:
                break

    # Position schedule (balanced within block + rotated across blocks)
    pos_schedule = make_block_position_schedule(
        block_idx_0=block_idx_0,
        trials_per_block=cfg.trials_per_block,
        dev_positions=cfg.deviant_positions,
        g=g,
    )
    if len(pos_schedule) != cfg.trials_per_block:
        raise RuntimeError("Position schedule length mismatch.")

    freqs = torch.full((cfg.trials_per_block, cfg.seq_len), float(f_std), dtype=torch.float32)
    labels = torch.empty((cfg.trials_per_block,), dtype=torch.long)

    for t in range(cfg.trials_per_block):
        dev_pos = int(pos_schedule[t])  # 1-indexed
        labels[t] = dev_pos
        freqs[t, dev_pos - 1] = float(f_dev)

    return freqs, labels, float(f_std), float(f_dev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_blocks", type=int, required=True)
    p.add_argument("--block_index", type=int, required=True, help="1-indexed block number to export")
    p.add_argument("--f_min", type=float, required=True)
    p.add_argument("--f_max", type=float, required=True)
    p.add_argument("--f_step", type=float, required=True)
    p.add_argument("--min_diff", type=float, default=None, help="min |f_dev-f_std|; default=f_step")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, required=True)

    # keep fixed but overridable if you insist
    p.add_argument("--trials_per_block", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=8)

    # exclusions
    p.add_argument("--exclude_freqs", type=float, nargs="*", default=[1455.0, 1500.0, 1600.0])

    args = p.parse_args()

    min_diff = args.min_diff if args.min_diff is not None else args.f_step

    cfg = GMConfig(
        n_blocks=args.n_blocks,
        block_index=args.block_index,
        trials_per_block=args.trials_per_block,
        seq_len=args.seq_len,
        f_min=args.f_min,
        f_max=args.f_max,
        f_step=args.f_step,
        min_diff=min_diff,
        exclude_freqs=tuple(float(v) for v in args.exclude_freqs),
        seed=args.seed,
    )

    if not (1 <= cfg.block_index <= cfg.n_blocks):
        raise ValueError(f"block_index must be within [1, n_blocks]. Got {cfg.block_index}/{cfg.n_blocks}.")
    if cfg.seq_len != 8:
        raise ValueError("This task assumes 8 tones per trial (seq_len=8).")
    if cfg.trials_per_block != 10:
        raise ValueError("This task assumes 10 trials per block (trials_per_block=10).")
    if cfg.deviant_positions != (4, 5, 6):
        raise ValueError("This task assumes deviant_positions=(4,5,6).")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    grid = make_freq_grid(cfg)
    g = torch.Generator().manual_seed(cfg.seed)

    target = cfg.block_index - 1
    out_freqs = None
    out_labels = None
    out_std = None
    out_dev = None

    prev_pair = None
    # If you generate tons of blocks, this set can grow huge; keep it optional.
    # Here we keep it to ensure blocks differ broadly; if memory becomes an issue, set to None.
    seen_pairs: Set[Tuple[float, float]] | None = set()

    for b in range(cfg.n_blocks):
        freqs, labels, f_std, f_dev = generate_one_block(
            cfg=cfg,
            grid=grid,
            g=g,
            block_idx_0=b,
            prev_pair=prev_pair,
            seen_pairs=seen_pairs,
        )
        prev_pair = (f_std, f_dev)

        if b == target:
            out_freqs = freqs.unsqueeze(0)   # (1,10,8)
            out_labels = labels.unsqueeze(0) # (1,10)
            out_std = f_std
            out_dev = f_dev

    assert out_freqs is not None and out_labels is not None

    torch.save(out_freqs, save_dir / "input_tensor.pt")
    torch.save(out_labels, save_dir / "labels_tensor.pt")

    meta = asdict(cfg)
    meta.update({
        "exported_block_index": cfg.block_index,
        "grid_size": int(grid.numel()),
        "block_standard_hz": out_std,
        "block_deviant_hz": out_dev,
        "label_definition": "deviant position per trial (1-indexed in {4,5,6})",
        "input_definition": "Hz frequencies shaped (1, trials_per_block=10, seq_len=8)",
        "within_block_position_balance": "10 trials -> counts approximately 4/3/3 across positions 4/5/6 (rotated across blocks)",
        "excluded_freqs_hz": list(cfg.exclude_freqs),
    })
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:")
    print(" -", (save_dir / "input_tensor.pt").resolve(), "shape=", tuple(out_freqs.shape))
    print(" -", (save_dir / "labels_tensor.pt").resolve(), "shape=", tuple(out_labels.shape))
    print(" - block standard/dev:", out_std, out_dev)
    print(" - excluded:", cfg.exclude_freqs)


if __name__ == "__main__":
    main()
