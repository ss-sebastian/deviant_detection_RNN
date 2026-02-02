# gm_stimuli.py
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Optional, Set

import torch


@dataclass
class GMConfig:
    n_blocks: int = 100
    block_index: int = 2                 # 1-indexed (used when exporting a single block)
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


# -------------------------
# Sampling helpers
# -------------------------
def _maybe_round(f: float, round_to: Optional[float]) -> float:
    if round_to is None:
        return float(f)
    if round_to <= 0:
        raise ValueError("round_to must be > 0 if provided.")
    return float(round(float(f) / round_to) * round_to)


def _is_excluded(f: float, exclude_freqs: Tuple[float, ...], tol: float) -> bool:
    for ex in exclude_freqs:
        if abs(float(f) - float(ex)) <= tol:
            return True
    return False


def sample_uniform_freq(cfg: GMConfig, g: torch.Generator, round_to: Optional[float]) -> float:
    # Uniform in [f_min, f_max]
    u = float(torch.rand((), generator=g).item())
    f = cfg.f_min + u * (cfg.f_max - cfg.f_min)
    return _maybe_round(f, round_to)


def sample_std_continuous(cfg: GMConfig, g: torch.Generator, round_to: Optional[float], exclude_tol: float) -> float:
    for _ in range(20000):
        f = sample_uniform_freq(cfg, g, round_to)
        if cfg.f_min <= f <= cfg.f_max and (not _is_excluded(f, cfg.exclude_freqs, exclude_tol)):
            return float(f)
    raise RuntimeError("Failed to sample standard frequency after many tries.")


def sample_dev_continuous(cfg: GMConfig, f_std: float, g: torch.Generator, round_to: Optional[float], exclude_tol: float) -> float:
    for _ in range(40000):
        f = sample_uniform_freq(cfg, g, round_to)
        if _is_excluded(f, cfg.exclude_freqs, exclude_tol):
            continue
        if abs(float(f) - float(f_std)) < float(cfg.min_diff):
            continue
        if cfg.f_min <= f <= cfg.f_max:
            return float(f)
    raise RuntimeError("Failed to sample deviant frequency after many tries.")


# -------------------------
# Discrete grid sampling (old behavior)
# -------------------------
def make_freq_grid(cfg: GMConfig, exclude_tol: float) -> torch.Tensor:
    if cfg.f_step <= 0:
        raise ValueError("f_step must be > 0")
    if cfg.f_max < cfg.f_min:
        raise ValueError("f_max must be >= f_min")

    n = int(torch.floor(torch.tensor((cfg.f_max - cfg.f_min) / cfg.f_step)).item()) + 1
    grid = cfg.f_min + torch.arange(n, dtype=torch.float32) * cfg.f_step
    grid = grid[grid <= (cfg.f_max + 1e-6)]

    # exclude exact frequencies (with tol)
    if cfg.exclude_freqs:
        excl = torch.tensor(list(cfg.exclude_freqs), dtype=torch.float32)
        mask = torch.ones_like(grid, dtype=torch.bool)
        for v in excl:
            mask &= (torch.abs(grid - v) > exclude_tol)
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


# -------------------------
# Deviant position scheduling
# -------------------------
def make_block_position_schedule(
    block_idx_0: int,
    trials_per_block: int,
    dev_positions: Tuple[int, ...],
    g: torch.Generator
) -> List[int]:
    """
    Within one block, make dev_pos distribution as even as possible.
    For 10 trials and 3 positions -> 4/3/3. Which position gets the extra +1
    rotates with block_idx_0 so that across many blocks the global ratio ~ 1/3.
    Then shuffle within the block.
    """
    if len(dev_positions) != 3:
        raise ValueError("This schedule assumes exactly 3 deviant positions (e.g., 4/5/6).")

    base = trials_per_block // 3
    rem = trials_per_block % 3
    counts = [base, base, base]
    for k in range(rem):
        counts[(block_idx_0 + k) % 3] += 1

    pos_list: List[int] = []
    for p, c in zip(dev_positions, counts):
        pos_list.extend([int(p)] * int(c))

    perm = torch.randperm(len(pos_list), generator=g).tolist()
    return [pos_list[i] for i in perm]


# -------------------------
# Block pair sampling
# -------------------------
def sample_block_pair(
    cfg: GMConfig,
    grid: Optional[torch.Tensor],
    g: torch.Generator,
    sample_mode: str,
    round_to: Optional[float],
    exclude_tol: float,
    prev_pair: Optional[Tuple[float, float]] = None,
    seen_pairs: Optional[Set[Tuple[float, float]]] = None,
    max_tries: int = 2000,
) -> Tuple[float, float]:
    """
    Sample one (f_std, f_dev) pair for the whole block.
    Ensure it differs from prev_pair; optionally avoid global repeats via seen_pairs.

    sample_mode:
      - "discrete": use f_step grid (old behavior)
      - "continuous": uniform sampling in [f_min,f_max] with optional rounding
    """
    if sample_mode not in ("discrete", "continuous"):
        raise ValueError("sample_mode must be 'discrete' or 'continuous'.")

    for _ in range(max_tries):
        if sample_mode == "continuous":
            f_std = sample_std_continuous(cfg, g, round_to, exclude_tol)
            f_dev = sample_dev_continuous(cfg, f_std, g, round_to, exclude_tol)
        else:
            assert grid is not None
            f_std = sample_from_grid(grid, g)
            f_dev = sample_deviant_freq(grid, f_std, cfg.min_diff, g)

        pair = (float(f_std), float(f_dev))

        if prev_pair is not None and pair == prev_pair:
            continue
        if seen_pairs is not None and pair in seen_pairs:
            continue

        if seen_pairs is not None:
            seen_pairs.add(pair)
        return float(f_std), float(f_dev)

    # fallback: only avoid previous
    while True:
        if sample_mode == "continuous":
            f_std = sample_std_continuous(cfg, g, round_to, exclude_tol)
            f_dev = sample_dev_continuous(cfg, f_std, g, round_to, exclude_tol)
        else:
            assert grid is not None
            f_std = sample_from_grid(grid, g)
            f_dev = sample_deviant_freq(grid, f_std, cfg.min_diff, g)

        pair = (float(f_std), float(f_dev))
        if prev_pair is None or pair != prev_pair:
            return float(f_std), float(f_dev)


def generate_one_block(
    cfg: GMConfig,
    grid: Optional[torch.Tensor],
    g: torch.Generator,
    block_idx_0: int,
    sample_mode: str,
    round_to: Optional[float],
    exclude_tol: float,
    prev_pair: Optional[Tuple[float, float]],
    seen_pairs: Optional[Set[Tuple[float, float]]],
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    One block:
      - fixed f_std and fixed f_dev across all 10 trials
      - deviant position changes (4/5/6) with balanced schedule
    Returns:
      freqs_hz: (10,8) float32
      labels:   (10,)  int64 values in {4,5,6}
      f_std, f_dev
    """
    f_std, f_dev = sample_block_pair(
        cfg=cfg,
        grid=grid,
        g=g,
        sample_mode=sample_mode,
        round_to=round_to,
        exclude_tol=exclude_tol,
        prev_pair=prev_pair,
        seen_pairs=seen_pairs,
    )

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
    p.add_argument("--block_index", type=int, default=1, help="1-indexed block number to export (single-block mode)")

    # export controls
    p.add_argument("--export_all", action="store_true", help="Export all blocks into input_blocks.pt / labels_blocks.pt")
    p.add_argument("--export_n", type=int, default=None, help="Export first N blocks into input_blocks.pt / labels_blocks.pt")

    p.add_argument("--f_min", type=float, required=True)
    p.add_argument("--f_max", type=float, required=True)
    p.add_argument("--f_step", type=float, required=True)
    p.add_argument("--min_diff", type=float, default=None, help="min |f_dev-f_std|; default=f_step")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--trials_per_block", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=8)

    p.add_argument("--exclude_freqs", type=float, nargs="*", default=[1455.0, 1500.0, 1600.0])
    p.add_argument("--exclude_tol", type=float, default=1e-6, help="tolerance for excluding specific frequencies")
    p.add_argument("--no_seen_pairs", action="store_true", help="Do not avoid reusing (std,dev) pairs globally")

    # NEW: sampling mode
    p.add_argument("--sample_mode", type=str, default="continuous", choices=["continuous", "discrete"],
                   help="continuous: uniform sampling in [f_min,f_max]; discrete: sample on f_step grid")
    p.add_argument("--round_to", type=float, default=None,
                   help="Optional rounding resolution in Hz for continuous sampling (e.g., 1 or 0.1).")

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

    # constraints for your paradigm
    if cfg.seq_len != 8:
        raise ValueError("This task assumes 8 tones per trial (seq_len=8).")
    if cfg.trials_per_block != 10:
        raise ValueError("This task assumes 10 trials per block (trials_per_block=10).")
    if cfg.deviant_positions != (4, 5, 6):
        raise ValueError("This task assumes deviant_positions=(4,5,6).")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(cfg.seed)

    # grid only needed for discrete mode
    grid = None
    if args.sample_mode == "discrete":
        grid = make_freq_grid(cfg, exclude_tol=args.exclude_tol)

    export_mode_all = bool(args.export_all)
    export_mode_n = args.export_n is not None

    if export_mode_all and export_mode_n:
        raise ValueError("Use only one of --export_all or --export_n, not both.")

    if export_mode_all or export_mode_n:
        n_export = cfg.n_blocks if export_mode_all else int(args.export_n)
        if n_export <= 0:
            raise ValueError("--export_n must be > 0")
        if n_export > cfg.n_blocks:
            raise ValueError("--export_n cannot exceed --n_blocks")

        X_all = torch.empty((n_export, cfg.trials_per_block, cfg.seq_len), dtype=torch.float32)
        Y_all = torch.empty((n_export, cfg.trials_per_block), dtype=torch.long)

        prev_pair = None
        seen_pairs = None if args.no_seen_pairs else set()

        stds: List[float] = []
        devs: List[float] = []

        for b in range(n_export):
            freqs, labels, f_std, f_dev = generate_one_block(
                cfg=cfg,
                grid=grid,
                g=g,
                block_idx_0=b,
                sample_mode=args.sample_mode,
                round_to=args.round_to,
                exclude_tol=args.exclude_tol,
                prev_pair=prev_pair,
                seen_pairs=seen_pairs,
            )
            prev_pair = (f_std, f_dev)
            X_all[b] = freqs
            Y_all[b] = labels
            stds.append(f_std)
            devs.append(f_dev)

        torch.save(X_all, save_dir / "input_blocks.pt")
        torch.save(Y_all, save_dir / "labels_blocks.pt")

        meta = asdict(cfg)
        meta.update({
            "export_mode": "all" if export_mode_all else "first_n",
            "n_exported": int(n_export),
            "excluded_freqs_hz": list(cfg.exclude_freqs),
            "exclude_tol": float(args.exclude_tol),
            "sample_mode": args.sample_mode,
            "round_to": args.round_to,
            "label_definition": "deviant position per trial (1-indexed in {4,5,6})",
            "input_definition": "Hz frequencies shaped (n_blocks, 10, 8)",
            "within_block_position_balance": "balanced schedule per block; extra count rotates across blocks",
            "block_standard_hz_first10": stds[:10],
            "block_deviant_hz_first10": devs[:10],
            "avoid_global_pair_reuse": (not args.no_seen_pairs),
        })
        (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("Saved:")
        print(" -", (save_dir / "input_blocks.pt").resolve(), "shape=", tuple(X_all.shape))
        print(" -", (save_dir / "labels_blocks.pt").resolve(), "shape=", tuple(Y_all.shape))
        print(" -", (save_dir / "meta.json").resolve())

    else:
        # single-block export
        if not (1 <= cfg.block_index <= cfg.n_blocks):
            raise ValueError(f"block_index must be within [1, n_blocks]. Got {cfg.block_index}/{cfg.n_blocks}.")

        target = cfg.block_index - 1
        out_freqs = None
        out_labels = None
        out_std = None
        out_dev = None

        prev_pair = None
        seen_pairs = None if args.no_seen_pairs else set()

        for b in range(cfg.n_blocks):
            freqs, labels, f_std, f_dev = generate_one_block(
                cfg=cfg,
                grid=grid,
                g=g,
                block_idx_0=b,
                sample_mode=args.sample_mode,
                round_to=args.round_to,
                exclude_tol=args.exclude_tol,
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
            "export_mode": "single",
            "exported_block_index": cfg.block_index,
            "excluded_freqs_hz": list(cfg.exclude_freqs),
            "exclude_tol": float(args.exclude_tol),
            "sample_mode": args.sample_mode,
            "round_to": args.round_to,
            "block_standard_hz": out_std,
            "block_deviant_hz": out_dev,
            "label_definition": "deviant position per trial (1-indexed in {4,5,6})",
            "input_definition": "Hz frequencies shaped (1, 10, 8)",
            "within_block_position_balance": "balanced schedule per block; extra count rotates across blocks",
            "avoid_global_pair_reuse": (not args.no_seen_pairs),
        })
        (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("Saved:")
        print(" -", (save_dir / "input_tensor.pt").resolve(), "shape=", tuple(out_freqs.shape))
        print(" -", (save_dir / "labels_tensor.pt").resolve(), "shape=", tuple(out_labels.shape))
        print(" - block standard/dev:", out_std, out_dev)
        print(" - excluded:", cfg.exclude_freqs)


if __name__ == "__main__":
    main()
