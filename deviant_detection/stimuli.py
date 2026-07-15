from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch


def _import_gm_stimuli():
    """
    Import gm_stimuli.py from the project root / PYTHONPATH.

    Expected symbols:
    - GMConfig
    - make_freq_grid
    - generate_one_block
    - PositionSampler
    """
    try:
        import gm_stimuli  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "无法 import gm_stimuli.py。请确认 gm_stimuli.py 与 train_online.py 在同一目录，"
            "或该目录在 PYTHONPATH 中。原始错误：\n" + str(e)
        )
    required = ["GMConfig", "make_freq_grid", "generate_one_block", "PositionSampler"]
    for name in required:
        if not hasattr(gm_stimuli, name):
            raise RuntimeError(f"gm_stimuli.py 缺少必需的符号: {name}")
    return gm_stimuli


def generate_stimuli_blocks(
    save_dir: Path,
    n_blocks: int,
    trials_per_block: int,
    seq_len: int,
    f_min: float,
    f_max: float,
    f_step: float,
    min_diff: Optional[float],
    seed: int,
    exclude_freqs: List[float],
    exclude_pairs: List[Tuple[float, float]],
    exclude_tol: float,
    no_seen_pairs: bool,
    sample_mode: str,
    round_to: Optional[float],
) -> None:
    """
    Generate and write block-level stimuli:
    - input_blocks.pt  (n_blocks, 10, 8) float32 Hz
    - labels_blocks.pt (n_blocks, 10) int64 deviant positions in {4, 5, 6}
    - meta.json
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    gm = _import_gm_stimuli()
    min_diff_hz = float(min_diff) if min_diff is not None else float(f_step)

    cfg = gm.GMConfig(
        n_blocks=int(n_blocks),
        block_index=1,
        trials_per_block=int(trials_per_block),
        seq_len=int(seq_len),
        f_min=float(f_min),
        f_max=float(f_max),
        f_step=float(f_step),
        min_diff=float(min_diff_hz),
        exclude_freqs=tuple(float(v) for v in exclude_freqs),
        exclude_pairs=tuple((float(a), float(b)) for a, b in exclude_pairs),
        seed=int(seed),
    )

    if cfg.seq_len != 8:
        raise ValueError("This task assumes 8 tones per trial (seq_len=8).")
    if cfg.trials_per_block != 10:
        raise ValueError("This task assumes 10 trials per block (trials_per_block=10).")
    if cfg.deviant_positions != (4, 5, 6):
        raise ValueError("This task assumes deviant_positions=(4,5,6).")

    generator = torch.Generator().manual_seed(cfg.seed)
    grid = gm.make_freq_grid(cfg, exclude_tol=float(exclude_tol)) if sample_mode == "discrete" else None

    x_all = torch.empty((int(n_blocks), cfg.trials_per_block, cfg.seq_len), dtype=torch.float32)
    y_all = torch.empty((int(n_blocks), cfg.trials_per_block), dtype=torch.long)

    prev_pair = None
    seen_pairs: Optional[Set[Tuple[float, float]]] = None if bool(no_seen_pairs) else set()
    position_sampler = gm.PositionSampler(
        trials_per_block=cfg.trials_per_block,
        dev_positions=cfg.deviant_positions,
        g=generator,
    )

    standards: List[float] = []
    deviants: List[float] = []

    for block_idx in range(int(n_blocks)):
        freqs, labels, f_std, f_dev = gm.generate_one_block(
            cfg=cfg,
            grid=grid,
            g=generator,
            block_idx_0=block_idx,
            sample_mode=str(sample_mode),
            round_to=round_to,
            exclude_tol=float(exclude_tol),
            prev_pair=prev_pair,
            seen_pairs=seen_pairs,
            position_sampler=position_sampler,
        )
        prev_pair = (float(f_std), float(f_dev))
        x_all[block_idx] = freqs
        y_all[block_idx] = labels
        standards.append(float(f_std))
        deviants.append(float(f_dev))

    torch.save(x_all, save_dir / "input_blocks.pt")
    torch.save(y_all, save_dir / "labels_blocks.pt")

    meta = asdict(cfg)
    meta.update({
        "export_mode": "all",
        "n_exported": int(n_blocks),
        "excluded_freqs_hz": list(cfg.exclude_freqs),
        "excluded_pairs_hz": [[float(a), float(b)] for a, b in cfg.exclude_pairs],
        "exclude_tol": float(exclude_tol),
        "sample_mode": str(sample_mode),
        "round_to": round_to,
        "label_definition": "deviant position per trial (1-indexed in {4,5,6})",
        "input_definition": "Hz frequencies shaped (n_blocks, 10, 8)",
        "within_block_position_balance": "none",
        "across_block_position_balance": "exact over consecutive 3-block spans via shuffled 30-position queue (10x P4, 10x P5, 10x P6)",
        "block_standard_hz_first10": standards[:10],
        "block_deviant_hz_first10": deviants[:10],
        "avoid_global_pair_reuse": (not bool(no_seen_pairs)),
    })
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[stimuli] Saved:")
    print("  -", (save_dir / "input_blocks.pt").resolve(), "shape=", tuple(x_all.shape))
    print("  -", (save_dir / "labels_blocks.pt").resolve(), "shape=", tuple(y_all.shape))
    print("  -", (save_dir / "meta.json").resolve())


def parse_exclude_pairs(values: Optional[List[str]]) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for raw in values or []:
        value = str(raw).strip()
        if not value:
            continue
        parts = None
        for sep in ("->", ",", ":", "/"):
            if sep in value:
                parts = value.split(sep, 1)
                break
        if parts is None or len(parts) != 2:
            raise ValueError(
                f"Invalid --stimuli_exclude_pairs value {raw!r}. Use directed pairs like 1455,1500."
            )
        pairs.append((float(parts[0]), float(parts[1])))
    return pairs

