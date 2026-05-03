#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_online.py

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Optional, List, Set, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from model import PredictiveGRU, ModelConfig

# --- optional metrics deps (AUC/F1) ---
try:
    from sklearn.metrics import f1_score, roc_auc_score
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

# ============================================================
# (NEW) Stimuli generation (import gm_stimuli and generate files)
# ============================================================
def _import_gm_stimuli():
    """
    Import gm_stimuli.py from the same directory / PYTHONPATH.
    We expect your gm_stimuli.py to define:
      - GMConfig
      - make_freq_grid
      - generate_one_block
    """
    try:
        import gm_stimuli  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "无法 import gm_stimuli.py。请确认 gm_stimuli.py 与 train_online.py 在同一目录，"
            "或该目录在 PYTHONPATH 中。原始错误：\n" + str(e)
        )
    required = ["GMConfig", "make_freq_grid", "generate_one_block"]
    for k in required:
        if not hasattr(gm_stimuli, k):
            raise RuntimeError(f"gm_stimuli.py 缺少必需的符号: {k}")
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
    exclude_tol: float,
    no_seen_pairs: bool,
    sample_mode: str,
    round_to: Optional[float],
) -> None:
    """
    Generate and write:
      - input_blocks.pt  (n_blocks, 10, 8) float32 (Hz)
      - labels_blocks.pt (n_blocks, 10)    int64   in {4,5,6}
      - meta.json
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    gm = _import_gm_stimuli()

    # match gm_stimuli default: min_diff defaults to f_step
    md = float(min_diff) if min_diff is not None else float(f_step)

    cfg = gm.GMConfig(
        n_blocks=int(n_blocks),
        block_index=1,
        trials_per_block=int(trials_per_block),
        seq_len=int(seq_len),
        f_min=float(f_min),
        f_max=float(f_max),
        f_step=float(f_step),
        min_diff=float(md),
        exclude_freqs=tuple(float(v) for v in exclude_freqs),
        seed=int(seed),
    )

    # paradigm constraints (same as your gm_stimuli)
    if cfg.seq_len != 8:
        raise ValueError("This task assumes 8 tones per trial (seq_len=8).")
    if cfg.trials_per_block != 10:
        raise ValueError("This task assumes 10 trials per block (trials_per_block=10).")
    if cfg.deviant_positions != (4, 5, 6):
        raise ValueError("This task assumes deviant_positions=(4,5,6).")

    g = torch.Generator().manual_seed(cfg.seed)

    grid = None
    if sample_mode == "discrete":
        grid = gm.make_freq_grid(cfg, exclude_tol=float(exclude_tol))

    X_all = torch.empty((int(n_blocks), cfg.trials_per_block, cfg.seq_len), dtype=torch.float32)
    Y_all = torch.empty((int(n_blocks), cfg.trials_per_block), dtype=torch.long)

    prev_pair = None
    seen_pairs: Optional[Set[Tuple[float, float]]] = None if bool(no_seen_pairs) else set()

    stds: List[float] = []
    devs: List[float] = []

    for b in range(int(n_blocks)):
        freqs, labels, f_std, f_dev = gm.generate_one_block(
            cfg=cfg,
            grid=grid,
            g=g,
            block_idx_0=b,
            sample_mode=str(sample_mode),
            round_to=round_to,
            exclude_tol=float(exclude_tol),
            prev_pair=prev_pair,
            seen_pairs=seen_pairs,
        )
        prev_pair = (float(f_std), float(f_dev))
        X_all[b] = freqs
        Y_all[b] = labels
        stds.append(float(f_std))
        devs.append(float(f_dev))

    torch.save(X_all, save_dir / "input_blocks.pt")
    torch.save(Y_all, save_dir / "labels_blocks.pt")

    meta = asdict(cfg)
    meta.update({
        "export_mode": "all",
        "n_exported": int(n_blocks),
        "excluded_freqs_hz": list(cfg.exclude_freqs),
        "exclude_tol": float(exclude_tol),
        "sample_mode": str(sample_mode),
        "round_to": round_to,
        "label_definition": "deviant position per trial (1-indexed in {4,5,6})",
        "input_definition": "Hz frequencies shaped (n_blocks, 10, 8)",
        "within_block_position_balance": "balanced schedule per block; extra count rotates across blocks",
        "block_standard_hz_first10": stds[:10],
        "block_deviant_hz_first10": devs[:10],
        "avoid_global_pair_reuse": (not bool(no_seen_pairs)),
    })
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[stimuli] Saved:")
    print("  -", (save_dir / "input_blocks.pt").resolve(), "shape=", tuple(X_all.shape))
    print("  -", (save_dir / "labels_blocks.pt").resolve(), "shape=", tuple(Y_all.shape))
    print("  -", (save_dir / "meta.json").resolve())


# -------------------------
# Utilities
# -------------------------
def labels_to_class_index(y_456: torch.Tensor) -> torch.Tensor:
    """
    y_456: (...,) values in {4,5,6} (1-indexed tone position)
    map to {0,1,2}
    """
    return (y_456 - 4).long()


def infer_end_indices_from_T(T: int, trials_per_block: int = 10) -> torch.Tensor:
    if T % trials_per_block != 0:
        raise ValueError(f"Cannot infer trial length: T={T} not divisible by {trials_per_block}")
    trial_T = T // trials_per_block
    return torch.tensor([(i + 1) * trial_T - 1 for i in range(trials_per_block)], dtype=torch.long)


def hz_to_erb_rate_np(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * f_hz)


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


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def safe_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not _HAVE_SKLEARN:
        return float("nan")
    try:
        return float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        return float("nan")


def safe_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 3) -> float:
    if not _HAVE_SKLEARN:
        return float("nan")
    try:
        # y_true shape (N,), y_prob shape (N,C)
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(n_classes))))
    except Exception:
        return float("nan")


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv_row(path: Path, header: List[str], row: Dict[str, Any]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def make_run_name(parts: Dict[str, Any]) -> str:
    # compact, filesystem-safe
    def fmt(v):
        if isinstance(v, float):
            s = f"{v:.6g}"
        else:
            s = str(v)
        return s.replace("/", "_").replace(" ", "")
    items = [f"{k}={fmt(v)}" for k, v in parts.items()]
    return "__".join(items)

def _is_valid_y_pos_456(y: torch.Tensor) -> bool:
    # y expected shape (...,10) with values in {4,5,6}
    return bool(torch.all((y == 4) | (y == 5) | (y == 6)).item())

def _unique_list(y: torch.Tensor, max_n: int = 30) -> List[int]:
    u = torch.unique(y.detach().cpu())
    if u.numel() > max_n:
        return u[:max_n].tolist() + ["..."]  # type: ignore
    return u.tolist()

def _dump_bad_batch(
    dump_dir: Path,
    tag: str,
    info: Dict[str, Any],
    tensors: Dict[str, torch.Tensor],
    max_dumps: int,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dump_dir.glob("bad_*.pt"))
    if len(existing) >= int(max_dumps):
        return
    path = dump_dir / f"bad_{tag}.pt"
    payload = {"info": info, "tensors": {}}
    for k, v in tensors.items():
        try:
            payload["tensors"][k] = v.detach().cpu()
        except Exception:
            pass
    torch.save(payload, path)
    print(f"[debug_labels] dumped -> {path.resolve()}")

def _check_y_and_maybe_debug(
    y_cpu: torch.Tensor,
    where: str,
    debug_on: bool,
    fatal: bool,
    dump_on: bool,
    dump_dir: Optional[Path],
    dump_tag: str,
    dump_info: Dict[str, Any],
    dump_tensors: Optional[Dict[str, torch.Tensor]] = None,
    max_dumps: int = 10,
) -> bool:
    """
    Returns True if y valid else False.
    """
    if not debug_on:
        return True
    y_cpu = y_cpu.detach().cpu().long()
    ok = _is_valid_y_pos_456(y_cpu)
    if ok:
        return True

    uniq = _unique_list(y_cpu)
    msg = f"[debug_labels] INVALID y at {where}: unique={uniq}"
    if fatal:
        raise RuntimeError(msg)
    else:
        print(msg)

    if dump_on and dump_dir is not None:
        tensors = dump_tensors or {}
        _dump_bad_batch(
            dump_dir=dump_dir,
            tag=dump_tag,
            info={"where": where, "unique": uniq, **dump_info},
            tensors={"y_cpu": y_cpu, **tensors},
            max_dumps=max_dumps,
        )
    return False

# -------------------------
# ERB one-hot renderer
# -------------------------
def make_erb_edges(
    f_min_hz: float,
    f_max_hz: float,
    n_bins: int,
) -> np.ndarray:
    erb_min = float(hz_to_erb_rate_np(np.array([f_min_hz], dtype=np.float32))[0])
    erb_max = float(hz_to_erb_rate_np(np.array([f_max_hz], dtype=np.float32))[0])
    edges = np.linspace(erb_min, erb_max, int(n_bins) + 1, dtype=np.float32)
    return edges


def freq_to_bin_erb(
    f_hz: float,
    edges_erb: np.ndarray,
) -> int:
    erb = float(hz_to_erb_rate_np(np.array([f_hz], dtype=np.float32))[0])
    j = int(np.searchsorted(edges_erb, erb, side="right") - 1)
    j = max(0, min(j, int(edges_erb.shape[0] - 2)))
    return j


def compute_trial_gap_hz(
    freqs_block: torch.Tensor,   # (10,8) float Hz
    y_pos_456: torch.Tensor,     # (10,) in {4,5,6}
) -> torch.Tensor:
    """
    Per-trial absolute frequency gap |f_dev - f_std| in Hz.
    We use the first tone as the standard frequency because stimuli are
    standard-standard-standard before the deviant in this paradigm.
    """
    dev_idx = (y_pos_456.long() - 1).clamp(min=0, max=7)
    std_freq = freqs_block[:, 0]
    dev_freq = freqs_block.gather(1, dev_idx.unsqueeze(1)).squeeze(1)
    return (dev_freq - std_freq).abs().float()


def unpack_batch_with_optional_gap(
    batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
    raise ValueError(f"Unexpected batch structure: {type(batch)}")


def make_gap_weight_tensor(
    gap_hz_bt: Optional[torch.Tensor],
    power: float,
    ref_hz: float,
    max_weight: float,
) -> Optional[torch.Tensor]:
    """
    Convert per-trial frequency gaps into normalized training weights.
    Smaller gaps get larger weights because they are harder and more likely to
    drive the human-like RT structure we care about.
    """
    if gap_hz_bt is None or float(power) <= 0.0:
        return None

    gap = gap_hz_bt.float().clamp(min=1.0)
    weights = (float(ref_hz) / gap).pow(float(power))

    max_w = max(1.0, float(max_weight))
    min_w = 1.0 / max_w
    weights = torch.clamp(weights, min=min_w, max=max_w)
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights


def scheduled_value(
    start_value: float,
    final_value: float,
    epoch_in_stage_1based: int,
    schedule_epochs: int,
) -> float:
    if int(schedule_epochs) <= 1 or not np.isfinite(float(final_value)) or float(final_value) < 0:
        return float(start_value)
    if float(start_value) == float(final_value):
        return float(start_value)

    progress = min(1.0, max(0.0, (int(epoch_in_stage_1based) - 1) / float(int(schedule_epochs) - 1)))
    return float(start_value) + progress * (float(final_value) - float(start_value))


# -------------------------
# Dataset: tokenized one-hot (10ms)
# -------------------------
class OnlineRenderDataset(Dataset):
    """
    Returns one block per item:
      x_flat: (T_tokens, D) where T_tokens = 10 * trial_T_tokens
      y:      (10,) labels in {4,5,6}
    """

    def __init__(
        self,
        data_dir: Path,
        seed: int,
        tone_ms: int,
        isi_ms: int,
        ramp_ms: int,  # kept for compatibility; not used in one-hot
        token_ms: int,
        f_min_hz: float,
        f_max_hz: float,
        n_bins: int,
        add_eos: bool,
        sigma_other_noise: float,
        p_other_noise: float,
        sigma_silence_noise: float,
        resample_noise_per_epoch: bool = False,
        quiet: bool = False,
        assert_labels: bool = True,
    ):
        self.data_dir = data_dir
        self.seed = int(seed)

        self.tone_ms = int(tone_ms)
        self.isi_ms = int(isi_ms)
        self.ramp_ms = int(ramp_ms)
        self.assert_labels = bool(assert_labels)

        self.token_ms = int(token_ms)
        if self.token_ms <= 0:
            raise ValueError("token_ms must be positive.")

        self.X, self.Y, self.layout = load_blocks_or_single(data_dir)
        self.B = int(self.X.shape[0])

        self.trial_T_ms = 7 * (self.tone_ms + self.isi_ms) + self.tone_ms
        if self.trial_T_ms <= 0:
            raise ValueError("Invalid tone_ms/isi_ms leading to non-positive trial length.")

        if self.trial_T_ms % self.token_ms != 0:
            raise ValueError(f"trial_T_ms={self.trial_T_ms} not divisible by token_ms={self.token_ms}")
        if self.tone_ms % self.token_ms != 0:
            raise ValueError(f"tone_ms={self.tone_ms} not divisible by token_ms={self.token_ms}")
        if self.isi_ms % self.token_ms != 0:
            raise ValueError(f"isi_ms={self.isi_ms} not divisible by token_ms={self.token_ms}")

        self.tone_T = self.tone_ms // self.token_ms
        self.isi_T = self.isi_ms // self.token_ms
        self.step_T = self.tone_T + self.isi_T

        self.trial_T_tokens = self.trial_T_ms // self.token_ms
        expected = 8 * self.tone_T + 7 * self.isi_T
        if expected != self.trial_T_tokens:
            raise RuntimeError(f"Token length mismatch: expected={expected} got={self.trial_T_tokens}")

        self.T = 10 * self.trial_T_tokens

        self.f_min_hz = float(f_min_hz)
        self.f_max_hz = float(f_max_hz)
        self.n_bins = int(n_bins)
        if self.n_bins <= 2:
            raise ValueError("n_bins must be >2")

        self.edges_erb = make_erb_edges(self.f_min_hz, self.f_max_hz, self.n_bins)
        self.add_eos = bool(add_eos)
        self.eos_dim = 1 if self.add_eos else 0
        self.input_dim = self.n_bins + self.eos_dim

        self.sigma_other_noise = float(sigma_other_noise)
        self.p_other_noise = float(p_other_noise)
        self.sigma_silence_noise = float(sigma_silence_noise)
        self.resample_noise_per_epoch = bool(resample_noise_per_epoch)
        self.current_epoch = 0

        if not quiet:
            # ---- sanity check: bin clipping / collisions
            Xhz = self.X.numpy()  # (B,10,8)
            bins = np.zeros_like(Xhz, dtype=np.int32)
            for b in range(Xhz.shape[0]):
                for t in range(Xhz.shape[1]):
                    for i in range(Xhz.shape[2]):
                        bins[b, t, i] = freq_to_bin_erb(float(Xhz[b, t, i]), self.edges_erb)

            clip_lo = (bins == 0).mean()
            clip_hi = (bins == (self.n_bins - 1)).mean()
            uniq_std = len(np.unique(bins[:, :, 0:3]))
            uniq_456 = len(np.unique(bins[:, :, 3:6]))

            std_bin = bins[:, :, 0]
            pred = np.abs(bins[:, :, 3:6] - std_bin[:, :, None]).argmax(axis=2) + 4
            acc_bin = (pred == self.Y.numpy()).mean()

            print(f"[sanity bins] clip_lo={clip_lo:.3f} clip_hi={clip_hi:.3f} "
                  f"uniq_std={uniq_std} uniq_456={uniq_456} rule_acc_on_bins={acc_bin:.3f}")

    def __len__(self) -> int:
        return self.B

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _seed_for(self, idx: int, trial: int) -> int:
        epoch_offset = (self.current_epoch * 1_000_003) if self.resample_noise_per_epoch else 0
        return self.seed + epoch_offset + idx * 1000 + trial * 17

    def _render_trial_tokens_onehot(self, freqs_8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        D = self.input_dim
        X = np.zeros((self.trial_T_tokens, D), dtype=np.float32)

        t = 0
        for i in range(8):
            bin_i = freq_to_bin_erb(float(freqs_8[i]), self.edges_erb)

            for _ in range(self.tone_T):
                X[t, bin_i] = 1.0
                if self.sigma_other_noise > 0 and self.p_other_noise > 0:
                    if float(rng.random()) < self.p_other_noise:
                        noise = rng.normal(0.0, self.sigma_other_noise, size=(self.n_bins,)).astype(np.float32)
                        noise[bin_i] = 0.0
                        X[t, :self.n_bins] += noise
                t += 1

            if i < 7:
                if self.sigma_silence_noise > 0:
                    for _ in range(self.isi_T):
                        X[t, :self.n_bins] = rng.normal(
                            0.0, self.sigma_silence_noise, size=(self.n_bins,)
                        ).astype(np.float32)
                        t += 1
                else:
                    t += self.isi_T

        if t != self.trial_T_tokens:
            raise RuntimeError(f"Trial token fill mismatch: t={t} trial_T_tokens={self.trial_T_tokens}")

        if self.add_eos:
            X[-1, self.n_bins] = 1.0

        return X

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.Y[idx]  # (10,) in {4,5,6}
        freqs_block_t = self.X[idx].float()                   # (10,8)
        freqs_block = freqs_block_t.numpy().astype(np.float32)

        out = np.zeros((10, self.trial_T_tokens, self.input_dim), dtype=np.float32)
        for tr in range(10):
            rng = np.random.default_rng(self._seed_for(idx, tr))
            out[tr] = self._render_trial_tokens_onehot(freqs_block[tr], rng=rng)

        x = torch.from_numpy(out.reshape(10 * self.trial_T_tokens, self.input_dim))
        gap_hz = compute_trial_gap_hz(freqs_block_t, y)
        if getattr(self, "assert_labels", False):
            if not torch.all((y == 4) | (y == 5) | (y == 6)):
                raise RuntimeError(f"[dataset assert] Bad y at idx={idx}: unique={torch.unique(y).tolist()}")
        return x, y, gap_hz


class TrialwiseRenderDataset(Dataset):
    """
    Debug/control dataset:
      - uses the exact same rendered representation as OnlineRenderDataset
      - but exposes each trial independently: x_trial (trial_T_tokens, D), y_trial scalar in {4,5,6}
    """

    def __init__(self, block_ds: OnlineRenderDataset):
        self.block_ds = block_ds
        self.B = int(block_ds.B)
        self.trials_per_block = 10
        self.N = self.B * self.trials_per_block
        self.trial_T_tokens = int(block_ds.trial_T_tokens)
        self.input_dim = int(block_ds.input_dim)
        self.Y = block_ds.Y

    def __len__(self) -> int:
        return self.N

    def set_epoch(self, epoch: int) -> None:
        self.block_ds.set_epoch(epoch)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bidx = int(idx // self.trials_per_block)
        tidx = int(idx % self.trials_per_block)
        x_block, y_block, _ = self.block_ds[bidx]  # x_block: (10*trial_T_tokens, D)
        x_trial = x_block.view(self.trials_per_block, self.trial_T_tokens, self.input_dim)[tidx]
        y_trial = y_block[tidx]
        return x_trial, y_trial


# -------------------------
# RT logic + token supervision mask
# -------------------------
def deviant_end_token_in_trial(
    y_pos_456: torch.Tensor,   # (B,10) values in {4,5,6}
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    dev_idx = (y_pos_456 - 1).long()  # {4,5,6}->{3,4,5}
    step = int(tone_T + isi_T)
    dev_end = dev_idx * step + int(tone_T) - 1
    return dev_end

def deviant_onset_token_in_trial(
    y_pos_456: torch.Tensor,   # (B,10) values in {4,5,6}
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    dev_idx = (y_pos_456 - 1).long()  # {4,5,6}->{3,4,5}
    step = int(tone_T + isi_T)
    dev_on = dev_idx * step
    return dev_on


def make_post_deviant_window_mask_tokens(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_ms: int,
    window_ms: int,
    start_offset_ms: int = 0,
    anchor: str = "deviant_end",  # "deviant_end" | "deviant_onset"
) -> torch.Tensor:
    trial_id = (abs_t // trial_T_tokens).long()
    within = (abs_t % trial_T_tokens).long()

    if anchor == "deviant_end":
        dev_anchor_bt = deviant_end_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    elif anchor == "deviant_onset":
        dev_anchor_bt = deviant_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    else:
        raise ValueError(f"Unknown anchor: {anchor}")
    dev_anchor_for_t = dev_anchor_bt[:, trial_id]

    offset_T = max(0, int(round(start_offset_ms / token_ms)))
    win_T = max(1, int(round(window_ms / token_ms)))

    lo = dev_anchor_for_t + offset_T
    hi = dev_anchor_for_t + offset_T + win_T
    return (within > lo) & (within <= hi)


def compute_online_decision_cost_from_logits(
    logits: torch.Tensor,            # (B,10,trial_T_tokens,3)
    y_cls: torch.Tensor,             # (B,10) in {0,1,2}
    y_pos_456: torch.Tensor,         # (B,10) in {4,5,6}
    tone_T: int,
    isi_T: int,
    rt_mode: str,
    rt_p_thresh: float,
    rt_entropy_thresh: float,
    rt_k_consec: int,
    min_rt_tokens: int,
    token_ms: int,
    decision_anchor: str = "deviant_onset",   # "trial_onset" | "deviant_onset" | "deviant_end"
    wrong_cost: float = 1.0,
    w_time: float = 0.001,
) -> Dict[str, float]:
    """
    Non-differentiable monitoring metric (NOT used for backprop):
      If wrong/miss => cost = wrong_cost
      If correct found at rt => cost = w_time * rt_ms
    """
    if decision_anchor == "trial_onset":
        dev_anchor = torch.zeros_like(y_pos_456, dtype=torch.long)
    elif decision_anchor == "deviant_onset":
        dev_anchor = deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
    elif decision_anchor == "deviant_end":
        dev_anchor = deviant_end_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
    else:
        raise ValueError(f"Unknown decision_anchor: {decision_anchor}")

    B, N, Tt, _ = logits.shape
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / np.log(3.0)
    pmax = probs.max(dim=-1).values

    K = int(max(1, rt_k_consec))
    min_rt_tokens = int(max(0, min_rt_tokens))
    total_cost = 0.0
    n_trials = int(B * N)
    n_found = 0
    n_correct_found = 0
    sum_rt_ms_correct = 0.0

    for b in range(B):
        for tr in range(N):
            t0 = int(dev_anchor[b, tr].item()) + 1
            if t0 >= Tt:
                total_cost += float(wrong_cost)
                continue
            search_start = t0 + min_rt_tokens
            if search_start >= Tt:
                total_cost += float(wrong_cost)
                continue

            decided = False
            for t in range(search_start, Tt - K + 1):
                cls0 = int(pred[b, tr, t].item())
                stable = True
                for j in range(1, K):
                    if int(pred[b, tr, t + j].item()) != cls0:
                        stable = False
                        break
                if not stable:
                    continue

                if rt_mode == "entropy":
                    ok = True
                    for j in range(K):
                        if float(ent[b, tr, t + j].item()) > float(rt_entropy_thresh):
                            ok = False
                            break
                else:  # confidence / oracle use p-threshold gate for monitoring
                    ok = True
                    for j in range(K):
                        if float(pmax[b, tr, t + j].item()) < float(rt_p_thresh):
                            ok = False
                            break

                if not ok:
                    continue

                decided = True
                n_found += 1
                is_correct = (cls0 == int(y_cls[b, tr].item()))
                if is_correct:
                    n_correct_found += 1
                    rt_tokens = (t - t0) + 1
                    rt_ms = float(rt_tokens * int(token_ms))
                    sum_rt_ms_correct += rt_ms
                    total_cost += float(w_time) * rt_ms
                else:
                    total_cost += float(wrong_cost)
                break

            if not decided:
                total_cost += float(wrong_cost)

    mean_cost = total_cost / max(1, n_trials)
    mean_rt_ms_correct = (sum_rt_ms_correct / max(1, n_correct_found)) if n_correct_found > 0 else float("nan")
    return {
        "online_decision_cost": float(mean_cost),
        "online_decision_found": int(n_found),
        "online_decision_correct_found": int(n_correct_found),
        "online_decision_mean_rt_ms_correct": float(mean_rt_ms_correct),
    }


def compute_anti_early_commit_loss(
    logits: torch.Tensor,        # (B, L, 3)
    mask: torch.Tensor,          # (B, L) bool
    max_conf: float,
) -> Tuple[torch.Tensor, float]:
    """
    Penalize over-confident predictions immediately after deviant onset so the
    model has to accumulate evidence instead of collapsing to a decision.
    """
    if (not mask.any()) or float(max_conf) >= 1.0:
        return logits.new_tensor(0.0), 0.0

    probs = torch.softmax(logits, dim=-1)
    pmax = probs.max(dim=-1).values
    penalty = F.relu(pmax[mask] - float(max_conf))
    return penalty.mean(), float(penalty.numel())


def compute_rt_from_logits(
    logits: torch.Tensor,          # (B,10,trial_T_tokens,3)
    y_cls: torch.Tensor,           # (B,10) in {0,1,2}; kept for downstream correctness check if needed
    dev_end: torch.Tensor,         # (B,10)
    p_thresh: float,
    k_consec: int,
    rt_mode: str = "entropy",      # "entropy" | "confidence" | "oracle"
    entropy_thresh: float = 0.35,  # normalized entropy threshold in [0,1]
    min_rt_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Literature-style RT readout.

    Returns
    -------
    rt : (B,N)
        1-based RT in tokens counted from the first token after deviant end.
    found : (B,N)
        Whether a decision was reached.
    pred_at_rt : (B,N)
        Predicted class at decision time; -1 if not found.
    """
    y_cls = y_cls.long()
    B, N, Tt, C = logits.shape
    assert N == 10 and C == 3

    probs = torch.softmax(logits, dim=-1)         # (B,N,Tt,C)
    pred = probs.argmax(dim=-1)                   # (B,N,Tt)

    # normalized entropy in [0,1]
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / np.log(C)  # (B,N,Tt)

    # max confidence
    pmax = probs.max(dim=-1).values               # (B,N,Tt)

    rt = torch.full((B, N), -1, dtype=torch.long, device=logits.device)
    found = torch.zeros((B, N), dtype=torch.bool, device=logits.device)
    pred_at_rt = torch.full((B, N), -1, dtype=torch.long, device=logits.device)

    K = int(max(1, k_consec))
    min_rt_tokens = int(max(0, min_rt_tokens))

    for b in range(B):
        for tr in range(N):
            t0 = int(dev_end[b, tr].item()) + 1
            if t0 >= Tt:
                continue

            search_start = t0 + min_rt_tokens
            if search_start >= Tt:
                continue

            for t in range(search_start, Tt - K + 1):
                cls0 = int(pred[b, tr, t].item())

                # require stable predicted class across K consecutive tokens
                stable = True
                for j in range(1, K):
                    if int(pred[b, tr, t + j].item()) != cls0:
                        stable = False
                        break
                if not stable:
                    continue

                if rt_mode == "entropy":
                    ok = True
                    for j in range(K):
                        if float(ent[b, tr, t + j].item()) > float(entropy_thresh):
                            ok = False
                            break

                elif rt_mode == "confidence":
                    ok = True
                    for j in range(K):
                        if float(pmax[b, tr, t + j].item()) < float(p_thresh):
                            ok = False
                            break

                elif rt_mode == "oracle":
                    # old behavior but with stability and min_rt_tokens
                    ok = True
                    for j in range(K):
                        cls_j = int(pred[b, tr, t + j].item())
                        if cls_j != int(y_cls[b, tr].item()):
                            ok = False
                            break
                        if float(probs[b, tr, t + j, cls_j].item()) < float(p_thresh):
                            ok = False
                            break
                else:
                    raise ValueError(f"Unknown rt_mode: {rt_mode}")

                if ok:
                    rt[b, tr] = (t - t0) + 1   # still 1-based from first token after deviant
                    found[b, tr] = True
                    pred_at_rt[b, tr] = cls0
                    break

    return rt, found, pred_at_rt


def make_online_eligibility_mask_tokens(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    start_mode: str = "deviant_start",   # trial_start | deviant_start | deviant_onset | deviant_end
    end_mode: str = "trial_end",         # trial_end | stimulus_end | sequence_end
) -> torch.Tensor:
    trial_id = (abs_t // trial_T_tokens).long()
    within = (abs_t % trial_T_tokens).long()
    B = int(y_pos_456.shape[0])
    Tchunk = int(abs_t.numel())

    if start_mode == "trial_start":
        lo = torch.zeros((B, Tchunk), dtype=torch.long, device=abs_t.device)
    elif start_mode in ("deviant_start", "deviant_onset"):
        lo = deviant_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)[:, trial_id]
    elif start_mode == "deviant_end":
        lo = deviant_end_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)[:, trial_id]
    else:
        raise ValueError(f"Unknown online_supervision_start: {start_mode}")

    if end_mode in ("trial_end", "stimulus_end", "sequence_end"):
        hi = torch.full((B, Tchunk), trial_T_tokens - 1, dtype=torch.long, device=abs_t.device)
    else:
        raise ValueError(f"Unknown online_supervision_end: {end_mode}")

    return (within >= lo) & (within <= hi)


def get_optimal_hazard_prior(y_pos_456: torch.Tensor, trial_T_tokens: int, tone_T: int, isi_T: int) -> torch.Tensor:
    """
    Returns log-prior tensor shaped (B,10,trial_T_tokens,3) with hazard-aware values:
      before P4 onset:   [1/3, 1/3, 1/3]
      P4..before P5:     [1/3, 1/2, 1]
      P5..before P6:     [0,   1/2, 1]
      >= P6 onset:       [0,   0,   1]
    """
    B, N = y_pos_456.shape
    device = y_pos_456.device
    Tt = int(trial_T_tokens)
    prior = torch.full((B, N, Tt, 3), 1.0 / 3.0, device=device)

    p4 = (4 - 1) * (tone_T + isi_T)
    p5 = (5 - 1) * (tone_T + isi_T)
    p6 = (6 - 1) * (tone_T + isi_T)

    if p4 < Tt:
        prior[:, :, p4:, 0] = 1.0 / 3.0
        prior[:, :, p4:, 1] = 1.0 / 2.0
        prior[:, :, p4:, 2] = 1.0
    if p5 < Tt:
        prior[:, :, p5:, 0] = 1e-6
        prior[:, :, p5:, 1] = 1.0 / 2.0
        prior[:, :, p5:, 2] = 1.0
    if p6 < Tt:
        prior[:, :, p6:, 0] = 1e-6
        prior[:, :, p6:, 1] = 1e-6
        prior[:, :, p6:, 2] = 1.0

    return torch.log(prior.clamp_min(1e-6))


def _decision_reference_tokens(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
    reference: str = "trial_onset",  # trial_onset | deviant_start | deviant_onset | deviant_end
) -> torch.Tensor:
    if reference == "trial_onset":
        return torch.zeros_like(y_pos_456, dtype=torch.long)
    if reference in ("deviant_start", "deviant_onset"):
        return deviant_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    if reference == "deviant_end":
        return deviant_end_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    raise ValueError(f"Unknown decision_reference: {reference}")


def resolve_time_cost_w(
    time_cost_w: Optional[float],
    derive_time_cost_from_timeout: bool,
    timeout_ms: Optional[float],
    default_w: float = 0.001,
) -> float:
    if time_cost_w is not None:
        return float(time_cost_w)
    if bool(derive_time_cost_from_timeout):
        if timeout_ms is None or float(timeout_ms) <= 0:
            raise ValueError("derive_time_cost_from_timeout=True requires positive timeout_ms.")
        return 1.0 / float(timeout_ms)
    return float(default_w)


def compute_expected_cost_softmin_loss(
    class_logits_trial: torch.Tensor,  # (B,10,T,3)
    y_cls: torch.Tensor,               # (B,10)
    y_pos_456: torch.Tensor,           # (B,10)
    eligible_mask_trial: torch.Tensor, # (B,10,T)
    token_ms: int,
    tone_T: int,
    isi_T: int,
    decision_reference: str = "trial_onset",
    timeout_ms: Optional[float] = None,
    time_cost_w: Optional[float] = 0.001,
    derive_time_cost_from_timeout: bool = False,
    softmin_tau: float = 0.05,
    sampling_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, N, Tt, _ = class_logits_trial.shape
    probs = torch.softmax(class_logits_trial / max(1e-6, float(sampling_temperature)), dim=-1)
    p_correct = probs.gather(dim=-1, index=y_cls.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Tt, 1)).squeeze(-1)
    ref_tok = _decision_reference_tokens(y_pos_456, tone_T=tone_T, isi_T=isi_T, reference=decision_reference)
    tgrid = torch.arange(Tt, device=class_logits_trial.device).view(1, 1, Tt)
    rt_tok = (tgrid - ref_tok.unsqueeze(-1)).clamp(min=0).float()
    rt_ms = rt_tok * float(token_ms)
    w = resolve_time_cost_w(
        time_cost_w=time_cost_w,
        derive_time_cost_from_timeout=bool(derive_time_cost_from_timeout),
        timeout_ms=timeout_ms,
        default_w=0.001,
    )
    expected_cost_t = (1.0 - p_correct) + (p_correct * (w * rt_ms))

    mask = eligible_mask_trial.bool()
    costs = expected_cost_t.masked_fill(~mask, float("inf"))
    tau = max(1e-6, float(softmin_tau))
    valid = mask.any(dim=-1)
    logits = (-costs / tau).masked_fill(~mask, -1e9)
    attn = torch.softmax(logits, dim=-1)
    softmin = (attn * costs.masked_fill(~mask, 0.0)).sum(dim=-1)
    trial_loss = torch.where(valid, softmin, torch.ones_like(softmin))
    return trial_loss.mean(), {
        "mean_expected_decision_time_ms": float((attn * rt_ms).sum(dim=-1)[valid].mean().item()) if valid.any() else float("nan"),
        "mean_correct_prob": float(p_correct[mask].mean().item()) if mask.any() else float("nan"),
        "posterior_entropy_mean": float((-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1))[mask].mean().item()) if mask.any() else float("nan"),
    }


def compute_stochastic_expected_cost_loss(
    class_logits_trial: torch.Tensor,          # (B,10,T,3)
    stop_logits_trial: Optional[torch.Tensor], # (B,10,T,1) or None
    y_cls: torch.Tensor,                       # (B,10)
    y_pos_456: torch.Tensor,                   # (B,10)
    eligible_mask_trial: torch.Tensor,         # (B,10,T)
    token_ms: int,
    tone_T: int,
    isi_T: int,
    decision_reference: str = "trial_onset",
    rt_logging_reference: Optional[str] = None,
    cost_reference: Optional[str] = None,
    clamp_negative_cost_time: bool = True,
    timeout_ms: Optional[float] = None,
    time_cost_w: Optional[float] = 0.001,
    derive_time_cost_from_timeout: bool = False,
    sampling_temperature: float = 1.0,
    stop_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, N, Tt, _ = class_logits_trial.shape
    probs = torch.softmax(class_logits_trial / max(1e-6, float(sampling_temperature)), dim=-1)
    p_correct = probs.gather(dim=-1, index=y_cls.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Tt, 1)).squeeze(-1)
    if stop_logits_trial is None:
        # fallback: map confidence to stop probability
        p_stop = probs.max(dim=-1).values.clamp(0.01, 0.99)
    else:
        p_stop = torch.sigmoid(stop_logits_trial.squeeze(-1) / max(1e-6, float(stop_temperature))).clamp(1e-4, 1 - 1e-4)

    mask = eligible_mask_trial.bool()
    p_stop = torch.where(mask, p_stop, torch.zeros_like(p_stop))

    tgrid = torch.arange(Tt, device=class_logits_trial.device).view(1, 1, Tt)
    decision_time_ms = tgrid.float() * float(token_ms)
    rt_ref_name = str(rt_logging_reference if rt_logging_reference is not None else decision_reference)
    cost_ref_name = str(cost_reference if cost_reference is not None else decision_reference)
    rt_ref_tok = _decision_reference_tokens(y_pos_456, tone_T=tone_T, isi_T=isi_T, reference=rt_ref_name)
    cost_ref_tok = _decision_reference_tokens(y_pos_456, tone_T=tone_T, isi_T=isi_T, reference=cost_ref_name)
    logged_rt_ms = decision_time_ms - (rt_ref_tok.unsqueeze(-1).float() * float(token_ms))
    raw_cost_time_ms = decision_time_ms - (cost_ref_tok.unsqueeze(-1).float() * float(token_ms))
    cost_time_ms = raw_cost_time_ms.clamp(min=0.0) if bool(clamp_negative_cost_time) else raw_cost_time_ms
    w = resolve_time_cost_w(
        time_cost_w=time_cost_w,
        derive_time_cost_from_timeout=bool(derive_time_cost_from_timeout),
        timeout_ms=timeout_ms,
        default_w=0.001,
    )
    decision_cost_t = ((1.0 - p_correct) * 1.0) + (p_correct * (w * cost_time_ms))

    q = (1.0 - p_stop).clamp(1e-6, 1.0)
    prev_survive = torch.cumprod(q, dim=-1)
    prev_survive = torch.cat([torch.ones_like(prev_survive[..., :1]), prev_survive[..., :-1]], dim=-1)
    p_first_stop = (p_stop * prev_survive) * mask.float()
    p_no_response = torch.where(mask, q, torch.ones_like(q))
    p_no_response = torch.cumprod(p_no_response, dim=-1)[..., -1]

    expected = (p_first_stop * decision_cost_t).sum(dim=-1) + p_no_response * 1.0
    loss = expected.mean()
    expected_found_prob = 1.0 - p_no_response
    expected_rt_logged_ms = (p_first_stop * logged_rt_ms).sum(dim=-1)
    expected_rt_from_trial_onset_ms = (p_first_stop * decision_time_ms).sum(dim=-1)
    dev_on_ms = deviant_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T).unsqueeze(-1).float() * float(token_ms)
    dev_end_ms = deviant_end_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T).unsqueeze(-1).float() * float(token_ms)
    expected_rt_from_deviant_onset_ms = (p_first_stop * (decision_time_ms - dev_on_ms)).sum(dim=-1)
    expected_rt_from_deviant_end_ms = (p_first_stop * (decision_time_ms - dev_end_ms)).sum(dim=-1)
    proportion_negative_rt = (p_first_stop * (logged_rt_ms < 0).float()).sum(dim=-1)
    proportion_decisions_before_deviant_end = (p_first_stop * (decision_time_ms < dev_end_ms).float()).sum(dim=-1)
    return loss, {
        "mean_p_stop": float(p_stop[mask].mean().item()) if mask.any() else float("nan"),
        "mean_no_response_prob": float(p_no_response.mean().item()),
        "mean_expected_decision_time_ms": float(expected_rt_from_trial_onset_ms.mean().item()),
        "expected_found_prob": float(expected_found_prob.mean().item()),
        "expected_rt_logged_ms": float(expected_rt_logged_ms.mean().item()),
        "expected_rt_from_trial_onset_ms": float(expected_rt_from_trial_onset_ms.mean().item()),
        "expected_rt_from_deviant_onset_ms": float(expected_rt_from_deviant_onset_ms.mean().item()),
        "expected_rt_from_deviant_end_ms": float(expected_rt_from_deviant_end_ms.mean().item()),
        "proportion_negative_rt": float(proportion_negative_rt.mean().item()),
        "proportion_decisions_before_deviant_end": float(proportion_decisions_before_deviant_end.mean().item()),
        "stochastic_expected_cost": float(loss.item()),
        "class_entropy_mean": float((-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1))[mask].mean().item()) if mask.any() else float("nan"),
        "stop_entropy_mean": float((-(p_stop * torch.log(p_stop.clamp_min(1e-12)) + (1-p_stop)*torch.log((1-p_stop).clamp_min(1e-12))))[mask].mean().item()) if mask.any() else float("nan"),
    }

# -------------------------
# Token-loss helpers (MODIFIED for mask support)
# -------------------------
def _weighted_token_ce(
    logits: torch.Tensor,   # (N,3)
    target: torch.Tensor,   # (N,)
    weights: torch.Tensor,  # (N,)
) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, reduction="none")
    w = weights.float().clamp(min=0.0)
    return (loss * w).sum() / (w.sum() + 1e-8)


def compute_masked_token_loss(
    logits: torch.Tensor,        # (B, L, 3)
    target: torch.Tensor,        # (B, L) in {0,1,2}
    mask: torch.Tensor,          # (B, L) bool, True表示有意义token
    reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算掩码损失：只计算mask=True的位置
    Returns:
        loss: 标量损失值
        loss_per_token: (B, L) 每个token的损失
    """
    # 计算所有位置的损失
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, 3), 
        target.reshape(-1), 
        reduction="none"
    ).reshape(target.shape)
    
    # 只在有意义的位置计算损失
    if reduction == "mean":
        loss = (loss_per_token * mask.float()).sum() / (mask.float().sum() + 1e-8)
    else:  # sum
        loss = (loss_per_token * mask.float()).sum()
    
    return loss, loss_per_token


def _fmt_hms(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "NA"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{ss:02d}"
    return f"{m:02d}:{ss:02d}"


# -------------------------
# TBPTT runner (MODIFIED with mask support)
# -------------------------
def _run_block_through_tbptt(
    model: PredictiveGRU,
    x: torch.Tensor,              # (B,T,D)
    y_pos_456: torch.Tensor,      # (B,10) in {4,5,6}
    trial_weights_bt: Optional[torch.Tensor],  # (B,10) or None
    end_idx: torch.Tensor,        # (10,)
    chunk_len: int,
    token_ce: nn.Module,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_ms: int,
    tok_window_ms: int,
    tok_start_offset_ms: int,
    token_loss_mode: str = "uniform",  # "uniform" or "exp"
    token_tau: float = 50.0,
    token_w_min: float = 0.05,
    tok_anchor: str = "deviant_end",  # "deviant_end" | "deviant_onset"
    anti_commit_window_ms: int = 0,
    anti_commit_start_offset_ms: int = 0,
    anti_commit_max_conf: float = 1.0,
    return_full_logits: bool = False,
    return_stop_logits: bool = False,
    use_mask_loss: bool = False,      # 新增：是否使用掩码损失
    important_token_indices: Optional[List[int]] = None,  # 有意义token的位置
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns:
        logits_end: (B,10,3)
        token_loss: scalar
        anti_commit_loss: scalar
        logits_all: (B,T,3) or None
        token_mask: (B,T) bool or None  # 新增：返回mask用于分析
    """
    if token_loss_mode not in ("uniform", "exp"):
        raise ValueError("token_loss_mode must be 'uniform' or 'exp'")

    B, T, D = x.shape
    h = None
    collected_end_logits = []
    token_loss_sum = x.new_tensor(0.0)
    token_count = x.new_tensor(0.0)
    anti_commit_loss_sum = x.new_tensor(0.0)
    anti_commit_count = x.new_tensor(0.0)

    logits_chunks = [] if return_full_logits else None
    stop_logits_chunks = [] if (return_full_logits and return_stop_logits) else None
    mask_chunks = [] if return_full_logits else None  # 收集mask

    # 创建全局token mask (如果有意义token位置指定)
    if use_mask_loss and important_token_indices is not None:
        # 每个trial有trial_T_tokens个token，其中只有少数位置是有意义的
        # 这里简化：假设有意义token是每个trial中deviant后的某个窗口
        # 实际应用中，你需要根据你的任务定义哪些token位置是有意义的
        global_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        
        # 示例：标记每个trial中deviant后的前10个token为有意义
        # 你需要根据实际任务调整这个逻辑
        for trial in range(10):
            trial_start = trial * trial_T_tokens
            for pos in important_token_indices:
                if pos < trial_T_tokens:  # 确保不越界
                    global_mask[:, trial_start + pos] = True
    else:
        global_mask = None

    for s in range(0, T, int(chunk_len)):
        e = min(s + int(chunk_len), T)
        x_in = x[:, s:e, :]
        h_seq, h = model.forward_chunk(x_in, h0=h)           # (B,L,H)
        logits = model.classify_tokens(h_seq)                # (B,L,3)
        stop_logits = model.classify_stop(h_seq) if (return_full_logits and return_stop_logits) else None

        if return_full_logits:
            logits_chunks.append(logits.detach())
            if stop_logits_chunks is not None:
                if stop_logits is None:
                    stop_logits_chunks.append(torch.full((B, e - s, 1), float("nan"), device=logits.device))
                else:
                    stop_logits_chunks.append(stop_logits.detach())
            if global_mask is not None:
                mask_chunks.append(global_mask[:, s:e])

        abs_t = torch.arange(s, e, device=x.device)
        trial_id = (abs_t // int(trial_T_tokens)).long()
        within = (abs_t % int(trial_T_tokens)).long()

        # 创建mask（用于token损失）
        if use_mask_loss and global_mask is not None:
            # 使用全局预计算的mask
            mask = global_mask[:, s:e]
        elif int(tok_window_ms) > 0:
            # 原有的窗口mask逻辑
            mask = make_post_deviant_window_mask_tokens(
                abs_t=abs_t,
                y_pos_456=y_pos_456,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
                token_ms=int(token_ms),
                window_ms=int(tok_window_ms),
                start_offset_ms=int(tok_start_offset_ms),
                anchor=str(tok_anchor),
            )
        else:
            if str(tok_anchor) == "deviant_end":
                dev_anchor_bt = deviant_end_token_in_trial(y_pos_456, tone_T=int(tone_T), isi_T=int(isi_T))
            elif str(tok_anchor) == "deviant_onset":
                dev_anchor_bt = deviant_onset_token_in_trial(y_pos_456, tone_T=int(tone_T), isi_T=int(isi_T))
            else:
                raise ValueError(f"Unknown tok_anchor: {tok_anchor}")
            dev_anchor_for_t = dev_anchor_bt[:, trial_id]
            mask = (within > dev_anchor_for_t)

        y_cls = labels_to_class_index(y_pos_456)
        target = y_cls[:, trial_id]

        if int(anti_commit_window_ms) > 0 and float(anti_commit_max_conf) < 1.0:
            anti_mask = make_post_deviant_window_mask_tokens(
                abs_t=abs_t,
                y_pos_456=y_pos_456,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
                token_ms=int(token_ms),
                window_ms=int(anti_commit_window_ms),
                start_offset_ms=int(anti_commit_start_offset_ms),
            )
            if anti_mask.any():
                anti_loss_mean, n_anti = compute_anti_early_commit_loss(
                    logits=logits,
                    mask=anti_mask,
                    max_conf=float(anti_commit_max_conf),
                )
                anti_commit_loss_sum = anti_commit_loss_sum + anti_loss_mean * n_anti
                anti_commit_count = anti_commit_count + n_anti

        if mask.any():
            assert int(target.min().item()) >= 0
            assert int(target.max().item()) <= 2
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, 3),
                target.reshape(-1),
                reduction="none",
            ).reshape(target.shape)
            loss_weights = mask.float()
            if trial_weights_bt is not None:
                loss_weights = loss_weights * trial_weights_bt[:, trial_id].float()

            if use_mask_loss:
                token_loss_sum = token_loss_sum + (loss_per_token * loss_weights).sum()
                token_count = token_count + loss_weights.sum()
            elif token_loss_mode == "uniform":
                token_loss_sum = token_loss_sum + (loss_per_token * loss_weights).sum()
                token_count = token_count + loss_weights.sum()
            else:  # exp
                dev_end_bt = deviant_end_token_in_trial(y_pos_456, tone_T=int(tone_T), isi_T=int(isi_T))
                dev_end_for_t = dev_end_bt[:, trial_id]

                offset_T = max(0, int(round(int(tok_start_offset_ms) / int(token_ms))))
                anchor = dev_end_for_t + 1 + offset_T

                dist = (within.unsqueeze(0) - anchor).clamp(min=0)
                w = torch.exp(-dist.float() / float(token_tau))
                w = torch.clamp(w, min=float(token_w_min), max=1.0)
                loss_weights = loss_weights * w
                token_loss_sum = token_loss_sum + (loss_per_token * loss_weights).sum()
                token_count = token_count + loss_weights.sum()

        m_end = (end_idx >= s) & (end_idx < e)
        if m_end.any():
            rel = (end_idx[m_end] - s).long()
            h_end = h_seq.index_select(dim=1, index=rel)
            collected_end_logits.append(model.classify_from_states(h_end))

        h = (h[0].detach(), h[1].detach()) if isinstance(h, tuple) else h.detach()

    if len(collected_end_logits) == 0:
        raise RuntimeError("No trial-end states collected. Check end_idx and T.")

    logits_end = torch.cat(collected_end_logits, dim=1)
    token_loss = token_loss_sum / (token_count + 1e-8)
    anti_commit_loss = anti_commit_loss_sum / (anti_commit_count + 1e-8)

    logits_all = None
    full_mask = None
    if return_full_logits:
        logits_all = torch.cat(logits_chunks, dim=1)
        if mask_chunks:
            full_mask = torch.cat(mask_chunks, dim=1)

    stop_logits_all = None
    if stop_logits_chunks is not None:
        stop_logits_all = torch.cat(stop_logits_chunks, dim=1)

    return logits_end, token_loss, anti_commit_loss, logits_all, full_mask, stop_logits_all


# -------------------------
# Metrics (acc, F1, AUC)
# -------------------------
def _collect_classification_metrics_from_logits(
    logits_end: torch.Tensor,   # (B,10,3)
    y_cls: torch.Tensor,        # (B,10)
) -> Dict[str, float]:
    with torch.no_grad():
        probs = torch.softmax(logits_end, dim=-1)  # (B,10,3)
        pred = probs.argmax(dim=-1)               # (B,10)
        acc = (pred == y_cls).float().mean().item()

        y_true = y_cls.reshape(-1).detach().cpu().numpy()
        y_pred = pred.reshape(-1).detach().cpu().numpy()
        y_prob = probs.reshape(-1, 3).detach().cpu().numpy()

        f1 = safe_f1_macro(y_true, y_pred)
        auc = safe_auc_ovr(y_true, y_prob, n_classes=3)

    return {"acc": float(acc), "f1_macro": float(f1), "auc_ovr": float(auc)}


# -------------------------
# Train / Eval (MODIFIED with mask support)
# -------------------------
def train_one_epoch(
    model: PredictiveGRU,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    chunk_len: int,
    lambda_token: float,
    grad_clip: float,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_ms: int = 10,
    tok_window_ms: int = 0,
    tok_start_offset_ms: int = 0,
    lambda_anti_commit: float = 0.0,
    anti_commit_window_ms: int = 0,
    anti_commit_start_offset_ms: int = 0,
    anti_commit_max_conf: float = 1.0,
    rt_p_thresh: float = 0.7,
    rt_k_consec: int = 3,
    rt_mode: str = "entropy",
    rt_entropy_thresh: float = 0.35,
    min_rt_tokens: int = 0,
    debug: bool = False,
    debug_steps: int = 0,
    log_every: int = 50,
    token_loss_mode: str = "uniform",
    token_tau: float = 50.0,
    token_w_min: float = 0.05,
    tok_anchor: str = "deviant_end",
    debug_labels: bool = False,
    debug_labels_fatal: bool = False,
    debug_labels_dump: bool = False,
    debug_labels_dump_dir: Optional[Path] = None,
    debug_labels_max_dumps: int = 10,
    debug_labels_first_n_batches: int = 0,
    epoch_global: int = 0,
    use_mask_loss: bool = False,
    important_token_indices: Optional[List[int]] = None,
    gap_weight_power: float = 0.0,
    gap_weight_ref_hz: float = 25.0,
    gap_weight_max: float = 2.5,
    online_cost_monitor: bool = False,
    online_cost_anchor: str = "deviant_onset",
    online_cost_wrong: float = 1.0,
    online_cost_w_time: float = 0.001,
    online_decision_training: bool = False,
    online_loss_weight: float = 0.5,
    online_ce_weight: float = 0.1,
    decision_cost_mode: str = "expected_cost_softmin",
    timeout_ms: Optional[float] = None,
    time_cost_w: Optional[float] = 0.001,
    derive_time_cost_from_timeout: bool = False,
    decision_reference: str = "trial_onset",
    rt_logging_reference: str = "deviant_end",
    cost_reference: str = "deviant_onset",
    clamp_negative_cost_time: bool = True,
    online_supervision_start: str = "deviant_start",
    online_supervision_end: str = "trial_end",
    aux_token_ce_weight: float = 0.0,
    aux_token_ce_start: str = "deviant_onset",
    aux_token_ce_end: str = "trial_end",
    decision_softmin_tau: float = 0.05,
    use_stop_head: bool = False,
    decision_policy: str = "deterministic_threshold",
    sampling_temperature: float = 1.0,
    stop_temperature: float = 1.0,
    use_hazard_prior: bool = False,
    hazard_prior_weight: float = 1.0,
    hazard_prior_mode: str = "add_log_prior",
    online_warmup_factor: float = 1.0,
    phase_name: str = "single_phase",
    include_online_decision_loss: bool = True,
    include_online_ce_loss: bool = True,
    anti_immediate_stop: bool = False,
    anti_immediate_stop_tokens: int = 5,
    anti_immediate_stop_weight: float = 0.1,
    pre_devend_stop_weight: float = 0.0,
    stop_entropy_weight: float = 0.0,
    stop_prior_weight: float = 0.0,
    stop_prior_target: float = 0.05,

    debug_loss_check: bool = False,
    debug_overfit_tiny: bool = False,
    debug_first_batch_done: bool = False,
    optimizer_step_state: Optional[Dict[str, int]] = None,
    max_optimizer_steps: Optional[int] = None,
) -> dict:
    model.train()
    ce_end = nn.CrossEntropyLoss(reduction="mean")
    ce_tok = nn.CrossEntropyLoss(reduction="mean")

    total_end = 0.0
    total_tok = 0.0
    total_anti = 0.0
    n_examples = 0

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []

    rt_tokens_sum = 0.0
    rt_ms_sum = 0.0
    rt_n = 0
    rt_miss = 0
    rt_not_first = 0
    online_cost_sum = 0.0
    online_cost_n = 0
    online_loss_sum = 0.0
    online_ce_sum = 0.0
    mean_p_stop_sum = 0.0
    mean_p_stop_n = 0
    mean_noresp_sum = 0.0
    mean_noresp_n = 0
    mean_expdt_sum = 0.0
    mean_expdt_n = 0
    expected_found_prob_sum = 0.0
    expected_found_prob_n = 0
    expected_rt_logged_sum = 0.0
    expected_rt_logged_n = 0
    expected_rt_from_devon_sum = 0.0
    expected_rt_from_devon_n = 0
    expected_rt_from_devend_sum = 0.0
    expected_rt_from_devend_n = 0
    prop_negative_rt_sum = 0.0
    prop_negative_rt_n = 0
    prop_before_devend_sum = 0.0
    prop_before_devend_n = 0
    anti_immediate_stop_sum = 0.0
    stop_entropy_sum = 0.0
    stop_prior_sum = 0.0
    aux_token_ce_sum = 0.0
    weighted_aux_token_ce_sum = 0.0
    last_token_acc_sum = 0.0
    last_token_acc_n = 0
    devend_acc_sum = 0.0
    devend_acc_n = 0

    n_steps = len(loader)
    t_epoch0 = time.time()
    ema_step = None
    ema_beta = 0.9

    step = 0
    first_batch_debug_printed = bool(debug_first_batch_done)
    for batch in loader:
        x, y, gap_hz = unpack_batch_with_optional_gap(batch)
        if debug_labels_first_n_batches and step <= int(debug_labels_first_n_batches) and debug_labels:
            print(f"[debug_labels] train batch {step} y(unique)={_unique_list(y)}")

        _check_y_and_maybe_debug(
            y_cpu=y,
            where="train/batch_received",
            debug_on=debug_labels,
            fatal=debug_labels_fatal,
            dump_on=debug_labels_dump,
            dump_dir=debug_labels_dump_dir,
            dump_tag=f"train_ep{epoch_global:04d}_step{step:04d}_received",
            dump_info={"epoch_global": epoch_global, "step": step},
            dump_tensors=None,
            max_dumps=debug_labels_max_dumps,
        )
        step += 1
        t0 = time.time()

        x = x.to(device, non_blocking=True)
        y_cpu = y.long()
        y = y_cpu.to(device, non_blocking=True)
        gap_weights_bt = make_gap_weight_tensor(
            gap_hz_bt=(gap_hz.to(device, non_blocking=True) if gap_hz is not None else None),
            power=float(gap_weight_power),
            ref_hz=float(gap_weight_ref_hz),
            max_weight=float(gap_weight_max),
        )

        if debug_labels:
            y_back = y.detach().to("cpu").long()
            _check_y_and_maybe_debug(
                y_cpu=y_back,
                where="train/after_to_device",
                debug_on=True,
                fatal=debug_labels_fatal,
                dump_on=debug_labels_dump,
                dump_dir=debug_labels_dump_dir,
                dump_tag=f"train_ep{epoch_global:04d}_step{step:04d}_after_to_device",
                dump_info={"epoch_global": epoch_global, "step": step, "device": str(device)},
                dump_tensors=None,
                max_dumps=debug_labels_max_dumps,
            )

        B, T, D = x.shape
        n_examples += B

        end_idx = infer_end_indices_from_T(T, trials_per_block=10).to(device)

        optimizer.zero_grad(set_to_none=True)

        w0 = None
        if debug and step <= int(debug_steps):
            w = next(model.parameters())
            w0 = w.detach().float().cpu().clone()

        logits_end, token_loss, anti_commit_loss, logits_all, token_mask, stop_logits_all = _run_block_through_tbptt(
            model=model,
            x=x,
            y_pos_456=y,
            trial_weights_bt=gap_weights_bt,
            end_idx=end_idx,
            chunk_len=int(chunk_len),
            token_ce=ce_tok,
            trial_T_tokens=int(trial_T_tokens),
            tone_T=int(tone_T),
            isi_T=int(isi_T),
            token_ms=int(token_ms),
            tok_window_ms=int(tok_window_ms),
            tok_start_offset_ms=int(tok_start_offset_ms),
            token_loss_mode=str(token_loss_mode),
            token_tau=float(token_tau),
            token_w_min=float(token_w_min),
            tok_anchor=str(tok_anchor),
            anti_commit_window_ms=int(anti_commit_window_ms),
            anti_commit_start_offset_ms=int(anti_commit_start_offset_ms),
            anti_commit_max_conf=float(anti_commit_max_conf),
            return_full_logits=True,
            return_stop_logits=bool(use_stop_head),
            use_mask_loss=use_mask_loss,
            important_token_indices=important_token_indices,
        )

        y_cls = labels_to_class_index(y)
        assert int(y_cls.min().item()) >= 0
        assert int(y_cls.max().item()) <= 2
        end_loss_per = F.cross_entropy(
            logits_end.reshape(-1, 3),
            y_cls.reshape(-1),
            reduction="none",
        ).reshape(y_cls.shape)
        if gap_weights_bt is None:
            end_loss = end_loss_per.mean()
        else:
            end_loss = (end_loss_per * gap_weights_bt).sum() / gap_weights_bt.sum().clamp_min(1e-8)

        total_loss = (
            end_loss
            + float(lambda_token) * token_loss
            + float(lambda_anti_commit) * anti_commit_loss
        )
        online_decision_loss = end_loss.new_tensor(0.0)
        online_ce_loss = end_loss.new_tensor(0.0)
        aux_token_ce_loss = end_loss.new_tensor(0.0)
        anti_immediate_stop_loss = end_loss.new_tensor(0.0)
        stop_entropy_mean = end_loss.new_tensor(0.0)
        stop_prior_loss = end_loss.new_tensor(0.0)
        if (bool(online_decision_training) or bool(include_online_ce_loss)) and (logits_all is not None):
            class_trial = logits_all.view(B, 10, int(trial_T_tokens), 3)
            if bool(use_hazard_prior) and str(hazard_prior_mode) == "add_log_prior":
                log_prior = get_optimal_hazard_prior(y, int(trial_T_tokens), int(tone_T), int(isi_T))
                class_trial = class_trial + float(hazard_prior_weight) * log_prior
            abs_t = torch.arange(0, int(T), device=x.device)
            elig = make_online_eligibility_mask_tokens(
                abs_t=abs_t,
                y_pos_456=y,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
                start_mode=str(online_supervision_start),
                end_mode=str(online_supervision_end),
            ).view(B, 10, int(trial_T_tokens))
            stop_trial = None
            if stop_logits_all is not None and torch.isfinite(stop_logits_all).any():
                stop_trial = stop_logits_all.view(B, 10, int(trial_T_tokens), 1)

            if str(decision_cost_mode) == "expected_cost_softmin":
                online_decision_loss, od_stats = compute_expected_cost_softmin_loss(
                    class_logits_trial=class_trial,
                    y_cls=y_cls,
                    y_pos_456=y,
                    eligible_mask_trial=elig,
                    token_ms=int(token_ms),
                    tone_T=int(tone_T),
                    isi_T=int(isi_T),
                    decision_reference=str(decision_reference),
                    timeout_ms=(None if timeout_ms is None else float(timeout_ms)),
                    time_cost_w=time_cost_w,
                    derive_time_cost_from_timeout=bool(derive_time_cost_from_timeout),
                    softmin_tau=float(decision_softmin_tau),
                    sampling_temperature=float(sampling_temperature),
                )
            elif str(decision_cost_mode) == "stochastic_expected_cost":
                online_decision_loss, od_stats = compute_stochastic_expected_cost_loss(
                    class_logits_trial=class_trial,
                    stop_logits_trial=stop_trial,
                    y_cls=y_cls,
                    y_pos_456=y,
                    eligible_mask_trial=elig,
                    token_ms=int(token_ms),
                    tone_T=int(tone_T),
                    isi_T=int(isi_T),
                    decision_reference=str(decision_reference),
                    rt_logging_reference=str(rt_logging_reference),
                    cost_reference=str(cost_reference),
                    clamp_negative_cost_time=bool(clamp_negative_cost_time),
                    timeout_ms=(None if timeout_ms is None else float(timeout_ms)),
                    time_cost_w=time_cost_w,
                    derive_time_cost_from_timeout=bool(derive_time_cost_from_timeout),
                    sampling_temperature=float(sampling_temperature),
                    stop_temperature=float(stop_temperature),
                )
            elif str(decision_cost_mode) == "masked_online_ce":
                tgt = y_cls.unsqueeze(-1).expand(-1, -1, int(trial_T_tokens))
                assert int(tgt.min().item()) >= 0
                assert int(tgt.max().item()) <= 2
                ce_tok = F.cross_entropy(class_trial.reshape(-1, 3), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
                online_decision_loss = (ce_tok * elig.float()).sum() / elig.float().sum().clamp_min(1e-8)
                od_stats = {}
            elif str(decision_cost_mode) == "policy_gradient":
                raise NotImplementedError("policy_gradient mode not implemented yet; use stochastic_expected_cost.")
            else:
                raise ValueError(f"Unknown decision_cost_mode: {decision_cost_mode}")

            tgt = y_cls.unsqueeze(-1).expand(-1, -1, int(trial_T_tokens))
            assert int(tgt.min().item()) >= 0
            assert int(tgt.max().item()) <= 2
            ce_tok = F.cross_entropy(class_trial.reshape(-1, 3), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
            online_ce_loss = (ce_tok * elig.float()).sum() / elig.float().sum().clamp_min(1e-8)
            if not torch.isfinite(online_ce_loss):
                online_ce_loss = end_loss.new_tensor(0.0)
            aux_mask = make_online_eligibility_mask_tokens(
                abs_t=abs_t,
                y_pos_456=y,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
                start_mode=str(aux_token_ce_start),
                end_mode=str(aux_token_ce_end),
            ).view(B, 10, int(trial_T_tokens))
            aux_token_ce_loss = (ce_tok * aux_mask.float()).sum() / aux_mask.float().sum().clamp_min(1e-8)
            if not torch.isfinite(aux_token_ce_loss):
                aux_token_ce_loss = end_loss.new_tensor(0.0)
            eff_w = float(online_loss_weight) * float(online_warmup_factor)
            if bool(include_online_decision_loss):
                total_loss = total_loss + eff_w * online_decision_loss
            if bool(include_online_ce_loss):
                total_loss = total_loss + float(online_ce_weight) * online_ce_loss
            if float(aux_token_ce_weight) != 0.0:
                total_loss = total_loss + float(aux_token_ce_weight) * aux_token_ce_loss

            # stop-head anti-collapse regularizers
            if stop_trial is not None:
                p_stop = torch.sigmoid(stop_trial.squeeze(-1) / max(1e-6, float(stop_temperature))).clamp(1e-6, 1 - 1e-6)
                if bool(anti_immediate_stop):
                    K = max(1, int(anti_immediate_stop_tokens))
                    early_mask = torch.zeros_like(elig, dtype=torch.bool)
                    for bidx in range(B):
                        for tridx in range(10):
                            idxs = torch.where(elig[bidx, tridx])[0]
                            if idxs.numel() > 0:
                                early = idxs[: min(K, int(idxs.numel()))]
                                early_mask[bidx, tridx, early] = True
                    if early_mask.any():
                        anti_immediate_stop_loss = p_stop[early_mask].mean()
                        total_loss = total_loss + float(anti_immediate_stop_weight) * anti_immediate_stop_loss
                        anti_immediate_stop_sum += float(anti_immediate_stop_loss.item()) * B
                if float(stop_entropy_weight) > 0.0:
                    stop_ent = -(p_stop * torch.log(p_stop) + (1 - p_stop) * torch.log(1 - p_stop))
                    stop_entropy_mean = stop_ent[elig].mean() if elig.any() else stop_ent.mean()
                    total_loss = total_loss - float(stop_entropy_weight) * stop_entropy_mean
                    stop_entropy_sum += float(stop_entropy_mean.item()) * B
                if float(stop_prior_weight) > 0.0:
                    mean_p = p_stop[elig].mean() if elig.any() else p_stop.mean()
                    stop_prior_loss = (mean_p - float(stop_prior_target)) ** 2
                    total_loss = total_loss + float(stop_prior_weight) * stop_prior_loss
                    stop_prior_sum += float(stop_prior_loss.item()) * B
                if float(pre_devend_stop_weight) > 0.0:
                    dev_end_tok = deviant_end_token_in_trial(y, tone_T=int(tone_T), isi_T=int(isi_T))
                    tgrid = torch.arange(int(trial_T_tokens), device=x.device).view(1, 1, -1)
                    before_devend_mask = (tgrid < dev_end_tok.unsqueeze(-1)) & elig
                    if before_devend_mask.any():
                        pre_devend_stop_loss = p_stop[before_devend_mask].mean()
                        total_loss = total_loss + float(pre_devend_stop_weight) * pre_devend_stop_loss
                        anti_immediate_stop_sum += float(pre_devend_stop_loss.item()) * B

            online_loss_sum += float(online_decision_loss.item()) * B

            online_ce_sum += float(online_ce_loss.item()) * B
            aux_token_ce_sum += float(aux_token_ce_loss.item()) * B
            weighted_aux_token_ce_sum += float(aux_token_ce_weight) * float(aux_token_ce_loss.item()) * B
            if "mean_p_stop" in od_stats and np.isfinite(od_stats["mean_p_stop"]):
                mean_p_stop_sum += float(od_stats["mean_p_stop"]) * B
                mean_p_stop_n += B
            if "mean_no_response_prob" in od_stats and np.isfinite(od_stats["mean_no_response_prob"]):
                mean_noresp_sum += float(od_stats["mean_no_response_prob"]) * B
                mean_noresp_n += B
            if "mean_expected_decision_time_ms" in od_stats and np.isfinite(od_stats["mean_expected_decision_time_ms"]):
                mean_expdt_sum += float(od_stats["mean_expected_decision_time_ms"]) * B
                mean_expdt_n += B
            if "expected_found_prob" in od_stats and np.isfinite(od_stats["expected_found_prob"]):
                expected_found_prob_sum += float(od_stats["expected_found_prob"]) * B
                expected_found_prob_n += B
            if "expected_rt_logged_ms" in od_stats and np.isfinite(od_stats["expected_rt_logged_ms"]):
                expected_rt_logged_sum += float(od_stats["expected_rt_logged_ms"]) * B
                expected_rt_logged_n += B
            if "expected_rt_from_deviant_onset_ms" in od_stats and np.isfinite(od_stats["expected_rt_from_deviant_onset_ms"]):
                expected_rt_from_devon_sum += float(od_stats["expected_rt_from_deviant_onset_ms"]) * B
                expected_rt_from_devon_n += B
            if "expected_rt_from_deviant_end_ms" in od_stats and np.isfinite(od_stats["expected_rt_from_deviant_end_ms"]):
                expected_rt_from_devend_sum += float(od_stats["expected_rt_from_deviant_end_ms"]) * B
                expected_rt_from_devend_n += B
            if "proportion_negative_rt" in od_stats and np.isfinite(od_stats["proportion_negative_rt"]):
                prop_negative_rt_sum += float(od_stats["proportion_negative_rt"]) * B
                prop_negative_rt_n += B
            if "proportion_decisions_before_deviant_end" in od_stats and np.isfinite(od_stats["proportion_decisions_before_deviant_end"]):
                prop_before_devend_sum += float(od_stats["proportion_decisions_before_deviant_end"]) * B
                prop_before_devend_n += B

            with torch.no_grad():
                last_pred = class_trial[:, :, -1, :].argmax(dim=-1)
                last_token_acc_sum += float((last_pred == y_cls).float().mean().item()) * B
                last_token_acc_n += B
                dev_end_tok = deviant_end_token_in_trial(y, tone_T=int(tone_T), isi_T=int(isi_T))
                gather_idx = dev_end_tok.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
                devend_logits = class_trial.gather(dim=2, index=gather_idx).squeeze(2)
                devend_pred = devend_logits.argmax(dim=-1)
                devend_acc_sum += float((devend_pred == y_cls).float().mean().item()) * B
                devend_acc_n += B

        if debug_overfit_tiny and not first_batch_debug_printed:
            print("[DEBUG first batch]")
            print("input shape", tuple(x.shape))
            print("labels shape", tuple(y.shape))
            print("unique raw labels", _unique_list(y.detach().cpu()))
            print("unique mapped labels", _unique_list(y_cls.detach().cpu()))
            print("end_idx preview", end_idx[: min(6, int(end_idx.numel()))].detach().cpu().tolist())
            print("logits_all shape", None if logits_all is None else tuple(logits_all.shape))
            print("final_logits shape", tuple(logits_end.shape))
            print("number of supervised endpoints", int(y_cls.numel()))
            print("target shape", tuple(y_cls.shape))
            print("target min/max", int(y_cls.min().item()), int(y_cls.max().item()))
            print("model output dimension", int(logits_end.shape[-1]))
            first_batch_debug_printed = True

        actual_backward_loss = float(total_loss.detach().item())
        weighted_token = float(lambda_token) * float(token_loss.detach().item())
        weighted_online_decision = float(online_loss_weight) * float(online_warmup_factor) * float(online_decision_loss.detach().item())
        weighted_online_ce = float(online_ce_weight) * float(online_ce_loss.detach().item())
        weighted_aux_token_ce = float(aux_token_ce_weight) * float(aux_token_ce_loss.detach().item())
        weighted_anti_immediate = float(anti_immediate_stop_weight) * float(anti_immediate_stop_loss.detach().item())
        weighted_stop_prior = float(stop_prior_weight) * float(stop_prior_loss.detach().item())
        weighted_neg_stop_entropy = -float(stop_entropy_weight) * float(stop_entropy_mean.detach().item())
        reconstructed_loss = (
            float(end_loss.detach().item())
            + weighted_token
            + weighted_online_decision
            + weighted_online_ce
            + weighted_aux_token_ce
            + weighted_anti_immediate
            + weighted_stop_prior
            + weighted_neg_stop_entropy
        )
        if debug_loss_check:
            diff = abs(actual_backward_loss - reconstructed_loss)
            print("[DEBUG loss tensor]")
            print("actual_backward_loss", actual_backward_loss)
            print("reconstructed_loss", reconstructed_loss)
            print("diff", diff)
            print("end_loss", float(end_loss.detach().item()))
            print("lambda_token * token_loss", weighted_token)
            print("online_loss_weight * online_decision_loss", weighted_online_decision)
            print("online_ce_weight * online_ce_loss", weighted_online_ce)
            print("aux_token_ce_weight * aux_token_ce_loss", weighted_aux_token_ce)
            print("anti_immediate_stop_weight * anti_immediate_stop_loss", weighted_anti_immediate)
            print("stop_prior_weight * stop_prior_loss", weighted_stop_prior)
            print("- stop_entropy_weight * stop_entropy_mean", weighted_neg_stop_entropy)
            assert diff < 1e-5, f"Loss mismatch: actual={actual_backward_loss} reconstructed={reconstructed_loss} diff={diff}"
        total_loss.backward()

        grad_norm_gru = float("nan")
        grad_norm_class_head = float("nan")
        gru_sq = 0.0
        class_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            gn = float(param.grad.detach().norm().item())
            if "gru" in name:
                gru_sq += gn * gn
            if ("head" in name) and ("stop_head" not in name):
                class_sq += gn * gn

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        optimizer.step()
        if optimizer_step_state is not None:
            optimizer_step_state["count"] = int(optimizer_step_state.get("count", 0)) + 1
            global_opt_step = int(optimizer_step_state["count"])
        else:
            global_opt_step = step

        if gru_sq > 0.0:
            grad_norm_gru = gru_sq ** 0.5
        if class_sq > 0.0:
            grad_norm_class_head = class_sq ** 0.5

        total_end += float(end_loss.item()) * B
        total_tok += float(token_loss.item()) * B
        total_anti += float(anti_commit_loss.item()) * B

        with torch.no_grad():
            probs = torch.softmax(logits_end, dim=-1).detach().cpu().numpy().reshape(-1, 3)
            pred = probs.argmax(axis=1)
            yt = y_cls.detach().cpu().numpy().reshape(-1)
            y_true_all.append(yt)
            y_pred_all.append(pred)
            y_prob_all.append(probs)

        if logits_all is None:
            rt_miss += int(y_cls.numel())
        else:
            logits_trial = logits_all.view(B, 10, int(trial_T_tokens), 3).detach().to("cpu")
            y_cpu_rt = y.detach().to("cpu").long()
            y_cls_cpu = (y_cpu_rt - 4).long()

            ymin = int(y_cls_cpu.min().item())
            ymax = int(y_cls_cpu.max().item())
            if ymin < 0 or ymax > 2:
                uniq = torch.unique(y_cpu_rt).tolist()
                uniq_cls = torch.unique(y_cls_cpu).tolist()
                print(f"[TRAIN RT warn] invalid y values: unique y_pos_456={uniq} -> unique y_cls={uniq_cls}. Skip RT for this batch.")
                rt_miss += int(y_cls_cpu.numel())
            else:
                dev_end_cpu = deviant_end_token_in_trial(
                    y_pos_456=y_cpu_rt,
                    tone_T=int(tone_T),
                    isi_T=int(isi_T),
                )

                rt_tokens_cpu, found_cpu, pred_at_rt_cpu = compute_rt_from_logits(
                    logits=logits_trial,
                    y_cls=y_cls_cpu,
                    dev_end=dev_end_cpu,
                    p_thresh=float(rt_p_thresh),
                    k_consec=int(rt_k_consec),
                    rt_mode=str(rt_mode),
                    entropy_thresh=float(rt_entropy_thresh),
                    min_rt_tokens=int(min_rt_tokens),
                )

                if found_cpu.any():
                    rt_vals_tokens = rt_tokens_cpu[found_cpu].float()
                    rt_tokens_sum += float(rt_vals_tokens.sum().item())
                    rt_vals_ms = rt_vals_tokens * float(token_ms)
                    rt_ms_sum += float(rt_vals_ms.sum().item())
                    rt_n += int(rt_vals_tokens.numel())
                    rt_not_first += int((rt_tokens_cpu[found_cpu] > 1).sum().item())

                rt_miss += int((~found_cpu).sum().item())

                if online_cost_monitor:
                    oc = compute_online_decision_cost_from_logits(
                        logits=logits_trial,
                        y_cls=y_cls_cpu,
                        y_pos_456=y_cpu_rt,
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                        rt_mode=str(rt_mode),
                        rt_p_thresh=float(rt_p_thresh),
                        rt_entropy_thresh=float(rt_entropy_thresh),
                        rt_k_consec=int(rt_k_consec),
                        min_rt_tokens=int(min_rt_tokens),
                        token_ms=int(token_ms),
                        decision_anchor=str(online_cost_anchor),
                        wrong_cost=float(online_cost_wrong),
                        w_time=float(online_cost_w_time),
                    )
                    online_cost_sum += float(oc["online_decision_cost"]) * float(B * 10)
                    online_cost_n += int(B * 10)

        if debug and step <= int(debug_steps):
            w1 = next(model.parameters()).detach().float().cpu()
            delta = (w1 - w0).abs().mean().item() if w0 is not None else float("nan")
            batch_acc = (torch.softmax(logits_end, dim=-1).argmax(dim=-1) == y_cls).float().mean().item()
            print(
                f"[debug step {step}] mean|Δparam|={delta:.6e} "
                f"end={end_loss.item():.4f} tok={float(token_loss.item()):.4f} "
                f"anti={float(anti_commit_loss.item()):.4f} "
                f"acc={batch_acc:.3f} logits_mean={logits_end.detach().float().mean().item():.4f}"
            )

        if debug_overfit_tiny and (global_opt_step == 1 or global_opt_step % 25 == 0):
            train_acc = float((torch.softmax(logits_end, dim=-1).argmax(dim=-1) == y_cls).float().mean().item())
            print(
                f"[debug_overfit step {global_opt_step}] "
                f"loss={actual_backward_loss:.6f} end_loss={float(end_loss.detach().item()):.6f} "
                f"train_acc={train_acc:.4f} grad_norm_gru={grad_norm_gru:.6f} "
                f"grad_norm_class_head={grad_norm_class_head:.6f}"
            )
            if (not np.isfinite(grad_norm_gru)) or grad_norm_gru == 0.0:
                print("[WARNING] grad_norm_gru is zero or non-finite")
            if (not np.isfinite(grad_norm_class_head)) or grad_norm_class_head == 0.0:
                print("[WARNING] grad_norm_class_head is zero or non-finite")

        dt = time.time() - t0
        if ema_step is None:
            ema_step = dt
        else:
            ema_step = ema_beta * ema_step + (1 - ema_beta) * dt

        if (log_every > 0) and (step % log_every == 0 or step == 1 or step == n_steps):
            steps_left = max(0, n_steps - step)
            eta_epoch = ema_step * steps_left if ema_step is not None else float("nan")
            elapsed = time.time() - t_epoch0

            with torch.no_grad():
                batch_acc = (torch.softmax(logits_end, dim=-1).argmax(dim=-1) == y_cls).float().mean().item()

            batch_rt_msg = "batch_meanRT=NA"
            if logits_all is not None:
                logits_trial = logits_all.view(B, 10, int(trial_T_tokens), 3).detach().to("cpu")
                y_cpu_rt = y.detach().to("cpu").long()
                y_cls_cpu = (y_cpu_rt - 4).long()
                ymin = int(y_cls_cpu.min().item())
                ymax = int(y_cls_cpu.max().item())
                if ymin >= 0 and ymax <= 2:
                    dev_end_cpu = deviant_end_token_in_trial(
                        y_pos_456=y_cpu_rt,
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                    )
                    rt_tokens_cpu, found_cpu, pred_at_rt_cpu = compute_rt_from_logits(
                        logits=logits_trial,
                        y_cls=y_cls_cpu,
                        dev_end=dev_end_cpu,
                        p_thresh=float(rt_p_thresh),
                        k_consec=int(rt_k_consec),
                        rt_mode=str(rt_mode),
                        entropy_thresh=float(rt_entropy_thresh),
                        min_rt_tokens=int(min_rt_tokens),
                    )
                    if found_cpu.any():
                        batch_mean_rt_tokens = float(rt_tokens_cpu[found_cpu].float().mean().item())
                        batch_mean_rt_ms = batch_mean_rt_tokens * float(token_ms)
                        batch_rt_msg = f"batch_meanRT={batch_mean_rt_tokens:.2f}tok/{batch_mean_rt_ms:.1f}ms"

            mask_info = ""
            if use_mask_loss and token_mask is not None:
                mask_ratio = token_mask.float().mean().item()
                mask_info = f" mask_ratio={mask_ratio:.3f}"

            print(
                f"[train step {step:>4d}/{n_steps}] "
                f"dt={dt:.2f}s ema={ema_step:.2f}s "
                f"ETA_epoch={_fmt_hms(eta_epoch)} elapsed={_fmt_hms(elapsed)} "
                f"end={end_loss.item():.4f} tok={float(token_loss.item()):.4f} "
                f"anti={float(anti_commit_loss.item()):.4f} "
                f"acc={batch_acc:.3f} {batch_rt_msg}{mask_info}"
            )

        if max_optimizer_steps is not None and global_opt_step >= int(max_optimizer_steps):
            break

    denom = max(1, n_examples)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.zeros((0, 3), dtype=np.float32)

    acc = float((y_true == y_pred).mean()) if y_true.size > 0 else float("nan")
    f1 = safe_f1_macro(y_true, y_pred) if y_true.size > 0 else float("nan")
    auc = safe_auc_ovr(y_true, y_prob, n_classes=3) if y_true.size > 0 else float("nan")

    mean_rt_tokens = (rt_tokens_sum / rt_n) if rt_n > 0 else float("nan")
    mean_rt_ms = (rt_ms_sum / rt_n) if rt_n > 0 else float("nan")
    rt_not_first_rate = (rt_not_first / rt_n) if rt_n > 0 else float("nan")

    return {
        "end_loss": total_end / denom,
        "token_loss": total_tok / denom,
        "anti_commit_loss": total_anti / denom,
        "total_loss": (
            total_end
            + float(lambda_token) * total_tok
            + float(lambda_anti_commit) * total_anti
        ) / denom,
        "acc": acc,
        "f1_macro": float(f1),
        "auc_ovr": float(auc),
        "mean_rt_tokens": float(mean_rt_tokens),
        "mean_rt_ms": float(mean_rt_ms),
        "rt_found": int(rt_n),
        "rt_miss": int(rt_miss),
        "rt_not_first": int(rt_not_first),
        "rt_not_first_rate": float(rt_not_first_rate),
        "online_decision_cost": float(online_cost_sum / max(1, online_cost_n)) if online_cost_n > 0 else float("nan"),
        "online_decision_loss": float(online_loss_sum / denom) if online_decision_training else float("nan"),
        "online_ce_loss": float(online_ce_sum / denom) if online_decision_training else float("nan"),
        "aux_token_ce_loss": float(aux_token_ce_sum / denom) if aux_token_ce_sum > 0 else 0.0,
        "phase_name": str(phase_name),
        "anti_immediate_stop_loss": float(anti_immediate_stop_sum / denom) if anti_immediate_stop_sum > 0 else 0.0,
        "stop_entropy_bonus": float(stop_entropy_sum / denom) if stop_entropy_sum > 0 else 0.0,
        "stop_prior_loss": float(stop_prior_sum / denom) if stop_prior_sum > 0 else 0.0,
        "effective_online_loss_weight": float(online_loss_weight) * float(online_warmup_factor) if include_online_decision_loss else 0.0,
        "weighted_token_loss": float(lambda_token) * (total_tok / denom),
        "weighted_online_decision_loss": (float(online_loss_weight) * float(online_warmup_factor) * (online_loss_sum / denom)) if include_online_decision_loss and denom > 0 else 0.0,
        "weighted_online_ce_loss": (float(online_ce_weight) * (online_ce_sum / denom)) if include_online_ce_loss and denom > 0 else 0.0,
        "weighted_aux_token_ce_loss": float(weighted_aux_token_ce_sum / denom) if weighted_aux_token_ce_sum > 0 else 0.0,
        "mean_p_stop": float(mean_p_stop_sum / max(1, mean_p_stop_n)) if mean_p_stop_n > 0 else float("nan"),
        "mean_no_response_prob": float(mean_noresp_sum / max(1, mean_noresp_n)) if mean_noresp_n > 0 else float("nan"),
        "mean_expected_decision_time_ms": float(mean_expdt_sum / max(1, mean_expdt_n)) if mean_expdt_n > 0 else float("nan"),
        "expected_found_prob": float(expected_found_prob_sum / max(1, expected_found_prob_n)) if expected_found_prob_n > 0 else float("nan"),
        "expected_rt_logged_ms": float(expected_rt_logged_sum / max(1, expected_rt_logged_n)) if expected_rt_logged_n > 0 else float("nan"),
        "expected_rt_from_deviant_onset_ms": float(expected_rt_from_devon_sum / max(1, expected_rt_from_devon_n)) if expected_rt_from_devon_n > 0 else float("nan"),
        "expected_rt_from_deviant_end_ms": float(expected_rt_from_devend_sum / max(1, expected_rt_from_devend_n)) if expected_rt_from_devend_n > 0 else float("nan"),
        "proportion_negative_rt": float(prop_negative_rt_sum / max(1, prop_negative_rt_n)) if prop_negative_rt_n > 0 else float("nan"),
        "proportion_decisions_before_deviant_end": float(prop_before_devend_sum / max(1, prop_before_devend_n)) if prop_before_devend_n > 0 else float("nan"),
        "last_token_acc": float(last_token_acc_sum / max(1, last_token_acc_n)) if last_token_acc_n > 0 else float("nan"),
        "deviant_end_acc": float(devend_acc_sum / max(1, devend_acc_n)) if devend_acc_n > 0 else float("nan"),
        "sampled_mean_rt_logged_ms": float("nan"),
        "epoch_time_sec": time.time() - t_epoch0,
    }

@torch.no_grad()
def evaluate(
    model: PredictiveGRU,
    loader,
    device: torch.device,
    chunk_len: int,
    lambda_token: float,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_loss_mode: str,
    token_tau: float,
    token_w_min: float,
    token_ms: int,
    tok_window_ms: int,
    tok_start_offset_ms: int,
    tok_anchor: str,
    lambda_anti_commit: float = 0.0,
    anti_commit_window_ms: int = 0,
    anti_commit_start_offset_ms: int = 0,
    anti_commit_max_conf: float = 1.0,
    rt_p_thresh: float = 0.7,
    rt_k_consec: int = 3,
    rt_mode: str = "entropy",
    rt_entropy_thresh: float = 0.35,
    min_rt_tokens: int = 0,
    debug_labels: bool = False,
    debug_labels_fatal: bool = False,
    debug_labels_dump: bool = False,
    debug_labels_dump_dir: Optional[Path] = None,
    debug_labels_max_dumps: int = 10,
    debug_labels_first_n_batches: int = 0,
    epoch_global: int = 0,
    use_mask_loss: bool = False,
    important_token_indices: Optional[List[int]] = None,
    gap_weight_power: float = 0.0,
    gap_weight_ref_hz: float = 25.0,
    gap_weight_max: float = 2.5,
    online_cost_monitor: bool = False,
    online_cost_anchor: str = "deviant_onset",
    online_cost_wrong: float = 1.0,
    online_cost_w_time: float = 0.001,
    online_decision_training: bool = False,
    online_loss_weight: float = 0.5,
    online_ce_weight: float = 0.1,
    decision_cost_mode: str = "expected_cost_softmin",
    timeout_ms: Optional[float] = None,
    time_cost_w: Optional[float] = 0.001,
    derive_time_cost_from_timeout: bool = False,
    decision_reference: str = "trial_onset",
    rt_logging_reference: str = "deviant_end",
    cost_reference: str = "deviant_onset",
    clamp_negative_cost_time: bool = True,
    online_supervision_start: str = "deviant_start",
    online_supervision_end: str = "trial_end",
    aux_token_ce_weight: float = 0.0,
    aux_token_ce_start: str = "deviant_onset",
    aux_token_ce_end: str = "trial_end",
    decision_softmin_tau: float = 0.05,
    use_stop_head: bool = False,
    decision_policy: str = "deterministic_threshold",
    sampling_temperature: float = 1.0,
    stop_temperature: float = 1.0,
    use_hazard_prior: bool = False,
    hazard_prior_weight: float = 1.0,
    hazard_prior_mode: str = "add_log_prior",
    online_warmup_factor: float = 1.0,
    phase_name: str = "single_phase",
    include_online_decision_loss: bool = True,
    include_online_ce_loss: bool = True,
    anti_immediate_stop: bool = False,
    anti_immediate_stop_tokens: int = 5,
    anti_immediate_stop_weight: float = 0.1,
    pre_devend_stop_weight: float = 0.0,
    stop_entropy_weight: float = 0.0,
    stop_prior_weight: float = 0.0,
    stop_prior_target: float = 0.05,
) -> dict:
    model.eval()
    ce_end = nn.CrossEntropyLoss(reduction="mean")
    ce_tok = nn.CrossEntropyLoss(reduction="mean")

    total_end = 0.0
    total_tok = 0.0
    total_anti = 0.0
    n_examples = 0

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []

    rt_tokens_sum = 0.0
    rt_ms_sum = 0.0
    rt_n = 0
    rt_miss = 0
    rt_not_first = 0
    online_cost_sum = 0.0
    online_cost_n = 0
    online_loss_sum = 0.0
    online_ce_sum = 0.0
    mean_p_stop_sum = 0.0
    mean_p_stop_n = 0
    mean_noresp_sum = 0.0
    mean_noresp_n = 0
    mean_expdt_sum = 0.0
    mean_expdt_n = 0
    expected_found_prob_sum = 0.0
    expected_found_prob_n = 0
    expected_rt_logged_sum = 0.0
    expected_rt_logged_n = 0
    expected_rt_from_devon_sum = 0.0
    expected_rt_from_devon_n = 0
    expected_rt_from_devend_sum = 0.0
    expected_rt_from_devend_n = 0
    prop_negative_rt_sum = 0.0
    prop_negative_rt_n = 0
    prop_before_devend_sum = 0.0
    prop_before_devend_n = 0
    anti_immediate_stop_sum = 0.0
    stop_entropy_sum = 0.0
    stop_prior_sum = 0.0
    aux_token_ce_sum = 0.0
    weighted_aux_token_ce_sum = 0.0
    last_token_acc_sum = 0.0
    last_token_acc_n = 0
    devend_acc_sum = 0.0
    devend_acc_n = 0

    for batch in loader:
        x, y, gap_hz = unpack_batch_with_optional_gap(batch)
        if debug_labels_first_n_batches and debug_labels:
            print(f"[debug_labels] eval batch y(unique)={_unique_list(y)}")

        _check_y_and_maybe_debug(
            y_cpu=y,
            where="eval/batch_received",
            debug_on=debug_labels,
            fatal=debug_labels_fatal,
            dump_on=debug_labels_dump,
            dump_dir=debug_labels_dump_dir,
            dump_tag=f"eval_ep{epoch_global:04d}_received",
            dump_info={"epoch_global": epoch_global},
            dump_tensors=None,
            max_dumps=debug_labels_max_dumps,
        )

        x = x.to(device, non_blocking=True)
        y_cpu = y.long()
        y = y_cpu.to(device, non_blocking=True)
        gap_weights_bt = make_gap_weight_tensor(
            gap_hz_bt=(gap_hz.to(device, non_blocking=True) if gap_hz is not None else None),
            power=float(gap_weight_power),
            ref_hz=float(gap_weight_ref_hz),
            max_weight=float(gap_weight_max),
        )
        B, T, D = x.shape
        n_examples += B

        end_idx = infer_end_indices_from_T(T, trials_per_block=10).to(device)

        logits_end, token_loss, anti_commit_loss, logits_all, _, stop_logits_all = _run_block_through_tbptt(
            model=model,
            x=x,
            y_pos_456=y,
            trial_weights_bt=gap_weights_bt,
            end_idx=end_idx,
            chunk_len=int(chunk_len),
            token_ce=ce_tok,
            trial_T_tokens=int(trial_T_tokens),
            tone_T=int(tone_T),
            isi_T=int(isi_T),
            token_ms=int(token_ms),
            tok_window_ms=int(tok_window_ms),
            tok_start_offset_ms=int(tok_start_offset_ms),
            token_loss_mode=str(token_loss_mode),
            token_tau=float(token_tau),
            token_w_min=float(token_w_min),
            tok_anchor=str(tok_anchor),
            anti_commit_window_ms=int(anti_commit_window_ms),
            anti_commit_start_offset_ms=int(anti_commit_start_offset_ms),
            anti_commit_max_conf=float(anti_commit_max_conf),
            return_full_logits=True,
            return_stop_logits=bool(use_stop_head),
            use_mask_loss=use_mask_loss,
            important_token_indices=important_token_indices,
        )

        y_cls = labels_to_class_index(y)
        assert int(y_cls.min().item()) >= 0
        assert int(y_cls.max().item()) <= 2
        end_loss_per = F.cross_entropy(
            logits_end.reshape(-1, 3),
            y_cls.reshape(-1),
            reduction="none",
        ).reshape(y_cls.shape)
        if gap_weights_bt is None:
            end_loss = end_loss_per.mean()
        else:
            end_loss = (end_loss_per * gap_weights_bt).sum() / gap_weights_bt.sum().clamp_min(1e-8)

        total_end += float(end_loss.item()) * B
        total_tok += float(token_loss.item()) * B
        total_anti += float(anti_commit_loss.item()) * B
        if (bool(online_decision_training) or bool(include_online_ce_loss)) and (logits_all is not None):
            class_trial = logits_all.view(B, 10, int(trial_T_tokens), 3)
            if bool(use_hazard_prior) and str(hazard_prior_mode) == "add_log_prior":
                log_prior = get_optimal_hazard_prior(y, int(trial_T_tokens), int(tone_T), int(isi_T))
                class_trial = class_trial + float(hazard_prior_weight) * log_prior
            abs_t = torch.arange(0, int(T), device=x.device)
            elig = make_online_eligibility_mask_tokens(
                abs_t=abs_t,
                y_pos_456=y,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
                start_mode=str(online_supervision_start),
                end_mode=str(online_supervision_end),
            ).view(B, 10, int(trial_T_tokens))
            stop_trial = None
            if stop_logits_all is not None and torch.isfinite(stop_logits_all).any():
                stop_trial = stop_logits_all.view(B, 10, int(trial_T_tokens), 1)

            if str(decision_cost_mode) == "expected_cost_softmin":
                online_decision_loss, od_stats = compute_expected_cost_softmin_loss(
                    class_logits_trial=class_trial,
                    y_cls=y_cls,
                    y_pos_456=y,
                    eligible_mask_trial=elig,
                    token_ms=int(token_ms),
                    tone_T=int(tone_T),
                    isi_T=int(isi_T),
                    decision_reference=str(decision_reference),
                    rt_logging_reference=str(rt_logging_reference),
                    cost_reference=str(cost_reference),
                    clamp_negative_cost_time=bool(clamp_negative_cost_time),
                    timeout_ms=(None if timeout_ms is None else float(timeout_ms)),
                    time_cost_w=time_cost_w,
                    derive_time_cost_from_timeout=bool(derive_time_cost_from_timeout),
                    softmin_tau=float(decision_softmin_tau),
                    sampling_temperature=float(sampling_temperature),
                )
            elif str(decision_cost_mode) == "stochastic_expected_cost":
                online_decision_loss, od_stats = compute_stochastic_expected_cost_loss(
                    class_logits_trial=class_trial,
                    stop_logits_trial=stop_trial,
                    y_cls=y_cls,
                    y_pos_456=y,
                    eligible_mask_trial=elig,
                    token_ms=int(token_ms),
                    tone_T=int(tone_T),
                    isi_T=int(isi_T),
                    decision_reference=str(decision_reference),
                    rt_logging_reference=str(rt_logging_reference),
                    cost_reference=str(cost_reference),
                    clamp_negative_cost_time=bool(clamp_negative_cost_time),
                    timeout_ms=(None if timeout_ms is None else float(timeout_ms)),
                    time_cost_w=time_cost_w,
                    derive_time_cost_from_timeout=bool(derive_time_cost_from_timeout),
                    sampling_temperature=float(sampling_temperature),
                    stop_temperature=float(stop_temperature),
                )
            elif str(decision_cost_mode) == "masked_online_ce":
                tgt = y_cls.unsqueeze(-1).expand(-1, -1, int(trial_T_tokens))
                assert int(tgt.min().item()) >= 0
                assert int(tgt.max().item()) <= 2
                ce_tok = F.cross_entropy(class_trial.reshape(-1, 3), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
                online_decision_loss = (ce_tok * elig.float()).sum() / elig.float().sum().clamp_min(1e-8)
                od_stats = {}
            elif str(decision_cost_mode) == "policy_gradient":
                raise NotImplementedError("policy_gradient mode not implemented yet; use stochastic_expected_cost.")
            else:
                raise ValueError(f"Unknown decision_cost_mode: {decision_cost_mode}")

            tgt = y_cls.unsqueeze(-1).expand(-1, -1, int(trial_T_tokens))
            assert int(tgt.min().item()) >= 0
            assert int(tgt.max().item()) <= 2
            ce_tok = F.cross_entropy(class_trial.reshape(-1, 3), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
            online_ce_loss = (ce_tok * elig.float()).sum() / elig.float().sum().clamp_min(1e-8)
            if not torch.isfinite(online_ce_loss):
                online_ce_loss = end_loss.new_tensor(0.0)
            aux_mask = make_online_eligibility_mask_tokens(
                abs_t=abs_t,
                y_pos_456=y,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
                start_mode=str(aux_token_ce_start),
                end_mode=str(aux_token_ce_end),
            ).view(B, 10, int(trial_T_tokens))
            aux_token_ce_loss = (ce_tok * aux_mask.float()).sum() / aux_mask.float().sum().clamp_min(1e-8)
            if not torch.isfinite(aux_token_ce_loss):
                aux_token_ce_loss = end_loss.new_tensor(0.0)
            if include_online_decision_loss:
                online_loss_sum += float(online_decision_loss.item()) * B
            if include_online_ce_loss:
                online_ce_sum += float(online_ce_loss.item()) * B
            aux_token_ce_sum += float(aux_token_ce_loss.item()) * B
            weighted_aux_token_ce_sum += float(aux_token_ce_weight) * float(aux_token_ce_loss.item()) * B
            if stop_trial is not None:
                p_stop = torch.sigmoid(stop_trial.squeeze(-1) / max(1e-6, float(stop_temperature))).clamp(1e-6, 1 - 1e-6)
                if bool(anti_immediate_stop):
                    K = max(1, int(anti_immediate_stop_tokens))
                    early_mask = torch.zeros_like(elig, dtype=torch.bool)
                    for bidx in range(B):
                        for tridx in range(10):
                            idxs = torch.where(elig[bidx, tridx])[0]
                            if idxs.numel() > 0:
                                early = idxs[: min(K, int(idxs.numel()))]
                                early_mask[bidx, tridx, early] = True
                    if early_mask.any():
                        anti_immediate_stop_sum += float(p_stop[early_mask].mean().item()) * B
                if float(stop_entropy_weight) > 0.0:
                    stop_ent = -(p_stop * torch.log(p_stop) + (1 - p_stop) * torch.log(1 - p_stop))
                    ent_bonus = stop_ent[elig].mean() if elig.any() else stop_ent.mean()
                    stop_entropy_sum += float(ent_bonus.item()) * B
                if float(stop_prior_weight) > 0.0:
                    mean_p = p_stop[elig].mean() if elig.any() else p_stop.mean()
                    sp_loss = (mean_p - float(stop_prior_target)) ** 2
                    stop_prior_sum += float(sp_loss.item()) * B
                if float(pre_devend_stop_weight) > 0.0:
                    dev_end_tok = deviant_end_token_in_trial(y, tone_T=int(tone_T), isi_T=int(isi_T))
                    tgrid = torch.arange(int(trial_T_tokens), device=x.device).view(1, 1, -1)
                    before_devend_mask = (tgrid < dev_end_tok.unsqueeze(-1)) & elig
                    if before_devend_mask.any():
                        pre_devend_stop_loss = p_stop[before_devend_mask].mean()
                        anti_immediate_stop_sum += float(pre_devend_stop_loss.item()) * B
            if "mean_p_stop" in od_stats and np.isfinite(od_stats["mean_p_stop"]):

                mean_p_stop_sum += float(od_stats["mean_p_stop"]) * B
                mean_p_stop_n += B
            if "mean_no_response_prob" in od_stats and np.isfinite(od_stats["mean_no_response_prob"]):
                mean_noresp_sum += float(od_stats["mean_no_response_prob"]) * B
                mean_noresp_n += B
            if "mean_expected_decision_time_ms" in od_stats and np.isfinite(od_stats["mean_expected_decision_time_ms"]):
                mean_expdt_sum += float(od_stats["mean_expected_decision_time_ms"]) * B
                mean_expdt_n += B
            if "expected_found_prob" in od_stats and np.isfinite(od_stats["expected_found_prob"]):
                expected_found_prob_sum += float(od_stats["expected_found_prob"]) * B
                expected_found_prob_n += B
            if "expected_rt_logged_ms" in od_stats and np.isfinite(od_stats["expected_rt_logged_ms"]):
                expected_rt_logged_sum += float(od_stats["expected_rt_logged_ms"]) * B
                expected_rt_logged_n += B
            if "expected_rt_from_deviant_onset_ms" in od_stats and np.isfinite(od_stats["expected_rt_from_deviant_onset_ms"]):
                expected_rt_from_devon_sum += float(od_stats["expected_rt_from_deviant_onset_ms"]) * B
                expected_rt_from_devon_n += B
            if "expected_rt_from_deviant_end_ms" in od_stats and np.isfinite(od_stats["expected_rt_from_deviant_end_ms"]):
                expected_rt_from_devend_sum += float(od_stats["expected_rt_from_deviant_end_ms"]) * B
                expected_rt_from_devend_n += B
            if "proportion_negative_rt" in od_stats and np.isfinite(od_stats["proportion_negative_rt"]):
                prop_negative_rt_sum += float(od_stats["proportion_negative_rt"]) * B
                prop_negative_rt_n += B
            if "proportion_decisions_before_deviant_end" in od_stats and np.isfinite(od_stats["proportion_decisions_before_deviant_end"]):
                prop_before_devend_sum += float(od_stats["proportion_decisions_before_deviant_end"]) * B
                prop_before_devend_n += B

            last_pred = class_trial[:, :, -1, :].argmax(dim=-1)
            last_token_acc_sum += float((last_pred == y_cls).float().mean().item()) * B
            last_token_acc_n += B
            dev_end_tok = deviant_end_token_in_trial(y, tone_T=int(tone_T), isi_T=int(isi_T))
            gather_idx = dev_end_tok.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
            devend_logits = class_trial.gather(dim=2, index=gather_idx).squeeze(2)
            devend_pred = devend_logits.argmax(dim=-1)
            devend_acc_sum += float((devend_pred == y_cls).float().mean().item()) * B
            devend_acc_n += B

        probs = torch.softmax(logits_end, dim=-1).detach().cpu().numpy().reshape(-1, 3)
        pred = probs.argmax(axis=1)
        yt = y_cls.detach().cpu().numpy().reshape(-1)
        y_true_all.append(yt)
        y_pred_all.append(pred)
        y_prob_all.append(probs)

        if logits_all is None:
            rt_miss += int(y_cls.numel())
        else:
            logits_trial = logits_all.view(B, 10, int(trial_T_tokens), 3)
            logits_trial_cpu = logits_trial.detach().to("cpu")
            y_cpu = y.detach().to("cpu").long()
            y_cls_cpu = (y_cpu - 4).long()

            ymin = int(y_cls_cpu.min().item())
            ymax = int(y_cls_cpu.max().item())
            if ymin < 0 or ymax > 2:
                uniq = torch.unique(y_cpu).tolist()
                uniq_cls = torch.unique(y_cls_cpu).tolist()
                print(f"[RT warn] invalid y values for RT: unique y_pos_456={uniq} -> unique y_cls={uniq_cls}. Skip RT for this batch.")
                rt_miss += int(y_cls_cpu.numel())
            else:
                dev_end_cpu = deviant_end_token_in_trial(
                    y_pos_456=y_cpu,
                    tone_T=int(tone_T),
                    isi_T=int(isi_T),
                )

                rt_tokens_cpu, found_cpu, pred_at_rt_cpu = compute_rt_from_logits(
                    logits=logits_trial_cpu,
                    y_cls=y_cls_cpu,
                    dev_end=dev_end_cpu,
                    p_thresh=float(rt_p_thresh),
                    k_consec=int(rt_k_consec),
                    rt_mode=str(rt_mode),
                    entropy_thresh=float(rt_entropy_thresh),
                    min_rt_tokens=int(min_rt_tokens),
                )

                if found_cpu.any():
                    rt_vals_tokens = rt_tokens_cpu[found_cpu].float()
                    rt_tokens_sum += float(rt_vals_tokens.sum().item())
                    rt_vals_ms = rt_vals_tokens * float(token_ms)
                    rt_ms_sum += float(rt_vals_ms.sum().item())
                    rt_n += int(rt_vals_tokens.numel())
                    rt_not_first += int((rt_tokens_cpu[found_cpu] > 1).sum().item())

                rt_miss += int((~found_cpu).sum().item())

                if online_cost_monitor:
                    oc = compute_online_decision_cost_from_logits(
                        logits=logits_trial_cpu,
                        y_cls=y_cls_cpu,
                        y_pos_456=y_cpu,
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                        rt_mode=str(rt_mode),
                        rt_p_thresh=float(rt_p_thresh),
                        rt_entropy_thresh=float(rt_entropy_thresh),
                        rt_k_consec=int(rt_k_consec),
                        min_rt_tokens=int(min_rt_tokens),
                        token_ms=int(token_ms),
                        decision_anchor=str(online_cost_anchor),
                        wrong_cost=float(online_cost_wrong),
                        w_time=float(online_cost_w_time),
                    )
                    online_cost_sum += float(oc["online_decision_cost"]) * float(B * 10)
                    online_cost_n += int(B * 10)

    denom = max(1, n_examples)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.zeros((0, 3), dtype=np.float32)

    acc = float((y_true == y_pred).mean()) if y_true.size > 0 else float("nan")
    f1 = safe_f1_macro(y_true, y_pred) if y_true.size > 0 else float("nan")
    auc = safe_auc_ovr(y_true, y_prob, n_classes=3) if y_true.size > 0 else float("nan")

    mean_rt_tokens = (rt_tokens_sum / rt_n) if rt_n > 0 else float("nan")
    mean_rt_ms = (rt_ms_sum / rt_n) if rt_n > 0 else float("nan")
    rt_not_first_rate = (rt_not_first / rt_n) if rt_n > 0 else float("nan")

    return {
        "end_loss": total_end / denom,
        "token_loss": total_tok / denom,
        "anti_commit_loss": total_anti / denom,
        "total_loss": (
            total_end
            + float(lambda_token) * total_tok
            + float(lambda_anti_commit) * total_anti
        ) / denom,
        "acc": acc,
        "f1_macro": float(f1),
        "auc_ovr": float(auc),
        "mean_rt_tokens": float(mean_rt_tokens),
        "mean_rt_ms": float(mean_rt_ms),
        "rt_found": int(rt_n),
        "rt_miss": int(rt_miss),
        "rt_not_first": int(rt_not_first),
        "rt_not_first_rate": float(rt_not_first_rate),
        "online_decision_cost": float(online_cost_sum / max(1, online_cost_n)) if online_cost_n > 0 else float("nan"),
        "online_decision_loss": float(online_loss_sum / denom) if online_decision_training else float("nan"),
        "online_ce_loss": float(online_ce_sum / denom) if online_decision_training else float("nan"),
        "aux_token_ce_loss": float(aux_token_ce_sum / denom) if aux_token_ce_sum > 0 else 0.0,
        "phase_name": str(phase_name),
        "anti_immediate_stop_loss": float(anti_immediate_stop_sum / denom) if anti_immediate_stop_sum > 0 else 0.0,
        "stop_entropy_bonus": float(stop_entropy_sum / denom) if stop_entropy_sum > 0 else 0.0,
        "stop_prior_loss": float(stop_prior_sum / denom) if stop_prior_sum > 0 else 0.0,
        "effective_online_loss_weight": float(online_loss_weight) * float(online_warmup_factor) if include_online_decision_loss else 0.0,
        "weighted_token_loss": float(lambda_token) * (total_tok / denom),
        "weighted_online_decision_loss": (float(online_loss_weight) * float(online_warmup_factor) * (online_loss_sum / denom)) if include_online_decision_loss and denom > 0 else 0.0,
        "weighted_online_ce_loss": (float(online_ce_weight) * (online_ce_sum / denom)) if include_online_ce_loss and denom > 0 else 0.0,
        "weighted_aux_token_ce_loss": float(weighted_aux_token_ce_sum / denom) if weighted_aux_token_ce_sum > 0 else 0.0,
        "mean_p_stop": float(mean_p_stop_sum / max(1, mean_p_stop_n)) if mean_p_stop_n > 0 else float("nan"),
        "mean_no_response_prob": float(mean_noresp_sum / max(1, mean_noresp_n)) if mean_noresp_n > 0 else float("nan"),
        "mean_expected_decision_time_ms": float(mean_expdt_sum / max(1, mean_expdt_n)) if mean_expdt_n > 0 else float("nan"),
        "expected_found_prob": float(expected_found_prob_sum / max(1, expected_found_prob_n)) if expected_found_prob_n > 0 else float("nan"),
        "expected_rt_logged_ms": float(expected_rt_logged_sum / max(1, expected_rt_logged_n)) if expected_rt_logged_n > 0 else float("nan"),
        "expected_rt_from_deviant_onset_ms": float(expected_rt_from_devon_sum / max(1, expected_rt_from_devon_n)) if expected_rt_from_devon_n > 0 else float("nan"),
        "expected_rt_from_deviant_end_ms": float(expected_rt_from_devend_sum / max(1, expected_rt_from_devend_n)) if expected_rt_from_devend_n > 0 else float("nan"),
        "proportion_negative_rt": float(prop_negative_rt_sum / max(1, prop_negative_rt_n)) if prop_negative_rt_n > 0 else float("nan"),
        "proportion_decisions_before_deviant_end": float(prop_before_devend_sum / max(1, prop_before_devend_n)) if prop_before_devend_n > 0 else float("nan"),
        "last_token_acc": float(last_token_acc_sum / max(1, last_token_acc_n)) if last_token_acc_n > 0 else float("nan"),
        "deviant_end_acc": float(devend_acc_sum / max(1, devend_acc_n)) if devend_acc_n > 0 else float("nan"),
        "sampled_mean_rt_logged_ms": float("nan"),
    }

# ============================================================
# (NEW) Export trial-level RTs on VAL using BEST checkpoint
# ============================================================
def export_model_trial_csv_from_checkpoint(
    ckpt_path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    isi_ms_for_export: int,
    out_name: str = "model_trial.csv",
) -> Optional[Path]:
    """
    Load best.pt (or any ckpt) and compute trial-level RTs on the VAL split
    for a specified ISI (e.g., 700). Save to run_dir/out_name with columns:
      model_id, isi_ms, position, rt_ms, found, block_idx, trial_idx

    Notes:
      - rt_ms can be negative (kept).
      - Uses the same val split indices as training (seed + val_split).
      - Uses args.rt_p_thresh / args.rt_k_consec as the RT criterion.
      - model_id defaults to run_dir.name (filesystem-safe unique id).
    """
    if not ckpt_path.exists():
        print(f"[export_model_csv] checkpoint not found: {ckpt_path}")
        return None

    try:
        import pandas as pd
        _HAVE_PANDAS = True
    except ImportError:
        print("[export_model_csv] pandas not installed; cannot write CSV.")
        return None

    # --- load checkpoint ---
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError("[export_model_csv] ckpt missing 'cfg'.")

    # build dataset once to get splits (same logic as training)
    base_ds = OnlineRenderDataset(
        data_dir=Path(args.data_dir),
        seed=args.seed,
        tone_ms=args.tone_ms,
        isi_ms=int(args.isi_schedule[0]),  # dummy; we only need B + X/Y for split
        ramp_ms=args.ramp_ms,
        token_ms=args.token_ms,
        f_min_hz=args.f_min_hz,
        f_max_hz=args.f_max_hz,
        n_bins=args.n_bins,
        add_eos=bool(args.add_eos),
        sigma_other_noise=float(args.sigma_other_noise),
        p_other_noise=float(args.p_other_noise),
        sigma_silence_noise=float(args.sigma_silence_noise),
        resample_noise_per_epoch=False,
        quiet=True,
    )
    # match training's max_blocks truncation
    if getattr(args, "max_blocks", 0) and int(args.max_blocks) > 0:
        m = min(int(args.max_blocks), int(base_ds.B))
        base_ds.X = base_ds.X[:m]
        base_ds.Y = base_ds.Y[:m]
        base_ds.B = m

    train_idx, val_idx = split_indices(len(base_ds), args.val_split, args.seed)

    # build loaders for the target ISI using same split
    device = resolve_device(args.device)
    ds, _, val_loader = build_loaders_for_isi(
        data_dir=Path(args.data_dir),
        seed=args.seed,
        tone_ms=args.tone_ms,
        isi_ms=int(isi_ms_for_export),
        ramp_ms=args.ramp_ms,
        token_ms=args.token_ms,
        f_min_hz=args.f_min_hz,
        f_max_hz=args.f_max_hz,
        n_bins=args.n_bins,
        add_eos=bool(args.add_eos),
        sigma_other_noise=float(args.sigma_other_noise),
        p_other_noise=float(args.p_other_noise),
        sigma_silence_noise=float(args.sigma_silence_noise),
        resample_noise_per_epoch=False,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        assert_labels=bool(getattr(args, "debug_dataset_assert", False)),
    )

    # rebuild model
    cfg = ModelConfig(
        input_dim=int(cfg_dict["input_dim"]),
        hidden_dim=int(cfg_dict["hidden_dim"]),
        num_layers=int(cfg_dict["num_layers"]),
        dropout=float(cfg_dict["dropout"]),
        layer_norm=bool(cfg_dict.get("layer_norm", False)),
        hidden_noise_std=float(cfg_dict.get("hidden_noise_std", 0.0)),
        use_stop_head=bool(cfg_dict.get("use_stop_head", False)),
    )
    model = PredictiveGRU(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    rows = []
    model_id = run_dir.name

    # iterate val loader, compute trial-level RTs
    with torch.no_grad():
        for batch_i, (x, y_pos_456) in enumerate(val_loader):
            x = x.to(device, non_blocking=True)
            y_pos_456 = y_pos_456.long().to(device, non_blocking=True)

            B, T, D = x.shape
            # run full logits
            end_idx = infer_end_indices_from_T(T, trials_per_block=10).to(device)
            logits_end, token_loss, _, logits_all, _, _ = _run_block_through_tbptt(
                model=model,
                x=x,
                y_pos_456=y_pos_456,
                trial_weights_bt=None,
                end_idx=end_idx,
                chunk_len=int(args.chunk_len),
                token_ce=nn.CrossEntropyLoss(reduction="mean"),
                trial_T_tokens=int(ds.trial_T_tokens),
                tone_T=int(ds.tone_T),
                isi_T=int(ds.isi_T),
                token_ms=int(ds.token_ms),
                tok_window_ms=int(args.tok_window_ms),
                tok_start_offset_ms=int(args.tok_start_offset_ms),
                token_loss_mode=str(args.token_loss_mode),
                token_tau=float(args.token_tau),
                token_w_min=float(args.token_w_min),
                tok_anchor=str(getattr(args, "tok_anchor", "deviant_end")),
                return_full_logits=True,
                use_mask_loss=bool(getattr(args, "use_mask_loss", False)),
                important_token_indices=getattr(args, "important_token_indices", None),
            )

            if logits_all is None:
                # if for any reason logits_all missing, write missing rows as found=0
                y_cpu = y_pos_456.detach().cpu()
                for b in range(B):
                    for tr in range(10):
                        rows.append({
                            "model_id": model_id,
                            "isi_ms": int(isi_ms_for_export),
                            "position": int(y_cpu[b, tr].item()),
                            "rt_ms": np.nan,
                            "found": 0,
                            "block_idx": int(batch_i * int(args.batch_size) + b),
                            "trial_idx": int(tr),
                        })
                continue

            # compute RT on CPU for safety
            logits_trial = logits_all.view(B, 10, int(ds.trial_T_tokens), 3).detach().cpu()
            y_cpu = y_pos_456.detach().cpu().long()
            y_cls_cpu = (y_cpu - 4).long()
            dev_end_cpu = deviant_end_token_in_trial(y_pos_456=y_cpu, tone_T=int(ds.tone_T), isi_T=int(ds.isi_T))

            rt_tokens_cpu, found_cpu, _ = compute_rt_from_logits(
                logits=logits_trial,
                y_cls=y_cls_cpu,
                dev_end=dev_end_cpu,
                p_thresh=float(args.rt_p_thresh),
                k_consec=int(args.rt_k_consec),
            )
            rt_ms_cpu = rt_tokens_cpu.float() * float(ds.token_ms)

            for b in range(B):
                for tr in range(10):
                    rows.append({
                        "model_id": model_id,
                        "isi_ms": int(isi_ms_for_export),
                        "position": int(y_cpu[b, tr].item()),
                        "rt_ms": float(rt_ms_cpu[b, tr].item()) if np.isfinite(rt_ms_cpu[b, tr].item()) else float(rt_ms_cpu[b, tr].item()),
                        "found": int(bool(found_cpu[b, tr].item())),
                        "block_idx": int(batch_i * int(args.batch_size) + b),
                        "trial_idx": int(tr),
                    })

    out_path = run_dir / out_name
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[export_model_csv] Saved: {out_path}")
    return out_path

# -------------------------
# Device
# -------------------------
def resolve_device(device_str: str) -> torch.device:
    s = device_str.lower()
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


# -------------------------
# Plotting
# -------------------------
def plot_history(history: List[Dict[str, Any]], out_png: Path) -> None:
    # history rows: epoch_global, stage, isi_ms, train_acc, val_acc, train_f1, val_f1, train_auc, val_auc, train_loss, val_loss
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    x = [h["epoch_global"] for h in history]

    def get(key):
        return [h.get(key, float("nan")) for h in history]

    plt.figure()
    plt.plot(x, get("train_acc"), label="train_acc")
    plt.plot(x, get("val_acc"), label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.with_name("acc.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.plot(x, get("train_f1_macro"), label="train_f1_macro")
    plt.plot(x, get("val_f1_macro"), label="val_f1_macro")
    plt.xlabel("epoch")
    plt.ylabel("F1 macro")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.with_name("f1.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.plot(x, get("train_auc_ovr"), label="train_auc_ovr")
    plt.plot(x, get("val_auc_ovr"), label="val_auc_ovr")
    plt.xlabel("epoch")
    plt.ylabel("AUC ovr")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.with_name("auc.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.plot(x, get("train_total_loss"), label="train_total_loss")
    plt.plot(x, get("val_total_loss"), label="val_total_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.with_name("loss.png"), dpi=160)
    plt.close()

# ============================================================
# (NEW) Post-run analysis: position effect slope (beta) + bootstrap CI
# ============================================================
try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False

_ID_CANDIDATES = ["subject_id", "model_id", "run_name", "run_dir", "id"]


def _detect_id_col(df, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in _ID_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find an id column. Tried={_ID_CANDIDATES}. cols={list(df.columns)}")


def _clean_trials_df(df, isi: Optional[int] = None):
    need = {"position", "rt_ms"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. cols={list(df.columns)}")

    out = df.copy()
    out["position"] = pd.to_numeric(out["position"], errors="coerce")
    out["rt_ms"] = pd.to_numeric(out["rt_ms"], errors="coerce")

    # keep only position 4/5/6
    out = out[out["position"].isin([4, 5, 6])].copy()

    # optional isi filter (if column exists)
    if isi is not None and "isi_ms" in out.columns:
        out["isi_ms"] = pd.to_numeric(out["isi_ms"], errors="coerce")
        out = out[out["isi_ms"] == int(isi)].copy()

    # keep negative RTs; drop NaN only
    out = out[out["rt_ms"].notna()].copy()
    return out


def _slope_per_id(df, id_col: str):
    rows = []
    for _id, g in df.groupby(id_col, sort=True):
        x = g["position"].to_numpy(dtype=float) - 5.0  # center at 5
        y = g["rt_ms"].to_numpy(dtype=float)

        if np.unique(x).size < 2 or len(y) < 3:
            beta = np.nan
            intercept = np.nan
        else:
            vx = np.var(x, ddof=0)
            beta = float(np.cov(x, y, ddof=0)[0, 1] / vx) if vx > 0 else np.nan
            intercept = float(np.mean(y) - beta * np.mean(x))

        rows.append(
            {
                id_col: _id,
                "n_trials": int(len(y)),
                "positions_present": ",".join(map(str, sorted(g["position"].unique().tolist()))),
                "beta_ms_per_pos": beta,
                "intercept_ms_at_pos5": intercept,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        samp = rng.choice(v, size=v.size, replace=True)
        means[i] = float(np.mean(samp))

    mean = float(np.mean(v))
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return (mean, lo, hi)


def run_position_effect_analysis(
    human_csv: Path,
    model_csv: Path,
    out_dir: Path,
    isi: Optional[int],
    human_id_col: Optional[str],
    model_id_col: Optional[str],
    n_boot: int,
    seed: int,
) -> None:
    if not _HAVE_PANDAS:
        print("[analysis] pandas not installed; skip position-effect analysis.")
        return
    if not human_csv.exists():
        print(f"[analysis] human_csv not found: {human_csv}. Skip.")
        return
    if not model_csv.exists():
        print(f"[analysis] model_csv not found: {model_csv}. Skip.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # human
    df_h = pd.read_csv(human_csv)
    df_h = _clean_trials_df(df_h, isi=isi)
    hid = _detect_id_col(df_h, preferred=human_id_col)
    per_h = _slope_per_id(df_h, hid)
    mh, loh, hih = _bootstrap_mean_ci(per_h["beta_ms_per_pos"].to_numpy(dtype=float), n_boot=n_boot, seed=seed)
    per_h_out = out_dir / "human_per_id_slopes.csv"
    per_h.to_csv(per_h_out, index=False)

    # model
    df_m = pd.read_csv(model_csv)
    df_m = _clean_trials_df(df_m, isi=isi)
    mid = _detect_id_col(df_m, preferred=model_id_col)
    per_m = _slope_per_id(df_m, mid)
    mm, lom, him = _bootstrap_mean_ci(per_m["beta_ms_per_pos"].to_numpy(dtype=float), n_boot=n_boot, seed=seed)
    per_m_out = out_dir / "model_per_id_slopes.csv"
    per_m.to_csv(per_m_out, index=False)

    summary = pd.DataFrame(
        [
            {
                "dataset": "human",
                "csv": str(human_csv),
                "id_col": hid,
                "isi_filter": isi,
                "n_ids_total": int(per_h.shape[0]),
                "n_ids_valid_beta": int(np.isfinite(per_h["beta_ms_per_pos"]).sum()),
                "mean_beta_ms_per_pos": mh,
                "ci95_low": loh,
                "ci95_high": hih,
            },
            {
                "dataset": "model",
                "csv": str(model_csv),
                "id_col": mid,
                "isi_filter": isi,
                "n_ids_total": int(per_m.shape[0]),
                "n_ids_valid_beta": int(np.isfinite(per_m["beta_ms_per_pos"]).sum()),
                "mean_beta_ms_per_pos": mm,
                "ci95_low": lom,
                "ci95_high": him,
            },
        ]
    )
    out_csv = out_dir / "position_effect_summary.csv"
    out_json = out_dir / "position_effect_summary.json"
    summary.to_csv(out_csv, index=False)
    out_json.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    print("[analysis] Saved position-effect outputs:")
    print("  -", out_csv)
    print("  -", out_json)
    print("  -", per_h_out)
    print("  -", per_m_out)

# -------------------------
# Curriculum + Early stopping
# -------------------------
def split_indices(n: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    if n <= 1:
        return list(range(n)), []
    if float(val_split) <= 0.0:
        return list(range(n)), []
    n_val = max(1, int(round(n * float(val_split))))
    n_train = n - n_val
    if n_train <= 0:
        raise ValueError("val_split too large for dataset size.")
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    return train_idx, val_idx


def build_loaders_for_isi(
    data_dir: Path,
    seed: int,
    tone_ms: int,
    isi_ms: int,
    ramp_ms: int,
    token_ms: int,
    f_min_hz: float,
    f_max_hz: float,
    n_bins: int,
    add_eos: bool,
    sigma_other_noise: float,
    p_other_noise: float,
    sigma_silence_noise: float,
    resample_noise_per_epoch: bool,
    train_idx: List[int],
    val_idx: List[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    assert_labels: bool = False
) -> Tuple[OnlineRenderDataset, DataLoader, DataLoader]:
    ds = OnlineRenderDataset(
        data_dir=data_dir,
        seed=seed,
        tone_ms=tone_ms,
        isi_ms=isi_ms,
        ramp_ms=ramp_ms,
        token_ms=token_ms,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        n_bins=n_bins,
        add_eos=bool(add_eos),
        sigma_other_noise=sigma_other_noise,
        p_other_noise=p_other_noise,
        sigma_silence_noise=sigma_silence_noise,
        resample_noise_per_epoch=bool(resample_noise_per_epoch),
        quiet=True,
        assert_labels=assert_labels,
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return ds, train_loader, val_loader


def build_trialwise_debug_loader(
    data_dir: Path,
    seed: int,
    tone_ms: int,
    isi_ms: int,
    ramp_ms: int,
    token_ms: int,
    f_min_hz: float,
    f_max_hz: float,
    n_bins: int,
    add_eos: bool,
    sigma_other_noise: float,
    p_other_noise: float,
    sigma_silence_noise: float,
    resample_noise_per_epoch: bool,
    n_blocks: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    assert_labels: bool = False,
) -> Tuple[OnlineRenderDataset, TrialwiseRenderDataset, DataLoader]:
    block_ds = OnlineRenderDataset(
        data_dir=data_dir,
        seed=seed,
        tone_ms=tone_ms,
        isi_ms=isi_ms,
        ramp_ms=ramp_ms,
        token_ms=token_ms,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        n_bins=n_bins,
        add_eos=bool(add_eos),
        sigma_other_noise=sigma_other_noise,
        p_other_noise=p_other_noise,
        sigma_silence_noise=sigma_silence_noise,
        resample_noise_per_epoch=bool(resample_noise_per_epoch),
        quiet=False,
        assert_labels=assert_labels,
    )
    if n_blocks > 0 and n_blocks < int(block_ds.B):
        block_ds.X = block_ds.X[: int(n_blocks)]
        block_ds.Y = block_ds.Y[: int(n_blocks)]
        block_ds.B = int(n_blocks)
    trial_ds = TrialwiseRenderDataset(block_ds)
    pin = (device.type == "cuda")
    loader = DataLoader(
        trial_ds,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=pin,
    )
    return block_ds, trial_ds, loader


def _maybe_set_epoch_on_dataset(ds_like: Any, epoch: int) -> None:
    target = getattr(ds_like, "dataset", ds_like)
    if hasattr(target, "set_epoch"):
        target.set_epoch(int(epoch))


def _grad_norm_from_named_params(model: nn.Module, include_key: str) -> float:
    sq = 0.0
    for name, p in model.named_parameters():
        if include_key in name and p.grad is not None:
            g = p.grad.detach()
            sq += float((g * g).sum().item())
    return sq ** 0.5 if sq > 0.0 else 0.0


def _compute_trial_rule_predictions_raw(x_trials_hz: torch.Tensor) -> torch.Tensor:
    """
    x_trials_hz: (N,8) raw per-trial frequencies.
    Returns predicted class indices in {0,1,2} for dev positions {4,5,6}, or -1 if invalid.
    """
    preds = []
    for trial in x_trials_hz:
        uniq, counts = torch.unique(trial, return_counts=True)
        majority = uniq[counts.argmax()]
        mismatch = torch.nonzero(trial != majority, as_tuple=False).flatten()
        if mismatch.numel() == 0:
            preds.append(-1)
            continue
        dev_pos_1based = int(mismatch[0].item()) + 1
        if dev_pos_1based not in (4, 5, 6):
            preds.append(-1)
            continue
        preds.append({4: 0, 5: 1, 6: 2}[dev_pos_1based])
    return torch.tensor(preds, dtype=torch.long)


class DebugAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_seq: (B,T,H)
        attn_logits = self.score(h_seq).squeeze(-1)    # (B,T)
        alpha = torch.softmax(attn_logits, dim=-1)
        pooled = torch.sum(h_seq * alpha.unsqueeze(-1), dim=1)
        return pooled, alpha


def _compute_rendered_trial_indices(
    y_raw: torch.Tensor,
    T: int,
    tone_tokens: int,
    isi_tokens: int,
    post_window_tokens: int,
) -> Dict[str, torch.Tensor]:
    deviant_pos = y_raw.long()
    deviant_idx0 = deviant_pos - 1
    deviant_onset_idx = deviant_idx0 * int(tone_tokens + isi_tokens)
    deviant_end_idx = deviant_onset_idx + int(tone_tokens) - 1
    trial_end_idx = torch.full_like(deviant_end_idx, fill_value=int(T - 1))
    post_end_exclusive = torch.clamp(
        deviant_end_idx + int(post_window_tokens),
        max=int(T),
    )

    assert int(deviant_onset_idx.min().item()) >= 0
    assert int(deviant_end_idx.max().item()) < int(T)
    assert int(post_end_exclusive.max().item()) <= int(T)

    return {
        "deviant_pos": deviant_pos,
        "deviant_onset_idx": deviant_onset_idx.long(),
        "deviant_end_idx": deviant_end_idx.long(),
        "trial_end_idx": trial_end_idx.long(),
        "post_end_exclusive": post_end_exclusive.long(),
    }


def _select_rendered_readout(
    mode: str,
    h_seq: torch.Tensor,
    idx_info: Dict[str, torch.Tensor],
    attention_pool: Optional[DebugAttentionPool],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    B, T, H = h_seq.shape
    batch_idx = torch.arange(B, device=h_seq.device)
    onset_idx = idx_info["deviant_onset_idx"].to(h_seq.device)
    end_idx = idx_info["deviant_end_idx"].to(h_seq.device)
    trial_end_idx = idx_info["trial_end_idx"].to(h_seq.device)
    post_end_exclusive = idx_info["post_end_exclusive"].to(h_seq.device)

    if mode in ("current", "last_token"):
        return h_seq[:, -1, :], None
    if mode == "deviant_onset":
        return h_seq[batch_idx, onset_idx, :], None
    if mode == "deviant_end":
        return h_seq[batch_idx, end_idx, :], None
    if mode == "post_deviant_mean":
        mask = (torch.arange(T, device=h_seq.device).unsqueeze(0) >= end_idx.unsqueeze(1)) & (
            torch.arange(T, device=h_seq.device).unsqueeze(0) < post_end_exclusive.unsqueeze(1)
        )
        pooled = (h_seq * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return pooled, None
    if mode == "post_deviant_max":
        mask = (torch.arange(T, device=h_seq.device).unsqueeze(0) >= end_idx.unsqueeze(1)) & (
            torch.arange(T, device=h_seq.device).unsqueeze(0) < post_end_exclusive.unsqueeze(1)
        )
        masked = h_seq.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        return pooled, None
    if mode == "deviant_to_end_mean":
        mask = (torch.arange(T, device=h_seq.device).unsqueeze(0) >= onset_idx.unsqueeze(1)) & (
            torch.arange(T, device=h_seq.device).unsqueeze(0) <= trial_end_idx.unsqueeze(1)
        )
        pooled = (h_seq * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return pooled, None
    if mode == "attention_pool":
        if attention_pool is None:
            raise RuntimeError("attention_pool mode requires attention module")
        pooled, alpha = attention_pool(h_seq)
        return pooled, alpha
    raise ValueError(f"Unknown debug_rendered_readout_mode: {mode}")


def _make_dynamic_eligible_mask(
    idx_info: Dict[str, torch.Tensor],
    T: int,
    start_mode: str,
    end_mode: str,
    device: torch.device,
) -> torch.Tensor:
    t = torch.arange(T, device=device).unsqueeze(0)
    if start_mode == "trial_start":
        lo = torch.zeros_like(idx_info["deviant_onset_idx"], device=device)
    elif start_mode == "deviant_onset":
        lo = idx_info["deviant_onset_idx"].to(device)
    elif start_mode == "deviant_end":
        lo = idx_info["deviant_end_idx"].to(device)
    else:
        raise ValueError(f"Unknown debug_dynamic_start: {start_mode}")

    if end_mode == "trial_end":
        hi = idx_info["trial_end_idx"].to(device)
    else:
        raise ValueError(f"Unknown debug_dynamic_end: {end_mode}")

    return (t >= lo.unsqueeze(1)) & (t <= hi.unsqueeze(1))


def _resolve_time_reference_idx(
    idx_info: Dict[str, torch.Tensor],
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    if mode == "trial_onset":
        return torch.zeros_like(idx_info["trial_end_idx"], device=device)
    if mode == "trial_start":
        return torch.zeros_like(idx_info["trial_end_idx"], device=device)
    if mode == "deviant_onset":
        return idx_info["deviant_onset_idx"].to(device)
    if mode == "deviant_end":
        return idx_info["deviant_end_idx"].to(device)
    raise ValueError(f"Unknown debug time reference mode: {mode}")


def _compute_aux_token_ce_debug(
    class_logits_all: torch.Tensor,   # (B,T,3)
    y_cls: torch.Tensor,              # (B,)
    mask: torch.Tensor,               # (B,T) bool
) -> torch.Tensor:
    if not mask.any():
        return class_logits_all.new_tensor(0.0)
    B, T, _ = class_logits_all.shape
    target = y_cls.view(B, 1).expand(B, T)
    flat_logits = class_logits_all.reshape(B * T, 3)
    flat_target = target.reshape(B * T)
    flat_mask = mask.reshape(B * T)
    assert int(flat_target.min().item()) >= 0
    assert int(flat_target.max().item()) <= 2
    loss_per = F.cross_entropy(flat_logits, flat_target, reduction="none")
    return loss_per[flat_mask].mean() if flat_mask.any() else class_logits_all.new_tensor(0.0)


def _compute_fixed_readout_accs(
    model: PredictiveGRU,
    h_seq: torch.Tensor,
    idx_info: Dict[str, torch.Tensor],
    y_cls: torch.Tensor,
) -> Dict[str, float]:
    outs = {}
    for mode in ("last_token", "deviant_end", "post_deviant_mean"):
        h_readout, _ = _select_rendered_readout(mode, h_seq, idx_info, attention_pool=None)
        logits = model.classify_from_states(h_readout)
        pred = logits.argmax(dim=-1)
        outs[f"{mode}_acc"] = float((pred == y_cls).float().mean().item())
    return outs


def _compute_expected_stop_loss_debug(
    class_logits_all: torch.Tensor,   # (B,T,3)
    stop_logits_all: torch.Tensor,    # (B,T,1)
    y_cls: torch.Tensor,              # (B,)
    eligible_mask: torch.Tensor,      # (B,T) bool
    idx_info: Dict[str, torch.Tensor],
    token_ms: int,
    class_temperature: float,
    stop_temperature: float,
    time_cost_w: float,
    wrong_cost: float,
    no_response_cost: float,
    stop_entropy_weight: float,
    stop_prior_weight: float,
    stop_prior_target: float,
    anti_immediate_stop_tokens: int,
    anti_immediate_stop_weight: float,
    rt_logging_reference: str,
    cost_reference: str,
    clamp_negative_cost_time: bool,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    B, T, _ = class_logits_all.shape
    probs = torch.softmax(class_logits_all / max(1e-6, float(class_temperature)), dim=-1)
    y_idx = y_cls.view(B, 1, 1).expand(B, T, 1)
    p_correct = probs.gather(-1, y_idx).squeeze(-1)          # (B,T)
    p_stop = torch.sigmoid(stop_logits_all.squeeze(-1) / max(1e-6, float(stop_temperature))).clamp(1e-5, 1 - 1e-5)
    p_stop = torch.where(eligible_mask, p_stop, torch.zeros_like(p_stop))
    q = torch.where(eligible_mask, 1.0 - p_stop, torch.ones_like(p_stop))

    prev_survive = torch.cumprod(q, dim=-1)
    prev_survive = torch.cat([torch.ones_like(prev_survive[:, :1]), prev_survive[:, :-1]], dim=-1)
    p_first_stop = p_stop * prev_survive * eligible_mask.float()
    p_no_response = torch.where(eligible_mask, q, torch.ones_like(q)).prod(dim=-1)

    t_idx = torch.arange(T, device=class_logits_all.device).unsqueeze(0).float()
    decision_time_ms = t_idx * float(token_ms)
    rt_log_ref = _resolve_time_reference_idx(idx_info, rt_logging_reference, class_logits_all.device).float()
    cost_ref = _resolve_time_reference_idx(idx_info, cost_reference, class_logits_all.device).float()

    logged_rt_ms = decision_time_ms - (rt_log_ref.unsqueeze(1) * float(token_ms))
    raw_cost_time_ms = decision_time_ms - (cost_ref.unsqueeze(1) * float(token_ms))
    if clamp_negative_cost_time:
        cost_time_ms = raw_cost_time_ms.clamp(min=0.0)
    else:
        cost_time_ms = raw_cost_time_ms

    cost_t = ((1.0 - p_correct) * float(wrong_cost)) + (p_correct * (float(time_cost_w) * cost_time_ms))

    expected_decision_loss_per = (p_first_stop * cost_t).sum(dim=-1) + (p_no_response * float(no_response_cost))
    expected_decision_loss = expected_decision_loss_per.mean()

    stop_entropy = -(p_stop.clamp_min(1e-12) * torch.log(p_stop.clamp_min(1e-12)) + (1.0 - p_stop).clamp_min(1e-12) * torch.log((1.0 - p_stop).clamp_min(1e-12)))
    stop_entropy_mean = stop_entropy[eligible_mask].mean() if eligible_mask.any() else stop_entropy.mean()
    mean_p_stop = p_stop[eligible_mask].mean() if eligible_mask.any() else p_stop.mean()
    stop_prior_loss = (mean_p_stop - float(stop_prior_target)) ** 2

    K = max(1, int(anti_immediate_stop_tokens))
    early_mask = torch.zeros_like(eligible_mask)
    for b in range(B):
        idxs = torch.where(eligible_mask[b])[0]
        if idxs.numel() > 0:
            early = idxs[: min(K, int(idxs.numel()))]
            early_mask[b, early] = True
    anti_immediate_stop_loss = p_stop[early_mask].mean() if early_mask.any() else p_stop.new_tensor(0.0)

    total_loss = expected_decision_loss
    total_loss = total_loss - float(stop_entropy_weight) * stop_entropy_mean
    total_loss = total_loss + float(stop_prior_weight) * stop_prior_loss
    total_loss = total_loss + float(anti_immediate_stop_weight) * anti_immediate_stop_loss

    expected_found_prob = 1.0 - p_no_response
    expected_rt_logged_ms = (p_first_stop * logged_rt_ms).sum(dim=-1)
    expected_rt_from_trial_onset_ms = (p_first_stop * decision_time_ms).sum(dim=-1)
    expected_rt_from_deviant_onset_ms = (
        p_first_stop * (decision_time_ms - (idx_info["deviant_onset_idx"].to(class_logits_all.device).float().unsqueeze(1) * float(token_ms)))
    ).sum(dim=-1)
    expected_rt_from_deviant_end_ms = (
        p_first_stop * (decision_time_ms - (idx_info["deviant_end_idx"].to(class_logits_all.device).float().unsqueeze(1) * float(token_ms)))
    ).sum(dim=-1)
    expected_correct_prob_at_stop = (p_first_stop * p_correct).sum(dim=-1)
    proportion_decisions_before_deviant_end = (
        p_first_stop
        * (decision_time_ms < (idx_info["deviant_end_idx"].to(class_logits_all.device).float().unsqueeze(1) * float(token_ms))).float()
    ).sum(dim=-1)
    proportion_negative_rt = (p_first_stop * (logged_rt_ms < 0).float()).sum(dim=-1)

    stats = {
        "stochastic_expected_stop_loss": float(total_loss.detach().item()),
        "expected_decision_loss": float(expected_decision_loss.detach().item()),
        "mean_p_stop": float(mean_p_stop.detach().item()),
        "stop_entropy": float(stop_entropy_mean.detach().item()),
        "no_response_prob_mean": float(p_no_response.mean().detach().item()),
        "expected_found_prob_mean": float(expected_found_prob.mean().detach().item()),
        "expected_rt_logged_ms": float(expected_rt_logged_ms.mean().detach().item()),
        "expected_rt_from_trial_onset_ms": float(expected_rt_from_trial_onset_ms.mean().detach().item()),
        "expected_rt_from_deviant_onset_ms": float(expected_rt_from_deviant_onset_ms.mean().detach().item()),
        "expected_rt_from_deviant_end_ms": float(expected_rt_from_deviant_end_ms.mean().detach().item()),
        "expected_correct_prob_at_stop": float(expected_correct_prob_at_stop.mean().detach().item()),
        "proportion_negative_rt": float(proportion_negative_rt.mean().detach().item()),
        "proportion_decisions_before_deviant_end": float(proportion_decisions_before_deviant_end.mean().detach().item()),
        "stop_prior_loss": float(stop_prior_loss.detach().item()),
        "anti_immediate_stop_loss": float(anti_immediate_stop_loss.detach().item()),
    }
    tensors = {
        "probs": probs.detach(),
        "p_correct": p_correct.detach(),
        "p_stop": p_stop.detach(),
        "p_first_stop": p_first_stop.detach(),
        "p_no_response": p_no_response.detach(),
        "rt_logged_ms": logged_rt_ms.detach(),
        "rt_from_trial_onset_ms": decision_time_ms.detach(),
        "rt_from_deviant_onset_ms": (decision_time_ms - (idx_info["deviant_onset_idx"].to(class_logits_all.device).float().unsqueeze(1) * float(token_ms))).detach(),
        "rt_from_deviant_end_ms": (decision_time_ms - (idx_info["deviant_end_idx"].to(class_logits_all.device).float().unsqueeze(1) * float(token_ms))).detach(),
        "decision_time_ms": decision_time_ms.detach(),
    }
    return total_loss, stats, tensors


@torch.no_grad()
def _sample_dynamic_debug_metrics(
    probs: torch.Tensor,             # (B,T,3)
    p_stop: torch.Tensor,            # (B,T)
    eligible_mask: torch.Tensor,     # (B,T)
    y_cls: torch.Tensor,             # (B,)
    rt_logged_ms: torch.Tensor,      # (B,T)
    rt_from_deviant_end_ms: torch.Tensor,  # (B,T)
    n_samples: int,
) -> Dict[str, Any]:
    B, T, _ = probs.shape
    found_count = 0
    correct_found_count = 0
    correct_all_count = 0
    rt_values: List[float] = []
    rt_de_list: List[float] = []
    pred_counts = torch.zeros(3, dtype=torch.long)

    for _ in range(int(n_samples)):
        for b in range(B):
            found = False
            pred_cls = -1
            for t in range(T):
                if not bool(eligible_mask[b, t].item()):
                    continue
                stop_draw = torch.bernoulli(p_stop[b, t]).item()
                if stop_draw >= 0.5:
                    pred_cls = int(probs[b, t].argmax().item())
                    pred_counts[pred_cls] += 1
                    found = True
                    found_count += 1
                    rt_values.append(float(rt_logged_ms[b, t].item()))
                    rt_de_list.append(float(rt_from_deviant_end_ms[b, t].item()))
                    if pred_cls == int(y_cls[b].item()):
                        correct_found_count += 1
                        correct_all_count += 1
                    break
            if not found:
                pred_cls = -1

    total = int(n_samples) * int(B)
    sampled_found_rate = found_count / max(1, total)
    sampled_acc_on_found = correct_found_count / max(1, found_count)
    sampled_acc_all_miss_wrong = correct_all_count / max(1, total)
    mean_rt = float(np.mean(rt_values)) if rt_values else float("nan")
    med_rt = float(np.median(rt_values)) if rt_values else float("nan")
    neg_prop = float(np.mean(np.array(rt_values) < 0)) if rt_values else float("nan")
    before_dev_end_prop = float(np.mean(np.array(rt_de_list) < 0)) if rt_de_list else float("nan")
    return {
        "sampled_found_rate": float(sampled_found_rate),
        "sampled_acc_on_found": float(sampled_acc_on_found),
        "sampled_acc_all_miss_wrong": float(sampled_acc_all_miss_wrong),
        "sampled_mean_rt_logged_ms": mean_rt,
        "sampled_median_rt_logged_ms": med_rt,
        "proportion_negative_rt": neg_prop,
        "proportion_decisions_before_deviant_end": before_dev_end_prop,
        "pred_counts_sampled": pred_counts.tolist(),
    }


@torch.no_grad()
def _proxy_pstop05_metrics(
    probs: torch.Tensor,             # (B,T,3)
    p_stop: torch.Tensor,            # (B,T)
    eligible_mask: torch.Tensor,     # (B,T)
    y_cls: torch.Tensor,             # (B,)
    rt_logged_ms: torch.Tensor,      # (B,T)
) -> Dict[str, float]:
    B, T, _ = probs.shape
    found = 0
    correct = 0
    rt_vals: List[float] = []
    for b in range(B):
        for t in range(T):
            if bool(eligible_mask[b, t].item()) and float(p_stop[b, t].item()) > 0.5:
                found += 1
                pred = int(probs[b, t].argmax().item())
                if pred == int(y_cls[b].item()):
                    correct += 1
                rt_vals.append(float(rt_logged_ms[b, t].item()))
                break
    total = int(B)
    return {
        "proxy_pstop05_found_rate": found / max(1, total),
        "proxy_pstop05_acc": correct / max(1, found) if found > 0 else float("nan"),
        "proxy_pstop05_mean_rt_logged_ms": float(np.mean(rt_vals)) if rt_vals else float("nan"),
    }


def run_trialwise_rendered_debug(args: argparse.Namespace, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(int(args.seed))
    device = resolve_device(args.device)
    print(f"[trialwise_debug] using device: {device}")

    block_ds, trial_ds, train_loader = build_trialwise_debug_loader(
        data_dir=Path(args.data_dir),
        seed=args.seed,
        tone_ms=args.tone_ms,
        isi_ms=int(args.isi_schedule[0]),
        ramp_ms=args.ramp_ms,
        token_ms=args.token_ms,
        f_min_hz=args.f_min_hz,
        f_max_hz=args.f_max_hz,
        n_bins=args.n_bins,
        add_eos=bool(args.add_eos),
        sigma_other_noise=float(args.sigma_other_noise),
        p_other_noise=float(args.p_other_noise),
        sigma_silence_noise=float(args.sigma_silence_noise),
        resample_noise_per_epoch=False,
        n_blocks=int(getattr(args, "debug_n_blocks", 16)),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        assert_labels=True,
    )
    print(
        f"[trialwise_debug data] blocks={block_ds.B} trials={len(trial_ds)} "
        f"trial_T_tokens={trial_ds.trial_T_tokens} input_dim={trial_ds.input_dim}"
    )
    raw_vals, raw_counts = torch.unique(trial_ds.Y[: int(block_ds.B)], return_counts=True)
    print("[trialwise_debug data] raw block-label unique:", list(zip(raw_vals.tolist(), raw_counts.tolist())))

    x_trials_hz = block_ds.X[: int(block_ds.B)].reshape(-1, 8).float()
    y_trials_raw = block_ds.Y[: int(block_ds.B)].reshape(-1).long()
    y_trials_cls = labels_to_class_index(y_trials_raw)
    rule_pred = _compute_trial_rule_predictions_raw(x_trials_hz)
    valid_rule = rule_pred >= 0
    rule_acc = float((rule_pred[valid_rule] == y_trials_cls[valid_rule]).float().mean().item()) if valid_rule.any() else float("nan")
    print(
        f"[trialwise_debug rule] rule_acc={rule_acc:.4f} "
        f"invalid_count={int((~valid_rule).sum().item())} "
        f"first20_rule={rule_pred[:20].tolist()} first20_labels={y_trials_cls[:20].tolist()}"
    )

    all_x_trials: List[torch.Tensor] = []
    all_y_trials: List[torch.Tensor] = []
    for i in range(len(trial_ds)):
        x_trial, y_trial = trial_ds[i]
        all_x_trials.append(x_trial)
        all_y_trials.append(y_trial)
    x_all = torch.stack(all_x_trials, dim=0).to(device)               # (N, trial_T, D)
    y_all_raw = torch.stack(all_y_trials, dim=0).long().to(device)    # (N,)
    y_all_cls = labels_to_class_index(y_all_raw)
    assert int(y_all_cls.min().item()) >= 0
    assert int(y_all_cls.max().item()) <= 2

    fixed_bs = min(int(args.batch_size), int(x_all.shape[0]))
    x_fixed = x_all[:fixed_bs]
    y_fixed_raw = y_all_raw[:fixed_bs]
    y_fixed_cls = y_all_cls[:fixed_bs]
    print(
        f"[trialwise_debug fixed_batch] x_fixed={tuple(x_fixed.shape)} y_fixed={tuple(y_fixed_cls.shape)} "
        f"fixed_batch_size={fixed_bs}"
    )

    tone_tokens = int(args.tone_ms // args.token_ms)
    isi_tokens = int(args.isi_ms if hasattr(args, "isi_ms") else args.isi_schedule[0]) // int(args.token_ms)
    if int(isi_tokens) <= 0:
        isi_tokens = int(args.isi_schedule[0] // args.token_ms)
    idx_fixed = _compute_rendered_trial_indices(
        y_raw=y_fixed_raw,
        T=int(x_fixed.shape[1]),
        tone_tokens=int(tone_tokens),
        isi_tokens=int(isi_tokens),
        post_window_tokens=int(getattr(args, "debug_post_deviant_window_tokens", 20)),
    )
    print("[trialwise_debug timing]")
    print("T", int(x_fixed.shape[1]))
    print("tone_tokens", int(tone_tokens))
    print("isi_tokens", int(isi_tokens))
    print("deviant_pos first 20", idx_fixed["deviant_pos"][:20].detach().cpu().tolist())
    print("deviant_onset_idx first 20", idx_fixed["deviant_onset_idx"][:20].detach().cpu().tolist())
    print("deviant_end_idx first 20", idx_fixed["deviant_end_idx"][:20].detach().cpu().tolist())
    print("trial_end_idx", int(idx_fixed["trial_end_idx"][0].item()))
    print("readout_mode", "stochastic_dynamic" if bool(getattr(args, "debug_stochastic_dynamic_readout", False)) else "fixed")
    print("add_eos_shared_last_token", bool(args.add_eos))
    print("rt_logging_reference", str(getattr(args, "debug_rt_logging_reference", "deviant_end")))
    print("cost_reference", str(getattr(args, "debug_cost_reference", "deviant_onset")))
    print("clamp_negative_cost_time", bool(getattr(args, "debug_clamp_negative_cost_time", True)))

    cfg = ModelConfig(
        input_dim=int(block_ds.input_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        layer_norm=(False if bool(getattr(args, "debug_disable_layer_norm", False)) else bool(args.layer_norm)),
        hidden_noise_std=float(getattr(args, "hidden_noise_std", 0.0)),
        use_stop_head=bool(getattr(args, "debug_stochastic_dynamic_readout", False)),
    )
    model = PredictiveGRU(cfg).to(device)
    extra_modules: List[nn.Module] = []
    param_groups = list(model.parameters())
    optim = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))

    first_batch_done = False
    step = 0
    best_acc = -1.0
    best_loss = float("inf")
    final_loss = float("nan")
    final_metrics: Dict[str, Any] = {}

    while step < int(getattr(args, "debug_max_steps", 500)):
        step += 1
        x_trial = x_fixed
        y_raw = y_fixed_raw
        y_cls = y_fixed_cls

        optim.zero_grad(set_to_none=True)
        h_seq, _ = model.forward_chunk(x_trial, h0=None)

        if bool(getattr(args, "debug_stochastic_dynamic_readout", False)):
            class_logits_all = model.classify_tokens(h_seq)                # (B,T,3)
            stop_logits_all = model.classify_stop(h_seq)
            if stop_logits_all is None:
                raise RuntimeError("debug_stochastic_dynamic_readout requires stop head")
            eligible_mask = _make_dynamic_eligible_mask(
                idx_info=idx_fixed,
                T=int(x_trial.shape[1]),
                start_mode=str(getattr(args, "debug_dynamic_start", "deviant_onset")),
                end_mode=str(getattr(args, "debug_dynamic_end", "trial_end")),
                device=x_trial.device,
            )
            stochastic_loss, stats, tensors = _compute_expected_stop_loss_debug(
                class_logits_all=class_logits_all,
                stop_logits_all=stop_logits_all,
                y_cls=y_cls,
                eligible_mask=eligible_mask,
                idx_info=idx_fixed,
                token_ms=int(args.token_ms),
                class_temperature=float(getattr(args, "debug_class_temperature", 1.0)),
                stop_temperature=float(getattr(args, "debug_stop_temperature", 1.0)),
                time_cost_w=float(getattr(args, "debug_time_cost_w", 0.001)),
                wrong_cost=float(getattr(args, "debug_wrong_cost", 1.0)),
                no_response_cost=float(getattr(args, "debug_no_response_cost", 1.0)),
                stop_entropy_weight=float(getattr(args, "debug_stop_entropy_weight", 0.01)),
                stop_prior_weight=float(getattr(args, "debug_stop_prior_weight", 0.01)),
                stop_prior_target=float(getattr(args, "debug_stop_prior_target", 0.05)),
                anti_immediate_stop_tokens=int(getattr(args, "debug_anti_immediate_stop_tokens", 5)),
                anti_immediate_stop_weight=float(getattr(args, "debug_anti_immediate_stop_weight", 0.1)),
                rt_logging_reference=str(getattr(args, "debug_rt_logging_reference", "deviant_end")),
                cost_reference=str(getattr(args, "debug_cost_reference", "deviant_onset")),
                clamp_negative_cost_time=bool(getattr(args, "debug_clamp_negative_cost_time", True)),
            )
            aux_mask = _make_dynamic_eligible_mask(
                idx_info=idx_fixed,
                T=int(x_trial.shape[1]),
                start_mode=str(getattr(args, "debug_aux_token_ce_start", "deviant_end")),
                end_mode=str(getattr(args, "debug_aux_token_ce_end", "trial_end")),
                device=x_trial.device,
            )
            aux_token_ce_loss = _compute_aux_token_ce_debug(
                class_logits_all=class_logits_all,
                y_cls=y_cls,
                mask=aux_mask,
            )
            aux_w = float(getattr(args, "debug_aux_token_ce_weight", 0.0))
            loss = stochastic_loss + (aux_w * aux_token_ce_loss)
            stats["aux_token_ce_loss"] = float(aux_token_ce_loss.detach().item())
            stats["weighted_aux_token_ce_loss"] = float((aux_w * aux_token_ce_loss).detach().item())
            stats["total_loss"] = float(loss.detach().item())
            if bool(getattr(args, "debug_loss_check", False)):
                actual_backward_loss = float(loss.detach().item())
                reconstructed_loss = float(
                    stochastic_loss.detach().item()
                    + aux_w * aux_token_ce_loss.detach().item()
                )
                diff = abs(actual_backward_loss - reconstructed_loss)
                print("[DEBUG trialwise loss tensor]")
                print("actual_backward_loss", actual_backward_loss)
                print("reconstructed_loss", reconstructed_loss)
                print("diff", diff)
                print("stochastic_expected_stop_loss", float(stochastic_loss.detach().item()))
                print("aux_token_ce_loss", float(aux_token_ce_loss.detach().item()))
                print("weighted_aux_token_ce_loss", float((aux_w * aux_token_ce_loss).detach().item()))
                print("total_loss", float(loss.detach().item()))
                assert diff < 1e-5
            loss.backward()

            grad_norm_gru = _grad_norm_from_named_params(model, "gru")
            grad_norm_head = _grad_norm_from_named_params(model, "head")
            last_head_before = model.head.weight.detach().clone()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optim.step()
            parameter_delta_last_layer = float((model.head.weight.detach() - last_head_before).norm().item())

            with torch.no_grad():
                fixed_accs = _compute_fixed_readout_accs(model, h_seq, idx_fixed, y_cls)
                probs = tensors["probs"]
                p_stop = tensors["p_stop"]
                sampled = _sample_dynamic_debug_metrics(
                    probs=probs,
                    p_stop=p_stop,
                    eligible_mask=eligible_mask,
                    y_cls=y_cls,
                    rt_logged_ms=tensors["rt_logged_ms"],
                    rt_from_deviant_end_ms=tensors["rt_from_deviant_end_ms"],
                    n_samples=int(getattr(args, "debug_n_stop_samples", 50)),
                )
                proxy = _proxy_pstop05_metrics(
                    probs=probs,
                    p_stop=p_stop,
                    eligible_mask=eligible_mask,
                    y_cls=y_cls,
                    rt_logged_ms=tensors["rt_logged_ms"],
                )
                current_acc = sampled["sampled_acc_all_miss_wrong"]
                best_acc = max(best_acc, current_acc)
                best_loss = min(best_loss, float(loss.item()))
                final_loss = float(loss.item())
                final_metrics = {**stats, **sampled, **proxy, **fixed_accs}

            if not first_batch_done:
                uniq_raw, cnt_raw = torch.unique(y_raw.detach().cpu(), return_counts=True)
                uniq_map, cnt_map = torch.unique(y_cls.detach().cpu(), return_counts=True)
                print("[DEBUG stochastic dynamic first batch]")
                print("input shape", tuple(x_trial.shape))
                print("labels shape", tuple(y_raw.shape))
                print("unique raw labels", list(zip(uniq_raw.tolist(), cnt_raw.tolist())))
                print("unique mapped labels", list(zip(uniq_map.tolist(), cnt_map.tolist())))
                print("class_logits_all shape", tuple(class_logits_all.shape))
                print("stop_logits_all shape", tuple(stop_logits_all.shape))
                print("eligible_count_per_trial first 10", eligible_mask.sum(dim=1)[:10].detach().cpu().tolist())
                first_batch_done = True

            if step == 1 or step % int(getattr(args, "debug_trialwise_print_every", 25)) == 0:
                print(
                    f"[trialwise_dynamic step {step:04d}] "
                    f"loss={float(loss.item()):.6f} "
                    f"stochastic_expected_stop_loss={stats['stochastic_expected_stop_loss']:.6f} "
                    f"expected_decision_loss={stats['expected_decision_loss']:.6f} "
                    f"aux_token_ce_loss={stats['aux_token_ce_loss']:.6f} "
                    f"weighted_aux_token_ce_loss={stats['weighted_aux_token_ce_loss']:.6f} "
                    f"mean_p_stop={stats['mean_p_stop']:.6f} "
                    f"stop_entropy={stats['stop_entropy']:.6f} "
                    f"no_response_prob_mean={stats['no_response_prob_mean']:.6f} "
                    f"expected_found_prob_mean={stats['expected_found_prob_mean']:.6f} "
                    f"expected_rt_logged_ms={stats['expected_rt_logged_ms']:.6f} "
                    f"expected_rt_from_trial_onset_ms={stats['expected_rt_from_trial_onset_ms']:.6f} "
                    f"expected_rt_from_deviant_onset_ms={stats['expected_rt_from_deviant_onset_ms']:.6f} "
                    f"expected_rt_from_deviant_end_ms={stats['expected_rt_from_deviant_end_ms']:.6f} "
                    f"expected_correct_prob_at_stop={stats['expected_correct_prob_at_stop']:.6f} "
                    f"sampled_found_rate={sampled['sampled_found_rate']:.6f} "
                    f"sampled_acc_on_found={sampled['sampled_acc_on_found']:.6f} "
                    f"sampled_acc_all_miss_wrong={sampled['sampled_acc_all_miss_wrong']:.6f} "
                    f"sampled_mean_rt_logged_ms={sampled['sampled_mean_rt_logged_ms']:.6f} "
                    f"sampled_median_rt_logged_ms={sampled['sampled_median_rt_logged_ms']:.6f} "
                    f"proportion_negative_rt={sampled['proportion_negative_rt']:.6f} "
                    f"proportion_decisions_before_deviant_end={sampled['proportion_decisions_before_deviant_end']:.6f} "
                    f"proxy_pstop05_found_rate={proxy['proxy_pstop05_found_rate']:.6f} "
                    f"proxy_pstop05_acc={proxy['proxy_pstop05_acc']:.6f} "
                    f"last_token_acc={fixed_accs['last_token_acc']:.6f} "
                    f"deviant_end_acc={fixed_accs['deviant_end_acc']:.6f} "
                    f"post_deviant_mean_acc={fixed_accs['post_deviant_mean_acc']:.6f} "
                    f"grad_norm_gru={grad_norm_gru:.6f} grad_norm_head={grad_norm_head:.6f} "
                    f"param_delta_head={parameter_delta_last_layer:.6f} "
                    f"pred_counts_sampled={sampled['pred_counts_sampled']} "
                    f"logits_mean={float(class_logits_all.mean().item()):.6f} logits_std={float(class_logits_all.std().item()):.6f}"
                )
        else:
            final_hidden = h_seq[:, -1, :]
            logits = model.classify_from_states(final_hidden)       # (B,3)
            loss = F.cross_entropy(logits, y_cls)
            if bool(getattr(args, "debug_loss_check", False)):
                actual_backward_loss = float(loss.detach().item())
                reconstructed_loss = float(loss.detach().item())
                diff = abs(actual_backward_loss - reconstructed_loss)
                print("[DEBUG trialwise loss tensor]")
                print("actual_backward_loss", actual_backward_loss)
                print("reconstructed_loss", reconstructed_loss)
                print("diff", diff)
                print("end_loss", float(loss.detach().item()))
                assert diff < 1e-5
            loss.backward()
            grad_norm_gru = _grad_norm_from_named_params(model, "gru")
            grad_norm_head = _grad_norm_from_named_params(model, "head")
            last_head_before = model.head.weight.detach().clone()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optim.step()
            parameter_delta_last_layer = float((model.head.weight.detach() - last_head_before).norm().item())
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                pred = probs.argmax(dim=-1)
                fixed_batch_acc = float((pred == y_cls).float().mean().item())
                best_acc = max(best_acc, fixed_batch_acc)
                best_loss = min(best_loss, float(loss.item()))
                final_loss = float(loss.item())
                h_all, _ = model.forward_chunk(x_all, h0=None)
                logits_all = model.classify_from_states(h_all[:, -1, :])
                all_probs = torch.softmax(logits_all, dim=-1)
                all_pred = all_probs.argmax(dim=-1)
                all_trials_acc = float((all_pred == y_all_cls).float().mean().item())
                final_metrics = {"all_trials_acc": all_trials_acc}
            if not first_batch_done:
                uniq_raw, cnt_raw = torch.unique(y_raw.detach().cpu(), return_counts=True)
                uniq_map, cnt_map = torch.unique(y_cls.detach().cpu(), return_counts=True)
                uniq_pred, cnt_pred = torch.unique(pred.detach().cpu(), return_counts=True)
                print("[DEBUG trialwise first batch]")
                print("input shape", tuple(x_trial.shape))
                print("labels shape", tuple(y_raw.shape))
                print("unique raw labels", list(zip(uniq_raw.tolist(), cnt_raw.tolist())))
                print("unique mapped labels", list(zip(uniq_map.tolist(), cnt_map.tolist())))
                print("final_hidden shape", tuple(final_hidden.shape))
                print("final_logits shape", tuple(logits.shape))
                print("final_hidden mean/std", float(final_hidden.mean().item()), float(final_hidden.std().item()))
                print("logits mean/std", float(logits.mean().item()), float(logits.std().item()))
                print("pred unique/counts", list(zip(uniq_pred.tolist(), cnt_pred.tolist())))
                print("first 20 raw labels", y_raw[:20].detach().cpu().tolist())
                print("first 20 mapped labels", y_cls[:20].detach().cpu().tolist())
                print("first 20 preds", pred[:20].detach().cpu().tolist())
                print("first 5 probs", probs[:5].detach().cpu().tolist())
                first_batch_done = True
            if step == 1 or step % int(getattr(args, "debug_trialwise_print_every", 25)) == 0:
                uniq_pred, cnt_pred = torch.unique(pred.detach().cpu(), return_counts=True)
                uniq_all_pred, cnt_all_pred = torch.unique(all_pred.detach().cpu(), return_counts=True)
                print(
                    f"[trialwise_debug step {step:04d}] "
                    f"loss={float(loss.item()):.6f} fixed_batch_acc={fixed_batch_acc:.4f} "
                    f"all_trials_acc={all_trials_acc:.4f} best_fixed_batch_acc={best_acc:.4f} "
                    f"grad_norm_gru={grad_norm_gru:.6f} grad_norm_head={grad_norm_head:.6f} "
                    f"param_delta_head={parameter_delta_last_layer:.6f} "
                    f"fixed_pred_counts={list(zip(uniq_pred.tolist(), cnt_pred.tolist()))} "
                    f"all_pred_counts={list(zip(uniq_all_pred.tolist(), cnt_all_pred.tolist()))}"
                )

    if bool(getattr(args, "debug_stochastic_dynamic_readout", False)):
        print(
            "[stochastic_dynamic_readout done] "
            f"best_metric_acc={best_acc:.6f} final_loss={final_loss:.6f} best_loss={best_loss:.6f} "
            f"final_metrics={json.dumps(final_metrics, ensure_ascii=False)}"
        )
    else:
        print(
            f"[trialwise_debug done] steps={step} best_fixed_batch_acc={best_acc:.4f} "
            f"final_all_trials_acc={final_metrics.get('all_trials_acc', float('nan')):.4f}"
        )


# ====== COPY-PASTE PATCH (only modify run_curriculum_training) ======
# 目标：同一个 condition 内，ISI0 -> ISI50 -> ISI300 -> ISI700 依次接着训练；
# early stopping 只在“当前 stage”内生效，触发后进入下一 stage（不 return，不重置模型）

def run_curriculum_training(
    args: argparse.Namespace,
    run_dir: Path,
    sweep_info: Dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    config = {
        "run_dir": str(run_dir.resolve()),
        "sweep": sweep_info,
        "args": vars(args),
        "have_sklearn": _HAVE_SKLEARN,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    set_all_seeds(int(args.seed))

    device = resolve_device(args.device)
    print(f"[device] using: {device}")

    base_ds = OnlineRenderDataset(
        data_dir=Path(args.data_dir),
        seed=args.seed,
        tone_ms=args.tone_ms,
        isi_ms=int(args.isi_schedule[0]),
        ramp_ms=args.ramp_ms,
        token_ms=args.token_ms,
        f_min_hz=args.f_min_hz,
        f_max_hz=args.f_max_hz,
        n_bins=args.n_bins,
        add_eos=bool(args.add_eos),
        sigma_other_noise=float(args.sigma_other_noise),
        p_other_noise=float(args.p_other_noise),
        sigma_silence_noise=float(args.sigma_silence_noise),
        resample_noise_per_epoch=bool(getattr(args, "resample_noise_per_epoch", False)),
        quiet=False,
    )

    if getattr(args, "max_blocks", 0) and int(args.max_blocks) > 0:
        m = min(int(args.max_blocks), int(base_ds.B))
        base_ds.X = base_ds.X[:m]
        base_ds.Y = base_ds.Y[:m]
        base_ds.B = m
        print(f"[debug] max_blocks applied: B={base_ds.B}")

    vals, counts = torch.unique(base_ds.Y, return_counts=True)
    print("[data] unique Y values:", list(zip(vals.tolist(), counts.tolist())))
    if bool(getattr(args, "debug_overfit_tiny", False)):
        mapped_vals = torch.unique(labels_to_class_index(base_ds.Y))
        print("[debug_overfit_tiny labels]")
        print("raw labels unique values", vals.tolist())
        print("mapped labels unique values", mapped_vals.tolist())

    train_idx, val_idx = split_indices(len(base_ds), args.val_split, args.seed)
    if bool(getattr(args, "debug_disable_val", False)):
        train_idx = list(range(len(base_ds)))
        val_idx = []

    cfg = ModelConfig(
        input_dim=int(base_ds.input_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        layer_norm=bool(args.layer_norm),
        hidden_noise_std=float(getattr(args, "hidden_noise_std", 0.0)),
        use_stop_head=bool(getattr(args, "use_stop_head", False)),
    )
    model = PredictiveGRU(cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    start_epoch_global = 1
    best_val = float("inf")
    best_epoch = 0
    
    # 1) init_from: weights only, for cross-condition finetuning / sweep
    if getattr(args, "init_from", ""):
        ckpt = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        print(f"[init_from] loaded model weights only: {args.init_from}")
        print("[init_from] optimizer / epoch / best_val are RESET for a new run.")
    
    # 2) resume: full resume, only for continuing the same run
    elif getattr(args, "resume", ""):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "optim_state" in ckpt:
            optim.load_state_dict(ckpt["optim_state"])
        start_epoch_global = int(ckpt.get("epoch_global", ckpt.get("epoch", 0))) + 1
        best_val = float(ckpt.get("best_val", best_val))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        print(f"[resume] loaded full state: {args.resume} start_epoch_global={start_epoch_global}")

    jsonl_path = run_dir / "logs" / "metrics.jsonl"
    csv_path = run_dir / "logs" / "metrics.csv"
    header = [
        "epoch_global", "stage", "isi_ms", "token_ms",
        "train_total_loss", "train_end_loss", "train_token_loss", "train_anti_commit_loss", "train_acc", "train_f1_macro", "train_auc_ovr",
        "train_mean_rt_tokens", "train_mean_rt_ms", "train_rt_found", "train_rt_miss",
        "train_rt_not_first", "train_rt_not_first_rate",
        "train_online_decision_cost",
        "train_online_decision_loss", "train_online_ce_loss",
        "train_mean_p_stop", "train_mean_no_response_prob", "train_mean_expected_decision_time_ms",
        "train_expected_found_prob", "train_expected_rt_logged_ms",
        "train_expected_rt_from_deviant_onset_ms", "train_expected_rt_from_deviant_end_ms",
        "train_proportion_negative_rt", "train_proportion_decisions_before_deviant_end",
        "train_phase_name", "train_effective_online_loss_weight",
        "train_weighted_token_loss", "train_weighted_online_decision_loss", "train_weighted_online_ce_loss",
        "train_aux_token_ce_loss", "train_weighted_aux_token_ce_loss",
        "train_anti_immediate_stop_loss", "train_stop_entropy_bonus", "train_stop_prior_loss",
        "train_sampled_mean_rt_logged_ms", "train_last_token_acc", "train_deviant_end_acc",
        "val_total_loss", "val_end_loss", "val_token_loss", "val_anti_commit_loss", "val_acc", "val_f1_macro", "val_auc_ovr",
        "val_mean_rt_tokens", "val_mean_rt_ms", "val_rt_found", "val_rt_miss",
        "val_rt_not_first", "val_rt_not_first_rate",
        "val_online_decision_cost",
        "val_online_decision_loss", "val_online_ce_loss",
        "val_mean_p_stop", "val_mean_no_response_prob", "val_mean_expected_decision_time_ms",
        "val_expected_found_prob", "val_expected_rt_logged_ms",
        "val_expected_rt_from_deviant_onset_ms", "val_expected_rt_from_deviant_end_ms",
        "val_proportion_negative_rt", "val_proportion_decisions_before_deviant_end",
        "val_phase_name", "val_effective_online_loss_weight",
        "val_weighted_token_loss", "val_weighted_online_decision_loss", "val_weighted_online_ce_loss",
        "val_aux_token_ce_loss", "val_weighted_aux_token_ce_loss",
        "val_anti_immediate_stop_loss", "val_stop_entropy_bonus", "val_stop_prior_loss",
        "val_sampled_mean_rt_logged_ms", "val_last_token_acc", "val_deviant_end_acc",
        "best_val", "best_epoch",
        "best_target_isi_val", "best_target_isi_epoch", "best_target_isi",
        "time_elapsed_sec",
    ]

    patience = int(getattr(args, "early_stop_patience", 0))
    min_delta = float(getattr(args, "early_stop_min_delta", 0.0))

    # 新增：stage性能阈值参数
    stage_perf_threshold = float(getattr(args, "stage_perf_threshold", 0.0))
    min_stage_epochs = int(getattr(args, "min_stage_epochs", 10))

    best_target_isi = int(getattr(args, "analysis_isi", 700))
    best_target_val = float("inf")
    best_target_epoch = 0
    best_target_path = run_dir / f"best_isi{best_target_isi}.pt"

    history: List[Dict[str, Any]] = []
    t_run0 = time.time()
    epoch_global = start_epoch_global - 1
    optimizer_step_state = {"count": 0}
    first_batch_debug_done = False

    # 解析有意义token的位置（如果有）
    important_token_indices = None
    if hasattr(args, "important_token_indices") and args.important_token_indices:
        important_token_indices = parse_list_of_ints(args.important_token_indices)
        print(f"[mask_loss] important token indices: {important_token_indices}")

    for stage, isi_ms in enumerate(args.isi_schedule, start=1):
        isi_ms = int(isi_ms)
        print(f"\n[curriculum] stage={stage}/{len(args.isi_schedule)}  isi_ms={isi_ms}")
        is_last_stage = (stage == len(args.isi_schedule))
        force_full_last_stage = bool(getattr(args, "force_full_last_stage", False))
        stage_threshold_active = bool(stage_perf_threshold > 0 and (not is_last_stage or not force_full_last_stage))
        stage_patience_active = bool(patience > 0 and (not is_last_stage or not force_full_last_stage))
        if bool(getattr(args, "debug_overfit_tiny", False)):
            stage_threshold_active = False
            stage_patience_active = False
        
        # 记录stage开始的epoch
        stage_start_epoch = epoch_global + 1

        stage_best_val = float("inf")
        stage_bad_count = 0
        stage_best_state: Optional[Dict[str, torch.Tensor]] = None
        stage_best_optim: Optional[Dict[str, Any]] = None
        stage_best_epoch_global: Optional[int] = None
        stage_best_path = run_dir / f"best_isi{int(isi_ms)}.pt"
        stage_reached_threshold = False  # 标记是否达到阈值

        ds, train_loader, val_loader = build_loaders_for_isi(
            data_dir=Path(args.data_dir),
            seed=args.seed,
            tone_ms=args.tone_ms,
            isi_ms=isi_ms,
            ramp_ms=args.ramp_ms,
            token_ms=args.token_ms,
            f_min_hz=args.f_min_hz,
            f_max_hz=args.f_max_hz,
            n_bins=args.n_bins,
            add_eos=bool(args.add_eos),
            sigma_other_noise=float(args.sigma_other_noise),
            p_other_noise=float(args.p_other_noise),
            sigma_silence_noise=float(args.sigma_silence_noise),
            resample_noise_per_epoch=bool(getattr(args, "resample_noise_per_epoch", False)),
            train_idx=train_idx,
            val_idx=val_idx,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            assert_labels=bool(getattr(args, "debug_dataset_assert", False)),
        )

        end_preview = infer_end_indices_from_T(int(ds.T), trials_per_block=10)
        print(
            f"[data@isi={isi_ms}] token_ms={ds.token_ms} trial_T_ms={ds.trial_T_ms} "
            f"trial_T_tokens={ds.trial_T_tokens} T={ds.T} end_idx preview: "
            f"{end_preview[:3].tolist()} ... {end_preview[-3:].tolist()}"
        )

        for _e in range(1, int(args.epochs_per_isi) + 1):
            if bool(getattr(args, "two_phase_training", False)):
                warm_ep = int(getattr(args, "classifier_warmup_epochs", 0))
                sf_ep = getattr(args, "stop_finetune_epochs", None)
                if sf_ep is not None and _e > (warm_ep + int(sf_ep)):
                    break
            epoch_global += 1
            _maybe_set_epoch_on_dataset(train_loader.dataset, epoch_global)
            _maybe_set_epoch_on_dataset(val_loader.dataset, 0)

            # two-phase schedule per-ISI stage (preferred option A)
            if bool(getattr(args, "two_phase_training", False)):
                warm_ep = int(getattr(args, "classifier_warmup_epochs", 0))
                in_warmup = (_e <= warm_ep)
                phase_name = "classifier_warmup" if in_warmup else "stop_finetune"
                phase_online_decision = (not in_warmup) and bool(getattr(args, "finetune_enable_stochastic_cost", True))
                phase_online_ce = bool(getattr(args, "warmup_use_online_ce", True) if in_warmup else True)
                phase_online_ce_weight = float(getattr(args, "warmup_online_ce_weight", 0.1) if in_warmup else getattr(args, "online_ce_weight", 0.1))
                phase_use_stop_head = bool(getattr(args, "warmup_enable_stop_head", False) if in_warmup else getattr(args, "use_stop_head", False))
                phase_decision_cost_mode = "masked_online_ce" if in_warmup else str(getattr(args, "decision_cost_mode", "stochastic_expected_cost"))
                phase_anti_immediate = bool((not in_warmup) and getattr(args, "anti_immediate_stop", False))
                phase_stop_entropy_w = float(0.0 if in_warmup else getattr(args, "stop_entropy_weight", 0.0))
                phase_stop_prior_w = float(0.0 if in_warmup else getattr(args, "stop_prior_weight", 0.0))
                # lr schedule
                if in_warmup:
                    warmup_lr_arg = getattr(args, "warmup_lr", None)
                    phase_lr = float(args.lr) if warmup_lr_arg is None else float(warmup_lr_arg)
                else:
                    finetune_lr_arg = getattr(args, "finetune_lr", 1e-4)
                    phase_lr = float(finetune_lr_arg)
                for pg in optim.param_groups:
                    pg["lr"] = phase_lr
            else:
                phase_name = "single_phase"
                phase_online_decision = bool(getattr(args, "online_decision_training", False))
                phase_online_ce = True
                phase_online_ce_weight = float(getattr(args, "online_ce_weight", 0.1))
                phase_use_stop_head = bool(getattr(args, "use_stop_head", False))
                phase_decision_cost_mode = str(getattr(args, "decision_cost_mode", "expected_cost_softmin"))
                phase_anti_immediate = bool(getattr(args, "anti_immediate_stop", False))
                phase_stop_entropy_w = float(getattr(args, "stop_entropy_weight", 0.0))
                phase_stop_prior_w = float(getattr(args, "stop_prior_weight", 0.0))
                phase_lr = float(getattr(args, "lr", 3e-4))

            if bool(getattr(args, "debug_end_loss_only", False)):
                phase_online_decision = False
                phase_online_ce = False
                phase_online_ce_weight = 0.0
                phase_use_stop_head = False
                phase_anti_immediate = False
                phase_stop_entropy_w = 0.0
                phase_stop_prior_w = 0.0

            stage_epoch_1based = int(_e)
            anti_schedule_epochs = int(getattr(args, "anti_commit_schedule_epochs", 0))
            lambda_anti_curr = scheduled_value(
                start_value=float(getattr(args, "lambda_anti_commit", 0.0)),
                final_value=float(getattr(args, "lambda_anti_commit_final", -1.0)),
                epoch_in_stage_1based=stage_epoch_1based,
                schedule_epochs=anti_schedule_epochs,
            )
            anti_window_curr = int(round(scheduled_value(
                start_value=float(getattr(args, "anti_commit_window_ms", 0)),
                final_value=float(getattr(args, "anti_commit_window_ms_final", -1)),
                epoch_in_stage_1based=stage_epoch_1based,
                schedule_epochs=anti_schedule_epochs,
            )))
            anti_conf_curr = scheduled_value(
                start_value=float(getattr(args, "anti_commit_max_conf", 1.0)),
                final_value=float(getattr(args, "anti_commit_max_conf_final", -1.0)),
                epoch_in_stage_1based=stage_epoch_1based,
                schedule_epochs=anti_schedule_epochs,
            )
            tok_window_curr = int(getattr(args, "tok_window_ms", 0))
            if bool(getattr(args, "train_online_decision", False)):
                tok_window_curr = int(ds.trial_T_ms)

            tr = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optim,
                device=device,
                chunk_len=int(args.chunk_len),
                lambda_token=float(args.lambda_token),
                grad_clip=float(args.grad_clip),
                trial_T_tokens=int(ds.trial_T_tokens),
                tone_T=int(ds.tone_T),
                isi_T=int(ds.isi_T),
                token_ms=int(ds.token_ms),
                tok_window_ms=int(tok_window_curr),
                tok_start_offset_ms=int(getattr(args, "tok_start_offset_ms", 0)),
                lambda_anti_commit=float(lambda_anti_curr),
                anti_commit_window_ms=int(anti_window_curr),
                anti_commit_start_offset_ms=int(getattr(args, "anti_commit_start_offset_ms", 0)),
                anti_commit_max_conf=float(anti_conf_curr),
                rt_p_thresh=float(args.rt_p_thresh),
                rt_k_consec=int(args.rt_k_consec),
                rt_mode=str(getattr(args, "rt_mode", "entropy")),
                rt_entropy_thresh=float(getattr(args, "rt_entropy_thresh", 0.35)),
                min_rt_tokens=int(getattr(args, "min_rt_tokens", 0)),
                debug=bool(getattr(args, "debug", False)),
                debug_steps=int(getattr(args, "debug_steps", 0)),
                log_every=int(args.log_every),
                token_loss_mode=str(args.token_loss_mode),
                token_tau=float(args.token_tau),
                token_w_min=float(args.token_w_min),
                tok_anchor=str(getattr(args, "tok_anchor", "deviant_end")),
                debug_labels=bool(getattr(args, "debug_labels", False)),
                debug_labels_fatal=bool(getattr(args, "debug_labels_fatal", False)),
                debug_labels_dump=bool(getattr(args, "debug_labels_dump", False)),
                debug_labels_dump_dir=(run_dir / "logs" / "bad_batches"),
                debug_labels_max_dumps=int(getattr(args, "debug_labels_max_dumps", 10)),
                debug_labels_first_n_batches=int(getattr(args, "debug_labels_first_n_batches", 0)),
                epoch_global=int(epoch_global),
                use_mask_loss=bool(getattr(args, "use_mask_loss", False)),
                important_token_indices=important_token_indices,
                gap_weight_power=float(getattr(args, "gap_weight_power", 0.0)),
                gap_weight_ref_hz=float(getattr(args, "gap_weight_ref_hz", 25.0)),
                gap_weight_max=float(getattr(args, "gap_weight_max", 2.5)),
                online_cost_monitor=bool(getattr(args, "online_cost_monitor", False)),
                online_cost_anchor=str(getattr(args, "online_cost_anchor", "deviant_onset")),
                online_cost_wrong=float(getattr(args, "online_cost_wrong", 1.0)),
                online_cost_w_time=float(getattr(args, "online_cost_w_time", 0.001)),
                online_decision_training=bool(phase_online_decision),
                online_loss_weight=float(getattr(args, "online_loss_weight", 0.5)),
                online_ce_weight=float(phase_online_ce_weight),
                decision_cost_mode=str(phase_decision_cost_mode),
                timeout_ms=(None if getattr(args, "timeout_ms", None) is None else float(getattr(args, "timeout_ms"))),
                time_cost_w=(None if getattr(args, "time_cost_w", None) is None else float(getattr(args, "time_cost_w"))),
                derive_time_cost_from_timeout=bool(getattr(args, "derive_time_cost_from_timeout", False)),
                decision_reference=str(getattr(args, "decision_reference", "trial_onset")),
                rt_logging_reference=str(getattr(args, "rt_logging_reference", "deviant_end")),
                cost_reference=str(getattr(args, "cost_reference", "deviant_onset")),
                clamp_negative_cost_time=bool(getattr(args, "clamp_negative_cost_time", True)),
                online_supervision_start=str(getattr(args, "online_supervision_start", "deviant_start")),
                online_supervision_end=str(getattr(args, "online_supervision_end", "trial_end")),
                aux_token_ce_weight=float(getattr(args, "aux_token_ce_weight", 0.0)),
                aux_token_ce_start=str(getattr(args, "aux_token_ce_start", "deviant_onset")),
                aux_token_ce_end=str(getattr(args, "aux_token_ce_end", "trial_end")),
                decision_softmin_tau=float(getattr(args, "decision_softmin_tau", 0.05)),
                use_stop_head=bool(phase_use_stop_head),
                decision_policy=str(getattr(args, "decision_policy", "deterministic_threshold")),
                sampling_temperature=float(getattr(args, "sampling_temperature", 1.0)),
                stop_temperature=float(getattr(args, "stop_temperature", 1.0)),
                use_hazard_prior=bool(getattr(args, "use_hazard_prior", False)),
                hazard_prior_weight=float(getattr(args, "hazard_prior_weight", 1.0)),
                hazard_prior_mode=str(getattr(args, "hazard_prior_mode", "add_log_prior")),
                online_warmup_factor=(1.0 if int(stage_epoch_1based) > int(getattr(args, "online_warmup_epochs", 2)) else (float(stage_epoch_1based) / max(1.0, float(getattr(args, "online_warmup_epochs", 2))))),
                phase_name=str(phase_name),
                include_online_decision_loss=bool(phase_online_decision),
                include_online_ce_loss=bool(phase_online_ce),
                anti_immediate_stop=bool(phase_anti_immediate),
                anti_immediate_stop_tokens=int(getattr(args, "anti_immediate_stop_tokens", 5)),
                anti_immediate_stop_weight=float(getattr(args, "anti_immediate_stop_weight", 0.1)),
                pre_devend_stop_weight=float(getattr(args, "pre_devend_stop_weight", 0.0)),
                stop_entropy_weight=float(phase_stop_entropy_w),
                stop_prior_weight=float(phase_stop_prior_w),
                stop_prior_target=float(getattr(args, "stop_prior_target", 0.05)),
                debug_loss_check=bool(getattr(args, "debug_loss_check", False)),
                debug_overfit_tiny=bool(getattr(args, "debug_overfit_tiny", False)),
                debug_first_batch_done=bool(first_batch_debug_done),
                optimizer_step_state=optimizer_step_state,
                max_optimizer_steps=(int(getattr(args, "debug_max_steps", 0)) if bool(getattr(args, "debug_overfit_tiny", False)) else None),
            )
            first_batch_debug_done = True


            if bool(getattr(args, "debug_disable_val", False)) or len(val_idx) == 0:
                va = dict(tr)
            else:
                va = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    chunk_len=int(args.chunk_len),
                    lambda_token=float(args.lambda_token),
                    trial_T_tokens=int(ds.trial_T_tokens),
                    tone_T=int(ds.tone_T),
                    isi_T=int(ds.isi_T),
                    token_loss_mode=str(args.token_loss_mode),
                    token_tau=float(args.token_tau),
                    token_w_min=float(args.token_w_min),
                    token_ms=int(ds.token_ms),
                    tok_window_ms=int(tok_window_curr),
                    tok_start_offset_ms=int(getattr(args, "tok_start_offset_ms", 0)),
                    tok_anchor=str(getattr(args, "tok_anchor", "deviant_end")),
                    lambda_anti_commit=float(lambda_anti_curr),
                    anti_commit_window_ms=int(anti_window_curr),
                    anti_commit_start_offset_ms=int(getattr(args, "anti_commit_start_offset_ms", 0)),
                    anti_commit_max_conf=float(anti_conf_curr),
                    rt_p_thresh=float(args.rt_p_thresh),
                    rt_k_consec=int(args.rt_k_consec),
                    rt_mode=str(getattr(args, "rt_mode", "entropy")),
                    rt_entropy_thresh=float(getattr(args, "rt_entropy_thresh", 0.35)),
                    min_rt_tokens=int(getattr(args, "min_rt_tokens", 0)),
                    debug_labels=bool(getattr(args, "debug_labels", False)),
                    debug_labels_fatal=bool(getattr(args, "debug_labels_fatal", False)),
                    debug_labels_dump=bool(getattr(args, "debug_labels_dump", False)),
                    debug_labels_dump_dir=(run_dir / "logs" / "bad_batches"),
                    debug_labels_max_dumps=int(getattr(args, "debug_labels_max_dumps", 10)),
                    debug_labels_first_n_batches=int(getattr(args, "debug_labels_first_n_batches", 0)),
                    epoch_global=int(epoch_global),
                    use_mask_loss=bool(getattr(args, "use_mask_loss", False)),
                    important_token_indices=important_token_indices,
                    gap_weight_power=float(getattr(args, "gap_weight_power", 0.0)),
                    gap_weight_ref_hz=float(getattr(args, "gap_weight_ref_hz", 25.0)),
                    gap_weight_max=float(getattr(args, "gap_weight_max", 2.5)),
                    online_cost_monitor=bool(getattr(args, "online_cost_monitor", False)),
                    online_cost_anchor=str(getattr(args, "online_cost_anchor", "deviant_onset")),
                    online_cost_wrong=float(getattr(args, "online_cost_wrong", 1.0)),
                    online_cost_w_time=float(getattr(args, "online_cost_w_time", 0.001)),
                    online_decision_training=bool(phase_online_decision),
                    online_loss_weight=float(getattr(args, "online_loss_weight", 0.5)),
                    online_ce_weight=float(phase_online_ce_weight),
                    decision_cost_mode=str(phase_decision_cost_mode),
                    timeout_ms=(None if getattr(args, "timeout_ms", None) is None else float(getattr(args, "timeout_ms"))),
                    time_cost_w=(None if getattr(args, "time_cost_w", None) is None else float(getattr(args, "time_cost_w"))),
                    derive_time_cost_from_timeout=bool(getattr(args, "derive_time_cost_from_timeout", False)),
                    decision_reference=str(getattr(args, "decision_reference", "trial_onset")),
                    rt_logging_reference=str(getattr(args, "rt_logging_reference", "deviant_end")),
                    cost_reference=str(getattr(args, "cost_reference", "deviant_onset")),
                    clamp_negative_cost_time=bool(getattr(args, "clamp_negative_cost_time", True)),
                    online_supervision_start=str(getattr(args, "online_supervision_start", "deviant_start")),
                    online_supervision_end=str(getattr(args, "online_supervision_end", "trial_end")),
                    aux_token_ce_weight=float(getattr(args, "aux_token_ce_weight", 0.0)),
                    aux_token_ce_start=str(getattr(args, "aux_token_ce_start", "deviant_onset")),
                    aux_token_ce_end=str(getattr(args, "aux_token_ce_end", "trial_end")),
                    decision_softmin_tau=float(getattr(args, "decision_softmin_tau", 0.05)),
                    use_stop_head=bool(phase_use_stop_head),
                    decision_policy=str(getattr(args, "decision_policy", "deterministic_threshold")),
                    sampling_temperature=float(getattr(args, "sampling_temperature", 1.0)),
                    stop_temperature=float(getattr(args, "stop_temperature", 1.0)),
                    use_hazard_prior=bool(getattr(args, "use_hazard_prior", False)),
                    hazard_prior_weight=float(getattr(args, "hazard_prior_weight", 1.0)),
                    hazard_prior_mode=str(getattr(args, "hazard_prior_mode", "add_log_prior")),
                    online_warmup_factor=(1.0 if int(stage_epoch_1based) > int(getattr(args, "online_warmup_epochs", 2)) else (float(stage_epoch_1based) / max(1.0, float(getattr(args, "online_warmup_epochs", 2))))),
                    phase_name=str(phase_name),
                    include_online_decision_loss=bool(phase_online_decision),
                    include_online_ce_loss=bool(phase_online_ce),
                    anti_immediate_stop=bool(phase_anti_immediate),
                    anti_immediate_stop_tokens=int(getattr(args, "anti_immediate_stop_tokens", 5)),
                    anti_immediate_stop_weight=float(getattr(args, "anti_immediate_stop_weight", 0.1)),
                    pre_devend_stop_weight=float(getattr(args, "pre_devend_stop_weight", 0.0)),
                    stop_entropy_weight=float(phase_stop_entropy_w),
                    stop_prior_weight=float(phase_stop_prior_w),
                    stop_prior_target=float(getattr(args, "stop_prior_target", 0.05)),
                )


            elapsed = time.time() - t_run0

            improved_global = (va["total_loss"] < (best_val - min_delta))
            if improved_global:
                best_val = float(va["total_loss"])
                best_epoch = int(epoch_global)

            improved_target = False
            if int(isi_ms) == int(best_target_isi):
                improved_target = (va["total_loss"] < (best_target_val - min_delta))
                if improved_target:
                    best_target_val = float(va["total_loss"])
                    best_target_epoch = int(epoch_global)

            improved_stage = (va["total_loss"] < (stage_best_val - min_delta))
            if improved_stage:
                stage_best_val = float(va["total_loss"])
                stage_bad_count = 0
                stage_best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
                stage_best_optim = copy.deepcopy(optim.state_dict())
                stage_best_epoch_global = int(epoch_global)
            else:
                stage_bad_count += 1

            target_msg = ""
            if int(isi_ms) == int(best_target_isi):
                target_msg = f" target_best={best_target_val:.4f}@{best_target_epoch}"

            mask_info = ""
            if bool(getattr(args, "use_mask_loss", False)):
                mask_info = " [MASK LOSS ENABLED]"

            print(
                f"[epoch {epoch_global:04d}] (stage={stage} isi={isi_ms} token_ms={ds.token_ms} phase={phase_name} lr={phase_lr:.2e}){mask_info}"
                f" anti_cfg=(lam={float(lambda_anti_curr):.3f},win={int(anti_window_curr)},conf={float(anti_conf_curr):.2f})"
                f" train: loss={tr['total_loss']:.4f} end={tr['end_loss']:.4f} tok={tr['token_loss']:.4f} anti={tr['anti_commit_loss']:.4f} "
                f"wTok={tr.get('weighted_token_loss', float('nan')):.4f} "
                f"onl={tr.get('online_decision_loss', float('nan')):.4f} wOnl={tr.get('weighted_online_decision_loss', float('nan')):.4f} "
                f"onlCE={tr.get('online_ce_loss', float('nan')):.4f} wOnlCE={tr.get('weighted_online_ce_loss', float('nan')):.4f} "
                f"auxCE={tr.get('aux_token_ce_loss', float('nan')):.4f} wAuxCE={tr.get('weighted_aux_token_ce_loss', float('nan')):.4f} "
                f"antiImm={tr.get('anti_immediate_stop_loss', float('nan')):.4f} stopEnt={tr.get('stop_entropy_bonus', float('nan')):.4f} stopPrior={tr.get('stop_prior_loss', float('nan')):.4f} "
                f"acc={tr['acc']:.4f} f1={tr['f1_macro']:.4f} auc={tr['auc_ovr']:.4f} "
                f"meanRT={tr['mean_rt_tokens']:.2f}tok/{tr['mean_rt_ms']:.1f}ms found={tr['rt_found']} miss={tr['rt_miss']} not_first={tr['rt_not_first']}({tr['rt_not_first_rate']:.1%}) | "
                f"onlineC={tr.get('online_decision_cost', float('nan')):.4f} foundP={tr.get('expected_found_prob', float('nan')):.4f} "
                f"noResp={tr.get('mean_no_response_prob', float('nan')):.4f} rtLog={tr.get('expected_rt_logged_ms', float('nan')):.2f}ms "
                f"negRT={tr.get('proportion_negative_rt', float('nan')):.3f} preEnd={tr.get('proportion_decisions_before_deviant_end', float('nan')):.3f} "
                f"pStop={tr.get('mean_p_stop', float('nan')):.4f} lastTok={tr.get('last_token_acc', float('nan')):.4f} devEnd={tr.get('deviant_end_acc', float('nan')):.4f} | "
                f"val: loss={va['total_loss']:.4f} end={va['end_loss']:.4f} tok={va['token_loss']:.4f} anti={va['anti_commit_loss']:.4f} "
                f"wTok={va.get('weighted_token_loss', float('nan')):.4f} "
                f"onl={va.get('online_decision_loss', float('nan')):.4f} wOnl={va.get('weighted_online_decision_loss', float('nan')):.4f} "
                f"onlCE={va.get('online_ce_loss', float('nan')):.4f} wOnlCE={va.get('weighted_online_ce_loss', float('nan')):.4f} "
                f"auxCE={va.get('aux_token_ce_loss', float('nan')):.4f} wAuxCE={va.get('weighted_aux_token_ce_loss', float('nan')):.4f} "
                f"antiImm={va.get('anti_immediate_stop_loss', float('nan')):.4f} stopEnt={va.get('stop_entropy_bonus', float('nan')):.4f} stopPrior={va.get('stop_prior_loss', float('nan')):.4f} "
                f"acc={va['acc']:.4f} f1={va['f1_macro']:.4f} auc={va['auc_ovr']:.4f} "
                f"meanRT={va['mean_rt_tokens']:.2f}tok/{va['mean_rt_ms']:.1f}ms found={va['rt_found']} miss={va['rt_miss']} not_first={va['rt_not_first']}({va['rt_not_first_rate']:.1%}) | "
                f"onlineC={va.get('online_decision_cost', float('nan')):.4f} foundP={va.get('expected_found_prob', float('nan')):.4f} "
                f"noResp={va.get('mean_no_response_prob', float('nan')):.4f} rtLog={va.get('expected_rt_logged_ms', float('nan')):.2f}ms "
                f"negRT={va.get('proportion_negative_rt', float('nan')):.3f} preEnd={va.get('proportion_decisions_before_deviant_end', float('nan')):.3f} "
                f"pStop={va.get('mean_p_stop', float('nan')):.4f} lastTok={va.get('last_token_acc', float('nan')):.4f} devEnd={va.get('deviant_end_acc', float('nan')):.4f} | "
                f"best_val={best_val:.4f}@{best_epoch} stage_bad={stage_bad_count}/{patience} "
                f"elapsed={_fmt_hms(elapsed)}{target_msg}"
            )

            ckpt = {
                "epoch_global": epoch_global,
                "stage": stage,
                "isi_ms": isi_ms,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "cfg": asdict(cfg),
                "args": vars(args),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "best_target_isi": best_target_isi,
                "best_target_val": best_target_val,
                "best_target_epoch": best_target_epoch,
                "sweep": sweep_info,
            }
            torch.save(ckpt, run_dir / "last.pt")
            if improved_global:
                torch.save(ckpt, run_dir / "best.pt")
            if improved_stage:
                torch.save(ckpt, stage_best_path)
            if improved_target:
                torch.save(ckpt, best_target_path)

            row = {
                "epoch_global": epoch_global,
                "stage": stage,
                "isi_ms": isi_ms,
                "token_ms": int(ds.token_ms),

                "train_total_loss": tr["total_loss"],
                "train_end_loss": tr["end_loss"],
                "train_token_loss": tr["token_loss"],
                "train_anti_commit_loss": tr["anti_commit_loss"],
                "train_acc": tr["acc"],
                "train_f1_macro": tr["f1_macro"],
                "train_auc_ovr": tr["auc_ovr"],
                "train_mean_rt_tokens": tr["mean_rt_tokens"],
                "train_mean_rt_ms": tr["mean_rt_ms"],
                "train_rt_found": tr["rt_found"],
                "train_rt_miss": tr["rt_miss"],
                "train_rt_not_first": tr["rt_not_first"],
                "train_rt_not_first_rate": tr["rt_not_first_rate"],
                "train_online_decision_cost": tr.get("online_decision_cost", float("nan")),
                "train_online_decision_loss": tr.get("online_decision_loss", float("nan")),
                "train_online_ce_loss": tr.get("online_ce_loss", float("nan")),
                "train_mean_p_stop": tr.get("mean_p_stop", float("nan")),
                "train_mean_no_response_prob": tr.get("mean_no_response_prob", float("nan")),
                "train_mean_expected_decision_time_ms": tr.get("mean_expected_decision_time_ms", float("nan")),
                "train_expected_found_prob": tr.get("expected_found_prob", float("nan")),
                "train_expected_rt_logged_ms": tr.get("expected_rt_logged_ms", float("nan")),
                "train_expected_rt_from_deviant_onset_ms": tr.get("expected_rt_from_deviant_onset_ms", float("nan")),
                "train_expected_rt_from_deviant_end_ms": tr.get("expected_rt_from_deviant_end_ms", float("nan")),
                "train_proportion_negative_rt": tr.get("proportion_negative_rt", float("nan")),
                "train_proportion_decisions_before_deviant_end": tr.get("proportion_decisions_before_deviant_end", float("nan")),
                "train_phase_name": tr.get("phase_name", ""),
                "train_effective_online_loss_weight": tr.get("effective_online_loss_weight", float("nan")),
                "train_weighted_token_loss": tr.get("weighted_token_loss", float("nan")),
                "train_weighted_online_decision_loss": tr.get("weighted_online_decision_loss", float("nan")),
                "train_weighted_online_ce_loss": tr.get("weighted_online_ce_loss", float("nan")),
                "train_aux_token_ce_loss": tr.get("aux_token_ce_loss", float("nan")),
                "train_weighted_aux_token_ce_loss": tr.get("weighted_aux_token_ce_loss", float("nan")),
                "train_anti_immediate_stop_loss": tr.get("anti_immediate_stop_loss", float("nan")),
                "train_stop_entropy_bonus": tr.get("stop_entropy_bonus", float("nan")),
                "train_stop_prior_loss": tr.get("stop_prior_loss", float("nan")),
                "train_sampled_mean_rt_logged_ms": tr.get("sampled_mean_rt_logged_ms", float("nan")),
                "train_last_token_acc": tr.get("last_token_acc", float("nan")),
                "train_deviant_end_acc": tr.get("deviant_end_acc", float("nan")),

                "val_total_loss": va["total_loss"],
                "val_end_loss": va["end_loss"],
                "val_token_loss": va["token_loss"],
                "val_anti_commit_loss": va["anti_commit_loss"],
                "val_acc": va["acc"],
                "val_f1_macro": va["f1_macro"],
                "val_auc_ovr": va["auc_ovr"],
                "val_mean_rt_tokens": va["mean_rt_tokens"],
                "val_mean_rt_ms": va["mean_rt_ms"],
                "val_rt_found": va["rt_found"],
                "val_rt_miss": va["rt_miss"],
                "val_rt_not_first": va["rt_not_first"],
                "val_rt_not_first_rate": va["rt_not_first_rate"],
                "val_online_decision_cost": va.get("online_decision_cost", float("nan")),
                "val_online_decision_loss": va.get("online_decision_loss", float("nan")),
                "val_online_ce_loss": va.get("online_ce_loss", float("nan")),
                "val_mean_p_stop": va.get("mean_p_stop", float("nan")),
                "val_mean_no_response_prob": va.get("mean_no_response_prob", float("nan")),
                "val_mean_expected_decision_time_ms": va.get("mean_expected_decision_time_ms", float("nan")),
                "val_expected_found_prob": va.get("expected_found_prob", float("nan")),
                "val_expected_rt_logged_ms": va.get("expected_rt_logged_ms", float("nan")),
                "val_expected_rt_from_deviant_onset_ms": va.get("expected_rt_from_deviant_onset_ms", float("nan")),
                "val_expected_rt_from_deviant_end_ms": va.get("expected_rt_from_deviant_end_ms", float("nan")),
                "val_proportion_negative_rt": va.get("proportion_negative_rt", float("nan")),
                "val_proportion_decisions_before_deviant_end": va.get("proportion_decisions_before_deviant_end", float("nan")),
                "val_phase_name": va.get("phase_name", ""),
                "val_effective_online_loss_weight": va.get("effective_online_loss_weight", float("nan")),
                "val_weighted_token_loss": va.get("weighted_token_loss", float("nan")),
                "val_weighted_online_decision_loss": va.get("weighted_online_decision_loss", float("nan")),
                "val_weighted_online_ce_loss": va.get("weighted_online_ce_loss", float("nan")),
                "val_aux_token_ce_loss": va.get("aux_token_ce_loss", float("nan")),
                "val_weighted_aux_token_ce_loss": va.get("weighted_aux_token_ce_loss", float("nan")),
                "val_anti_immediate_stop_loss": va.get("anti_immediate_stop_loss", float("nan")),
                "val_stop_entropy_bonus": va.get("stop_entropy_bonus", float("nan")),
                "val_stop_prior_loss": va.get("stop_prior_loss", float("nan")),
                "val_sampled_mean_rt_logged_ms": va.get("sampled_mean_rt_logged_ms", float("nan")),
                "val_last_token_acc": va.get("last_token_acc", float("nan")),
                "val_deviant_end_acc": va.get("deviant_end_acc", float("nan")),

                "best_val": best_val,
                "best_epoch": best_epoch,
                "best_target_isi_val": best_target_val,
                "best_target_isi_epoch": best_target_epoch,
                "best_target_isi": best_target_isi,
                "time_elapsed_sec": elapsed,
            }

            write_jsonl(jsonl_path, row)
            write_csv_row(csv_path, header, row)
            history.append(row)

            if int(getattr(args, "plot_every", 0)) > 0 and (epoch_global % int(args.plot_every) == 0):
                plot_history(history, run_dir / "plots" / "dummy.png")

            # 新增：检查是否达到stage性能阈值（可对最后stage关闭）
            if stage_threshold_active and not stage_reached_threshold:
                if va['acc'] >= stage_perf_threshold:
                    stage_reached_threshold = True
                    current_stage_epochs = epoch_global - stage_start_epoch + 1
                    
                    print(f"\n🎯 Stage {stage} (isi={isi_ms}) reached performance threshold {stage_perf_threshold:.2%} "
                          f"at epoch {epoch_global} with val_acc={va['acc']:.2%}")
                    
                    if current_stage_epochs >= min_stage_epochs:
                        print(f"✓ Minimum epochs ({min_stage_epochs}) satisfied. Moving to next stage.")
                        break  # 跳出当前stage的epoch循环
                    else:
                        print(f"⏳ Need at least {min_stage_epochs} epochs in this stage "
                              f"(currently {current_stage_epochs}). Continuing training...")
                        stage_reached_threshold = False  # 重置标记，继续训练

            # 原有的early stopping（可对最后stage关闭）
            if stage_patience_active and stage_bad_count >= patience:
                print(
                    f"[early_stop] stage {stage} (isi={isi_ms}) patience reached: "
                    f"{stage_bad_count}/{patience}. End this stage."
                )
                break

            if bool(getattr(args, "debug_overfit_tiny", False)) and optimizer_step_state["count"] >= int(getattr(args, "debug_max_steps", 500)):
                print(f"[debug_overfit_tiny] reached max optimizer steps: {optimizer_step_state['count']}")
                break

        if stage_best_state is not None:
            model.load_state_dict(stage_best_state, strict=True)
            if stage_best_optim is not None:
                optim.load_state_dict(stage_best_optim)
            print(
                f"[stage_best] restore-once stage={stage} isi={isi_ms} "
                f"stage_best_val={stage_best_val:.4f} at epoch_global={stage_best_epoch_global}"
            )
        else:
            print(
                f"[stage_best] WARNING: no stage-best snapshot captured for stage={stage} isi={isi_ms}. "
                f"(If you set epochs_per_isi=0, or val crashed, this can happen.)"
            )

        if bool(getattr(args, "debug_overfit_tiny", False)) and optimizer_step_state["count"] >= int(getattr(args, "debug_max_steps", 500)):
            break

    plot_history(history, run_dir / "plots" / "dummy.png")
    print(f"[done] Saved run to: {run_dir.resolve()}")
    print("  - best.pt / last.pt")
    saved_stage_ckpts = []
    for isi_ms in args.isi_schedule:
        p = run_dir / f"best_isi{int(isi_ms)}.pt"
        if p.exists():
            saved_stage_ckpts.append(p.name)
    for name in saved_stage_ckpts:
        print(f"  - {name}")
    print("  - logs/metrics.jsonl + logs/metrics.csv")
    print("  - plots/acc.png f1.png auc.png loss.png")

    if bool(getattr(args, "run_post_analysis", False)):
        human_csv = Path(str(getattr(args, "human_trial_csv", ""))).expanduser()
        isi_for_export = int(getattr(args, "analysis_isi", 700))
        model_csv_name = str(getattr(args, "model_trial_csv_name", "model_trial.csv"))

        ckpt_candidates = [
            run_dir / f"best_isi{isi_for_export}.pt",
            run_dir / "best.pt",
            run_dir / "last.pt",
        ]
        ckpt_to_use = None
        for p in ckpt_candidates:
            if p.exists():
                ckpt_to_use = p
                break

        if ckpt_to_use is None:
            print("[analysis] no checkpoint found (best_isi/best/last). Skip.")
            return

        print(f"[export_model_csv] using checkpoint: {ckpt_to_use.name}")

        model_csv_path = None
        try:
            model_csv_path = export_model_trial_csv_from_checkpoint(
                ckpt_path=ckpt_to_use,
                args=args,
                run_dir=run_dir,
                isi_ms_for_export=isi_for_export,
                out_name=model_csv_name,
            )
        except Exception as e:
            print(f"[export_model_csv] failed: {e}")

        if model_csv_path is not None:
            out_sub = Path(str(getattr(args, "analysis_out_subdir", "analysis/position_effect")))
            out_dir = run_dir / out_sub
            try:
                run_position_effect_analysis(
                    human_csv=human_csv,
                    model_csv=model_csv_path,
                    out_dir=out_dir,
                    isi=isi_for_export,
                    human_id_col=getattr(args, "analysis_human_id_col", "subject_id"),
                    model_id_col=getattr(args, "analysis_model_id_col", None),
                    n_boot=int(getattr(args, "analysis_n_boot", 5000)),
                    seed=int(getattr(args, "analysis_seed", 0)),
                )
            except Exception as e:
                print(f"[analysis] position-effect analysis failed: {e}")
        else:
            print("[analysis] model_csv export failed; skip analysis.")# -------------------------
# Main
# -------------------------
def parse_list_of_floats(s: str) -> List[float]:
    # allow "0,0.05,0.1" or "0 0.05 0.1"
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    parts = [p for p in s.replace(",", " ").split(" ") if p]
    return [float(p) for p in parts]


def parse_list_of_ints(s: str) -> List[int]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    parts = [p for p in s.replace(",", " ").split(" ") if p]
    return [int(float(p)) for p in parts]


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, required=True,
                   help="训练数据目录（会读取 input_blocks.pt/labels_blocks.pt 或 input_tensor.pt/labels_tensor.pt）")
    p.add_argument("--save_dir", type=str, required=True)

    # (NEW) regenerate stimuli before training
    p.add_argument("--regen_stimuli", action="store_true",
                   help="训练前先生成新的 stimuli（覆盖写入 input_blocks.pt/labels_blocks.pt/meta.json）")
    p.add_argument("--stimuli_out_dir", type=str, default="",
                   help="生成 stimuli 的输出目录；默认空=写到 data_dir 里")
    p.add_argument("--stimuli_seed", type=int, default=-1,
                   help="生成 stimuli 用的 seed；-1 表示复用 --seed")

    # stimuli params (gm_stimuli-compatible)
    p.add_argument("--stimuli_n_blocks", type=int, default=10000)
    p.add_argument("--stimuli_f_min", type=float, default=1300.0)
    p.add_argument("--stimuli_f_max", type=float, default=1700.0)
    p.add_argument("--stimuli_f_step", type=float, default=5.0)
    p.add_argument("--stimuli_min_diff", type=float, default=-1.0,
                   help="min |f_dev-f_std|；-1 表示等于 f_step")
    p.add_argument("--stimuli_exclude_freqs", type=float, nargs="*", default=[1455.0, 1500.0, 1600.0])
    p.add_argument("--stimuli_exclude_tol", type=float, default=1e-6)
    p.add_argument("--stimuli_no_seen_pairs", action="store_true",
                   help="允许全局重复使用 (std,dev) pair（不去重）")
    p.add_argument("--stimuli_sample_mode", type=str, default="continuous", choices=["continuous", "discrete"])
    p.add_argument("--stimuli_round_to", type=float, default=None)

    # reproducibility
    p.add_argument("--seed", type=int, default=42)

    # device
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")

    # training base
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)

    # model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_norm", action="store_true")

    # tbptt + losses
    p.add_argument("--chunk_len", type=int, default=512)
    p.add_argument("--lambda_token", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--token_loss_mode", type=str, default="exp", choices=["uniform", "exp"])
    p.add_argument("--token_tau", type=float, default=50.0)
    p.add_argument("--token_w_min", type=float, default=0.05)

    # timing (ms)
    p.add_argument("--tone_ms", type=int, default=50)
    p.add_argument("--ramp_ms", type=int, default=5)  # kept
    p.add_argument("--token_ms", type=int, default=10)

    # Curriculum
    p.add_argument("--isi_schedule", type=str, default="0,50,300,700",
                   help="ISI 课程学习列表，逗号分隔，例如 '0,50,300,700'")
    p.add_argument("--epochs_per_isi", type=int, default=5,
                   help="每个 ISI 阶段训练多少个 epoch（总 epoch = len(isi_schedule)*epochs_per_isi）")

    # ERB one-hot space
    p.add_argument("--f_min_hz", type=float, default=1300.0)
    p.add_argument("--f_max_hz", type=float, default=1700.0)
    p.add_argument("--n_bins", type=int, default=128)
    p.add_argument("--add_eos", action="store_true")

    # noise (single-run values; can be overridden by sweeps)
    p.add_argument("--sigma_other_noise", type=float, default=0.05)
    p.add_argument("--p_other_noise", type=float, default=1.0)
    p.add_argument("--sigma_silence_noise", type=float, default=0.0)
    p.add_argument("--resample_noise_per_epoch", action="store_true",
                   help="训练集在每个 epoch 重采样输入噪声；验证集保持固定噪声。")
    p.add_argument("--hidden_noise_std", type=float, default=0.0,
                   help="训练期加在 layer-norm 后 hidden states 上的高斯噪声标准差。")

    # RT criterion (eval only; can be overridden by sweeps)
    p.add_argument("--rt_p_thresh", type=float, default=0.7)
    p.add_argument("--rt_k_consec", type=int, default=3)

    # resume
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt (best.pt/last.pt)")

    # debug / sanity
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_steps", type=int, default=3)
    p.add_argument("--max_blocks", type=int, default=0)
    p.add_argument("--debug_loss_check", action="store_true")
    p.add_argument("--debug_overfit_tiny", action="store_true")
    p.add_argument("--debug_n_blocks", type=int, default=16)
    p.add_argument("--debug_max_steps", type=int, default=500)
    p.add_argument("--debug_disable_val", action="store_true")
    p.add_argument("--debug_no_noise", action="store_true")
    p.add_argument("--debug_no_curriculum", action="store_true")
    p.add_argument("--debug_end_loss_only", action="store_true")
    p.add_argument("--debug_trialwise_rendered", action="store_true")
    p.add_argument("--debug_trialwise_print_every", type=int, default=25)
    p.add_argument("--debug_stochastic_dynamic_readout", action="store_true")
    p.add_argument("--debug_train_stochastic_stop", action="store_true")
    p.add_argument("--debug_disable_layer_norm", action="store_true")
    p.add_argument("--debug_n_stop_samples", type=int, default=50)
    p.add_argument("--debug_stop_temperature", type=float, default=1.0)
    p.add_argument("--debug_class_temperature", type=float, default=1.0)
    p.add_argument(
        "--debug_dynamic_start",
        type=str,
        default="deviant_onset",
        choices=["trial_start", "deviant_onset", "deviant_end"],
    )
    p.add_argument(
        "--debug_dynamic_end",
        type=str,
        default="trial_end",
        choices=["trial_end"],
    )
    p.add_argument(
        "--debug_rt_logging_reference",
        type=str,
        default="deviant_end",
        choices=["trial_onset", "deviant_onset", "deviant_end"],
    )
    p.add_argument(
        "--debug_cost_reference",
        type=str,
        default="deviant_onset",
        choices=["trial_onset", "deviant_onset", "deviant_end"],
    )
    p.add_argument("--debug_clamp_negative_cost_time", action="store_true")
    p.add_argument("--debug_time_cost_w", type=float, default=0.001)
    p.add_argument("--debug_no_response_cost", type=float, default=1.0)
    p.add_argument("--debug_wrong_cost", type=float, default=1.0)
    p.add_argument("--debug_use_expected_stop_loss", action="store_true")
    p.add_argument("--debug_stop_entropy_weight", type=float, default=0.01)
    p.add_argument("--debug_stop_prior_weight", type=float, default=0.01)
    p.add_argument("--debug_stop_prior_target", type=float, default=0.05)
    p.add_argument("--debug_anti_immediate_stop_tokens", type=int, default=5)
    p.add_argument("--debug_anti_immediate_stop_weight", type=float, default=0.1)
    p.add_argument("--debug_aux_token_ce_weight", type=float, default=0.0)
    p.add_argument(
        "--debug_aux_token_ce_start",
        type=str,
        default="deviant_onset",
        choices=["trial_start", "deviant_onset", "deviant_end"],
    )
    p.add_argument(
        "--debug_aux_token_ce_end",
        type=str,
        default="trial_end",
        choices=["trial_end"],
    )
    p.add_argument("--tok_window_ms", type=int, default=300)
    p.add_argument("--tok_start_offset_ms", type=int, default=0)
    p.add_argument("--tok_anchor", type=str, default="deviant_end",
                   choices=["deviant_end", "deviant_onset"],
                   help="token-level supervision window anchor. deviant_onset enables during-stimulus decision learning.")
    p.add_argument("--train_online_decision", action="store_true",
                   help="快捷开关：把 token 监督锚点改到 deviant onset，并扩大窗口到整段 trial。")
    p.add_argument("--lambda_anti_commit", type=float, default=0.0,
                   help="训练期 anti-early-commitment 正则系数；>0 时会惩罚 deviant 后早期过高置信度。")
    p.add_argument("--lambda_anti_commit_final", type=float, default=-1.0,
                   help="stage 内 anti-commit 系数的目标终值；<0 表示保持不变。")
    p.add_argument("--anti_commit_window_ms", type=int, default=0,
                   help="deviant 后用于 anti-early-commitment 的时间窗长度（ms）；0=关闭。")
    p.add_argument("--anti_commit_window_ms_final", type=int, default=-1,
                   help="stage 内 anti-commit window 的目标终值；<0 表示保持不变。")
    p.add_argument("--anti_commit_start_offset_ms", type=int, default=0,
                   help="anti-early-commitment 时间窗相对 deviant end 的起始偏移（ms）。")
    p.add_argument("--anti_commit_max_conf", type=float, default=1.0,
                   help="anti-early-commitment 允许的最大 token 置信度上限；越小惩罚越强。")
    p.add_argument("--anti_commit_max_conf_final", type=float, default=-1.0,
                   help="stage 内 anti-commit 最大允许置信度的目标终值；<0 表示保持不变。")
    p.add_argument("--anti_commit_schedule_epochs", type=int, default=0,
                   help="在每个 stage 的前多少个 epoch 内线性插值 anti-commit 参数；0=关闭 schedule。")
    p.add_argument("--gap_weight_power", type=float, default=0.0,
                   help="按 |f_dev-f_std| 对 trial loss 加权；>0 会提高小频差 trial 的训练权重。")
    p.add_argument("--gap_weight_ref_hz", type=float, default=25.0,
                   help="gap-aware weighting 的参考频差（Hz）。")
    p.add_argument("--gap_weight_max", type=float, default=2.5,
                   help="gap-aware weighting 的最大/最小权重夹紧范围。")
    # ---- debug labels / dumping ----
    p.add_argument("--debug_labels", action="store_true",
                help="开启 label 链路 debug：检查 dataset/batch/device 的 y 是否只含 {4,5,6}，并在发现异常时打印/可选保存 dump")
    p.add_argument("--debug_labels_fatal", action="store_true",
                help="遇到 label 异常就直接 raise（默认只 warn 并尽量跳过RT）")
    p.add_argument("--debug_labels_dump", action="store_true",
                help="遇到 label 异常时保存 dump 到 run_dir/logs/bad_batches/ 方便复现")
    p.add_argument("--debug_labels_max_dumps", type=int, default=10,
                help="最多保存多少个 bad batch dump（防止爆硬盘）")
    p.add_argument("--debug_labels_first_n_batches", type=int, default=0,
                help="前 N 个 batch 强制打印 y 的 unique（0=不打印）")
    p.add_argument("--debug_dataset_assert", action="store_true",
                help="在 Dataset.__getitem__ 里 assert y 只能是 {4,5,6}（能定位是否数据源本身坏）")
    # ---- post-run analysis: position effect ----
    p.add_argument("--run_post_analysis", action="store_true",
                   help="每个 run 结束后自动跑 position-effect 分析（需要 human_trial.csv + run_dir/model_trial.csv）")
    p.add_argument("--human_trial_csv", type=str, default="",
                   help="human_trial.csv 的路径（trial-level：subject_id, position, rt_ms, isi_ms 可选）")
    p.add_argument("--model_trial_csv_name", type=str, default="model_trial.csv",
                   help="每个 run_dir 里 model trial 文件名（默认 model_trial.csv）")
    p.add_argument("--analysis_out_subdir", type=str, default="analysis/position_effect",
                   help="结果输出到 run_dir 下的子目录")
    p.add_argument("--analysis_isi", type=int, default=700,
                   help="可选：只分析某个 isi_ms（列存在时才过滤；你现在默认 700）")
    p.add_argument("--analysis_human_id_col", type=str, default="subject_id")
    p.add_argument("--analysis_model_id_col", type=str, default=None)
    p.add_argument("--analysis_n_boot", type=int, default=5000)
    p.add_argument("--analysis_seed", type=int, default=0)

    # logging/plot
    p.add_argument("--plot_every", type=int, default=1, help="每隔多少个 epoch 更新一次图（0=只在最后画）")

    # early stopping
    p.add_argument("--early_stop_patience", type=int, default=10,
                   help="val_total_loss 连续不提升多少个 epoch 后停止（0=不早停）")
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                   help="认为“提升”的最小阈值：old - new > min_delta")

    # --- sweep controls ---
    p.add_argument("--sweep", action="store_true",
                   help="开启 grid sweep：自动组合 noise 和 threshold，分别跑到不同子文件夹")
    p.add_argument("--sweep_sigma_other", type=str, default="",
                   help="例如 '0,0.02,0.05,0.1'；空=不 sweep 该项(用当前 sigma_other_noise)")
    p.add_argument("--sweep_p_other", type=str, default="",
                   help="例如 '0.25,0.5,1.0'；空=不 sweep 该项(用当前 p_other_noise)")
    p.add_argument("--sweep_sigma_silence", type=str, default="",
                   help="例如 '0,0.01,0.02'；空=不 sweep 该项(用当前 sigma_silence_noise)")
    p.add_argument("--sweep_rt_p_thresh", type=str, default="",
                   help="例如 '0.6,0.7,0.8'；空=不 sweep(用当前 rt_p_thresh)")
    p.add_argument("--sweep_rt_k_consec", type=str, default="",
                   help="例如 '1,2,3,4'；空=不 sweep(用当前 rt_k_consec)")
    p.add_argument("--sweep_token_ms", type=str, default="",
                help="例如 '1,2,5,10,25,50'；空=不 sweep(用当前 token_ms)")

    # --- 新增：掩码损失参数 ---
    p.add_argument("--use_mask_loss", action="store_true",
                   help="是否使用掩码损失（只计算有意义token的损失）")
    p.add_argument("--important_token_indices", type=str, default="",
                   help="有意义token的位置索引，例如 '10,20,30'（每个trial中的相对位置）")

    p.add_argument("--stage_perf_threshold", type=float, default=0.0,
                   help="每个stage达到此准确率阈值后可以提前结束（0-1之间，0表示不启用）")
    p.add_argument("--min_stage_epochs", type=int, default=10,
                   help="每个stage最少训练的epoch数，即使达到阈值也要训练够这个数")
    p.add_argument("--force_full_last_stage", action="store_true",
                   help="最后一个ISI stage关闭阈值跳转与patience早停，强制跑满epochs_per_isi。")
    p.add_argument(
        "--init_from", type=str, default="",
        help="只加载模型权重作为初始化；不恢复 optimizer / epoch / best_val。适合跨条件微调或 sweep。"
    )
    p.add_argument(
        "--eval_only_ckpt", type=str, default="",
        help="只用指定 checkpoint 做导出/分析，不训练。会使用当前 rt_p_thresh / rt_k_consec / analysis 参数。"
    )

    p.add_argument("--rt_mode", type=str, default="entropy",
               choices=["entropy", "confidence", "oracle"])
    p.add_argument("--rt_entropy_thresh", type=float, default=0.35)
    p.add_argument("--min_rt_tokens", type=int, default=0)
    p.add_argument("--online_cost_monitor", action="store_true",
                   help="记录 online speed-accuracy 代价监控: wrong=1, correct=w_time*rt_ms (不参与反传)")
    p.add_argument("--online_cost_anchor", type=str, default="trial_onset",
                   choices=["trial_onset", "deviant_onset", "deviant_end"])
    p.add_argument("--online_cost_wrong", type=float, default=1.0)
    p.add_argument("--online_cost_w_time", type=float, default=None,
                   help="online 决策代价里时间项权重 w (cost= w*rt_ms).")
    p.add_argument("--decision_policy", type=str, default="deterministic_threshold",
                   choices=["deterministic_threshold", "posterior_sampling", "softmax_policy", "hazard_bayesian", "gumbel_softmax", "expected_cost_softmin"])
    p.add_argument("--decision_threshold", type=float, default=0.75)
    p.add_argument("--sampling_temperature", type=float, default=1.0)
    p.add_argument("--stop_temperature", type=float, default=1.0)
    p.add_argument("--n_policy_samples", type=int, default=1)
    p.add_argument("--use_stop_head", action="store_true")
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--gumbel_hard", action="store_true")
    p.add_argument("--gumbel_tau_anneal", action="store_true")
    p.add_argument("--gumbel_tau_min", type=float, default=0.2)
    p.add_argument("--use_hazard_prior", action="store_true")
    p.add_argument("--hazard_prior_weight", type=float, default=1.0)
    p.add_argument("--learn_hazard_prior_weight", action="store_true")
    p.add_argument("--hazard_prior_mode", type=str, default="add_log_prior",
                   choices=["add_log_prior", "stop_bias", "cost_modulation"])
    p.add_argument("--online_decision_training", action="store_true")
    p.add_argument("--online_loss_weight", type=float, default=0.5)
    p.add_argument("--online_ce_weight", type=float, default=0.1)
    p.add_argument("--decision_cost_mode", type=str, default="expected_cost_softmin",
                   choices=["expected_cost_softmin", "stochastic_expected_cost", "policy_gradient", "masked_online_ce"])
    p.add_argument("--timeout_ms", type=float, default=None)
    p.add_argument("--time_cost_w", type=float, default=0.001)
    p.add_argument("--derive_time_cost_from_timeout", action="store_true",
                   help="仅在显式设置时，使用 w=1/timeout_ms。默认关闭。")
    p.add_argument("--decision_reference", type=str, default="trial_onset",
                   choices=["trial_onset", "deviant_start", "deviant_end"])
    p.add_argument("--online_supervision_start", type=str, default="deviant_start",
                   choices=["trial_start", "deviant_start", "deviant_end"])
    p.add_argument("--online_supervision_end", type=str, default="trial_end",
                   choices=["trial_end", "stimulus_end", "sequence_end"])
    p.add_argument("--dynamic_start", type=str, default="deviant_onset",
                   choices=["trial_start", "deviant_onset", "deviant_end"])
    p.add_argument("--rt_logging_reference", type=str, default="deviant_end",
                   choices=["trial_onset", "deviant_onset", "deviant_end"])
    p.add_argument("--cost_reference", type=str, default="deviant_onset",
                   choices=["trial_onset", "deviant_onset", "deviant_end"])
    p.add_argument("--clamp_negative_cost_time", action="store_true")
    p.add_argument("--aux_token_ce_weight", type=float, default=0.0)
    p.add_argument("--aux_token_ce_start", type=str, default="deviant_onset",
                   choices=["trial_start", "deviant_onset", "deviant_end"])
    p.add_argument("--aux_token_ce_end", type=str, default="trial_end",
                   choices=["trial_end", "stimulus_end", "sequence_end"])
    p.add_argument("--decision_softmin_tau", type=float, default=0.05)
    p.add_argument("--online_warmup_epochs", type=int, default=2)
    p.add_argument("--policy_gradient_baseline", type=str, default="running_mean",
                   choices=["none", "running_mean", "value_head"])
    p.add_argument("--policy_entropy_weight", type=float, default=0.01)
    p.add_argument("--block_context_training", action="store_true")
    p.add_argument("--detach_hidden_between_trials", action="store_true")
    p.add_argument("--sweep_time_cost_w", type=str, default="",
                   help="例如 '0.0003333333,0.0005,0.001,0.002'；空=不 sweep(用当前 time_cost_w)")
    p.add_argument("--two_phase_training", action="store_true")
    p.add_argument("--classifier_warmup_epochs", type=int, default=0)
    p.add_argument("--stop_finetune_epochs", type=int, default=None)
    p.add_argument("--warmup_lr", type=float, default=None)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--warmup_enable_stop_head", action="store_true")
    p.add_argument("--warmup_use_online_ce", action="store_true")
    p.add_argument("--warmup_online_ce_weight", type=float, default=0.1)
    p.add_argument("--finetune_enable_stochastic_cost", action="store_true")
    p.add_argument("--anti_immediate_stop", action="store_true")
    p.add_argument("--anti_immediate_stop_tokens", type=int, default=5)
    p.add_argument("--anti_immediate_stop_weight", type=float, default=0.1)
    p.add_argument("--stop_entropy_weight", type=float, default=0.0)
    p.add_argument("--stop_prior_weight", type=float, default=0.0)
    p.add_argument("--stop_prior_target", type=float, default=0.05)
    p.add_argument("--gap_curriculum", action="store_true")
    p.add_argument("--gap_schedule", type=str, default="25,10,1")
    p.add_argument("--epochs_per_gap", type=int, default=10)
    args = p.parse_args()
    if not hasattr(args, "warmup_use_online_ce"):
        args.warmup_use_online_ce = True
    # keep requested default True
    if "--warmup_use_online_ce" not in __import__("sys").argv:
        args.warmup_use_online_ce = True
    if "--debug_use_expected_stop_loss" not in __import__("sys").argv:
        args.debug_use_expected_stop_loss = True
    if "--debug_clamp_negative_cost_time" not in __import__("sys").argv:
        args.debug_clamp_negative_cost_time = True
    if "--clamp_negative_cost_time" not in __import__("sys").argv:
        args.clamp_negative_cost_time = True
    if "--dynamic_start" in __import__("sys").argv and "--online_supervision_start" not in __import__("sys").argv:
        dyn_to_sup = {
            "trial_start": "trial_start",
            "deviant_onset": "deviant_start",
            "deviant_end": "deviant_end",
        }
        args.online_supervision_start = dyn_to_sup[str(args.dynamic_start)]

    # parse isi schedule
    args.isi_schedule = parse_list_of_ints(args.isi_schedule)
    if not args.isi_schedule:
        raise ValueError("--isi_schedule 不能为空。例：0,50,300,700")

    if bool(getattr(args, "debug_overfit_tiny", False)):
        args.max_blocks = int(getattr(args, "debug_n_blocks", 16))
        if bool(getattr(args, "debug_disable_val", False)):
            args.val_split = 0.0
        if bool(getattr(args, "debug_no_noise", False)):
            args.sigma_other_noise = 0.0
            args.p_other_noise = 0.0
            args.sigma_silence_noise = 0.0
            args.hidden_noise_std = 0.0
            args.resample_noise_per_epoch = False
        if bool(getattr(args, "debug_no_curriculum", False)) and len(args.isi_schedule) > 1:
            args.isi_schedule = [int(args.isi_schedule[0])]
        if bool(getattr(args, "debug_end_loss_only", False)):
            args.lambda_token = 0.0
            args.lambda_anti_commit = 0.0
            args.lambda_anti_commit_final = -1.0
            args.train_online_decision = False
            args.online_decision_training = False
            args.online_loss_weight = 0.0
            args.online_ce_weight = 0.0
            args.warmup_online_ce_weight = 0.0
            args.warmup_use_online_ce = False
            args.use_stop_head = False
            args.warmup_enable_stop_head = False
            args.finetune_enable_stochastic_cost = False
            args.use_hazard_prior = False
            args.hazard_prior_weight = 0.0
            args.stop_entropy_weight = 0.0
            args.stop_prior_weight = 0.0
            args.anti_immediate_stop = False
            args.two_phase_training = False
        args.epochs_per_isi = max(int(args.epochs_per_isi), int(getattr(args, "debug_max_steps", 500)))
        print(
            "[debug_overfit_tiny] "
            f"n_blocks={args.max_blocks} max_steps={args.debug_max_steps} "
            f"val_split={args.val_split} isi_schedule={args.isi_schedule} "
            f"no_noise={bool(args.debug_no_noise)} end_loss_only={bool(args.debug_end_loss_only)}"
        )

    # parse important token indices
    if args.important_token_indices:
        args.important_token_indices = parse_list_of_ints(args.important_token_indices)
        print(f"[mask_loss] important token indices: {args.important_token_indices}")
    else:
        args.important_token_indices = None

    if bool(getattr(args, "force_full_last_stage", False)):
        print("[curriculum] force_full_last_stage=ON (last stage ignores stage threshold and early-stop patience)")

    # time-cost convention:
    # default is free parameter w; timeout-derived w only when explicitly requested.
    if bool(getattr(args, "derive_time_cost_from_timeout", False)):
        if getattr(args, "timeout_ms", None) is None or float(args.timeout_ms) <= 0:
            raise ValueError("--derive_time_cost_from_timeout requires positive --timeout_ms.")
        args.time_cost_w = 1.0 / float(args.timeout_ms)
    elif getattr(args, "time_cost_w", None) is None:
        args.time_cost_w = 0.001

    # monitor defaults to training w unless explicitly overridden
    if getattr(args, "online_cost_w_time", None) is None:
        args.online_cost_w_time = float(args.time_cost_w)
    elif not np.isclose(float(args.online_cost_w_time), float(args.time_cost_w), rtol=0, atol=1e-12):
        print("[warn] online_cost_w_time and time_cost_w differ; monitor and training objective use different time costs.")

    args.hard_timeout_enabled = False

    if str(getattr(args, "decision_policy", "deterministic_threshold")) == "hazard_bayesian":
        args.use_hazard_prior = True
        if not bool(getattr(args, "use_stop_head", False)):
            args.use_stop_head = True
    if str(getattr(args, "decision_policy", "deterministic_threshold")) in ("softmax_policy", "gumbel_softmax"):
        if not bool(getattr(args, "use_stop_head", False)):
            args.use_stop_head = True

    # 快捷模式：训练端允许并鼓励刺激期间决策
    if bool(getattr(args, "train_online_decision", False)):
        args.tok_anchor = "deviant_onset"
        args.online_decision_training = True
        trial_T_ms = 7 * int(args.tone_ms + args.isi_schedule[0]) + int(args.tone_ms)
        # 覆盖为“整段 trial 监督”，让模型尽早进入正确类别
        if int(args.tok_window_ms) <= 0:
            args.tok_window_ms = int(trial_T_ms)
        # 推荐同时打开监控
        args.online_cost_monitor = True
        print(
            f"[train_online_decision] ON: tok_anchor={args.tok_anchor}, "
            f"tok_window_ms={args.tok_window_ms}, min_rt_tokens={args.min_rt_tokens}"
        )

    # (optional) regenerate stimuli before everything
    data_dir = Path(args.data_dir)
    if args.regen_stimuli:
        out_dir = Path(args.stimuli_out_dir) if args.stimuli_out_dir else data_dir
        stim_seed = int(args.seed if args.stimuli_seed == -1 else args.stimuli_seed)
        md = None if float(args.stimuli_min_diff) < 0 else float(args.stimuli_min_diff)

        print("[stimuli] regen_stimuli=ON")
        print(f"[stimuli] out_dir={out_dir.resolve()}")
        print(f"[stimuli] n_blocks={args.stimuli_n_blocks} seed={stim_seed} "
              f"mode={args.stimuli_sample_mode} round_to={args.stimuli_round_to}")

        generate_stimuli_blocks(
            save_dir=out_dir,
            n_blocks=int(args.stimuli_n_blocks),
            trials_per_block=10,
            seq_len=8,
            f_min=float(args.stimuli_f_min),
            f_max=float(args.stimuli_f_max),
            f_step=float(args.stimuli_f_step),
            min_diff=md,
            seed=stim_seed,
            exclude_freqs=[float(v) for v in args.stimuli_exclude_freqs],
            exclude_tol=float(args.stimuli_exclude_tol),
            no_seen_pairs=bool(args.stimuli_no_seen_pairs),
            sample_mode=str(args.stimuli_sample_mode),
            round_to=args.stimuli_round_to,
        )
        args.data_dir = str(out_dir)

    root = Path(args.save_dir)
    root.mkdir(parents=True, exist_ok=True)

    if bool(getattr(args, "debug_trialwise_rendered", False)):
        print("[debug_trialwise_rendered] ON")
        run_trialwise_rendered_debug(args=args, run_dir=root)
        return

    # eval-only mode: use an existing checkpoint to export model_trial.csv / run post-analysis
    if args.eval_only_ckpt:
        root = Path(args.save_dir)
        root.mkdir(parents=True, exist_ok=True)
    
        ckpt_path = Path(args.eval_only_ckpt).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--eval_only_ckpt not found: {ckpt_path}")
    
        print(f"[eval_only] using checkpoint: {ckpt_path}")
    
        model_csv_path = export_model_trial_csv_from_checkpoint(
            ckpt_path=ckpt_path,
            args=args,
            run_dir=root,
            isi_ms_for_export=int(getattr(args, "analysis_isi", args.isi_schedule[0])),
            out_name=str(getattr(args, "model_trial_csv_name", "model_trial.csv")),
        )
    
        if bool(getattr(args, "run_post_analysis", False)) and model_csv_path is not None:
            human_csv = Path(str(getattr(args, "human_trial_csv", ""))).expanduser()
            out_sub = Path(str(getattr(args, "analysis_out_subdir", "analysis/position_effect")))
            out_dir = root / out_sub
    
            run_position_effect_analysis(
                human_csv=human_csv,
                model_csv=model_csv_path,
                out_dir=out_dir,
                isi=int(getattr(args, "analysis_isi", args.isi_schedule[0])),
                human_id_col=getattr(args, "analysis_human_id_col", "subject_id"),
                model_id_col=getattr(args, "analysis_model_id_col", None),
                n_boot=int(getattr(args, "analysis_n_boot", 5000)),
                seed=int(getattr(args, "analysis_seed", 0)),
            )

        print(f"[eval_only done] outputs saved under: {root.resolve()}")
        return

    # If not sweeping: run a single experiment in save_dir (as-is)
    if not args.sweep:
        sweep_info = {
            "sigma_other_noise": float(args.sigma_other_noise),
            "p_other_noise": float(args.p_other_noise),
            "sigma_silence_noise": float(args.sigma_silence_noise),
            "rt_p_thresh": float(args.rt_p_thresh),
            "rt_k_consec": int(args.rt_k_consec),
            "time_cost_w": float(args.time_cost_w),
            "online_cost_w_time": float(args.online_cost_w_time),
            "derive_time_cost_from_timeout": bool(args.derive_time_cost_from_timeout),
            "timeout_ms": (None if args.timeout_ms is None else float(args.timeout_ms)),
            "hard_timeout_enabled": False,
            "isi_schedule": list(args.isi_schedule),
            "use_mask_loss": bool(args.use_mask_loss),
            "important_token_indices": args.important_token_indices,
        }
        run_curriculum_training(args=args, run_dir=root, sweep_info=sweep_info)
        return

    # Sweeping mode: build grid over provided lists; if empty, use current scalar
    sigmas_other = parse_list_of_floats(args.sweep_sigma_other) or [float(args.sigma_other_noise)]
    p_others = parse_list_of_floats(args.sweep_p_other) or [float(args.p_other_noise)]
    sigmas_sil = parse_list_of_floats(args.sweep_sigma_silence) or [float(args.sigma_silence_noise)]
    rt_ps = parse_list_of_floats(args.sweep_rt_p_thresh) or [float(args.rt_p_thresh)]
    rt_ks = parse_list_of_ints(args.sweep_rt_k_consec) or [int(args.rt_k_consec)]
    token_mss = parse_list_of_ints(args.sweep_token_ms) or [int(args.token_ms)]
    time_cost_ws = parse_list_of_floats(args.sweep_time_cost_w) or [float(args.time_cost_w)]

    combos = list(itertools.product(sigmas_other, p_others, sigmas_sil, rt_ps, rt_ks, token_mss, time_cost_ws))
    print(f"[sweep] total combinations: {len(combos)}")
    print(f"[sweep] sigma_other: {sigmas_other}")
    print(f"[sweep] p_other: {p_others}")
    print(f"[sweep] sigma_silence: {sigmas_sil}")
    print(f"[sweep] rt_p_thresh: {rt_ps}")
    print(f"[sweep] rt_k_consec: {rt_ks}")
    print(f"[sweep] token_ms: {token_mss}")
    print(f"[sweep] time_cost_w: {time_cost_ws}")


    for i, (sigma_other, p_other, sigma_sil, rt_p, rt_k, token_ms, w_cost) in enumerate(combos, start=1):
        run_args = copy.deepcopy(args)
        run_args.sigma_other_noise = float(sigma_other)
        run_args.p_other_noise = float(p_other)
        run_args.sigma_silence_noise = float(sigma_sil)
        run_args.rt_p_thresh = float(rt_p)
        run_args.rt_k_consec = int(rt_k)
        run_args.token_ms = int(token_ms)
        run_args.time_cost_w = float(w_cost)
        # keep monitor consistent unless user explicitly wants mismatch
        run_args.online_cost_w_time = float(w_cost)

        parts = {
            "i": i,
            "sig_other": sigma_other,
            "p_other": p_other,
            "sig_sil": sigma_sil,
            "rtp": rt_p,
            "rtk": rt_k,
            "tokms": token_ms,
            "w": w_cost,
        }

        run_name = make_run_name(parts)
        run_dir = root / run_name

        print("\n" + "=" * 80)
        print(f"[sweep] ({i}/{len(combos)}) -> {run_dir.name}")
        print("=" * 80)

        sweep_info = {
            "sigma_other_noise": float(sigma_other),
            "p_other_noise": float(p_other),
            "sigma_silence_noise": float(sigma_sil),
            "rt_p_thresh": float(rt_p),
            "rt_k_consec": int(rt_k),
            "token_ms": int(token_ms),
            "time_cost_w": float(w_cost),
            "online_cost_w_time": float(run_args.online_cost_w_time),
            "derive_time_cost_from_timeout": bool(run_args.derive_time_cost_from_timeout),
            "timeout_ms": (None if run_args.timeout_ms is None else float(run_args.timeout_ms)),
            "hard_timeout_enabled": False,
            "isi_schedule": list(run_args.isi_schedule),
            "combo_index": i,
            "combo_total": len(combos),
            "use_mask_loss": bool(run_args.use_mask_loss),
            "important_token_indices": run_args.important_token_indices,
        }

        run_curriculum_training(args=run_args, run_dir=run_dir, sweep_info=sweep_info)

    print(f"\n[sweep done] all runs saved under: {root.resolve()}")


if __name__ == "__main__":
    main()
