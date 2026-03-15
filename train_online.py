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

    def _seed_for(self, idx: int, trial: int) -> int:
        return self.seed + idx * 1000 + trial * 17

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.Y[idx]  # (10,) in {4,5,6}
        freqs_block = self.X[idx].numpy().astype(np.float32)  # (10,8)

        out = np.zeros((10, self.trial_T_tokens, self.input_dim), dtype=np.float32)
        for tr in range(10):
            rng = np.random.default_rng(self._seed_for(idx, tr))
            out[tr] = self._render_trial_tokens_onehot(freqs_block[tr], rng=rng)

        x = torch.from_numpy(out.reshape(10 * self.trial_T_tokens, self.input_dim))
        if getattr(self, "assert_labels", False):
            if not torch.all((y == 4) | (y == 5) | (y == 6)):
                raise RuntimeError(f"[dataset assert] Bad y at idx={idx}: unique={torch.unique(y).tolist()}")
        return x, y


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


def make_post_deviant_window_mask_tokens(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_ms: int,
    window_ms: int,
    start_offset_ms: int = 0,
) -> torch.Tensor:
    trial_id = (abs_t // trial_T_tokens).long()
    within = (abs_t % trial_T_tokens).long()

    dev_end_bt = deviant_end_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    dev_end_for_t = dev_end_bt[:, trial_id]

    offset_T = max(0, int(round(start_offset_ms / token_ms)))
    win_T = max(1, int(round(window_ms / token_ms)))

    lo = dev_end_for_t + offset_T
    hi = dev_end_for_t + offset_T + win_T
    return (within > lo) & (within <= hi)


def compute_rt_from_logits(
    logits: torch.Tensor,          # (B,10,trial_T_tokens,3)
    y_cls: torch.Tensor,           # (B,10) in {0,1,2}
    dev_end: torch.Tensor,         # (B,10)
    p_thresh: float,
    k_consec: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      rt_tokens: (B,10) number of tokens needed to first reach correct+confident detection
                 after deviant end, counted from 1.
                 Example:
                   first valid token immediately after deviant end -> 1
                   second valid token -> 2
      found:     (B,10) bool
    """
    y_cls = y_cls.long()
    mn = int(y_cls.min().item())
    mx = int(y_cls.max().item())
    if mn < 0 or mx >= 3:
        raise ValueError(f"compute_rt_from_logits got invalid y_cls range [{mn},{mx}] (expected 0..2).")

    B, N, Tt, C = logits.shape
    assert N == 10 and C == 3

    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)

    y = y_cls.unsqueeze(-1).expand(B, N, Tt)
    correct = (pred == y)

    py = probs.gather(dim=-1, index=y_cls.view(B, N, 1, 1).expand(B, N, Tt, 1)).squeeze(-1)
    confident = py >= float(p_thresh)
    ok = correct & confident

    # not found -> -1
    rt = torch.full((B, N), -1, dtype=torch.long, device=logits.device)
    found = torch.zeros((B, N), dtype=torch.bool, device=logits.device)

    K = int(max(1, k_consec))
    for b in range(B):
        for tr in range(N):
            t0 = int(dev_end[b, tr].item()) + 1  # first token AFTER deviant end
            if t0 >= Tt:
                continue

            run = 0
            for t in range(t0, Tt):
                if bool(ok[b, tr, t].item()):
                    run += 1
                    if run >= K:
                        first_t = t - K + 1
                        # 1-based token count:
                        # first valid token after dev_end => first_t == t0 => rt = 1
                        rt[b, tr] = (first_t - t0) + 1
                        found[b, tr] = True
                        break
                else:
                    run = 0

    return rt, found


# -------------------------
# Token-loss helpers
# -------------------------
def _weighted_token_ce(
    logits: torch.Tensor,   # (N,3)
    target: torch.Tensor,   # (N,)
    weights: torch.Tensor,  # (N,)
) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, reduction="none")
    w = weights.float().clamp(min=0.0)
    return (loss * w).sum() / (w.sum() + 1e-8)


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
# TBPTT runner
# -------------------------
def _run_block_through_tbptt(
    model: PredictiveGRU,
    x: torch.Tensor,              # (B,T,D)
    y_pos_456: torch.Tensor,      # (B,10) in {4,5,6}
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
    return_full_logits: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if token_loss_mode not in ("uniform", "exp"):
        raise ValueError("token_loss_mode must be 'uniform' or 'exp'")

    B, T, D = x.shape
    h = None
    collected_end_logits = []
    token_loss_sum = x.new_tensor(0.0)
    token_count = x.new_tensor(0.0)

    logits_chunks = [] if return_full_logits else None

    for s in range(0, T, int(chunk_len)):
        e = min(s + int(chunk_len), T)
        x_in = x[:, s:e, :]
        h_seq, h = model.forward_chunk(x_in, h0=h)           # (B,L,H)
        logits = model.classify_tokens(h_seq)                # (B,L,3)

        if return_full_logits:
            logits_chunks.append(logits.detach())

        abs_t = torch.arange(s, e, device=x.device)
        trial_id = (abs_t // int(trial_T_tokens)).long()
        within = (abs_t % int(trial_T_tokens)).long()

        if int(tok_window_ms) > 0:
            mask = make_post_deviant_window_mask_tokens(
                abs_t=abs_t,
                y_pos_456=y_pos_456,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
                token_ms=int(token_ms),
                window_ms=int(tok_window_ms),
                start_offset_ms=int(tok_start_offset_ms),
            )
        else:
            dev_end_bt = deviant_end_token_in_trial(y_pos_456, tone_T=int(tone_T), isi_T=int(isi_T))
            dev_end_for_t = dev_end_bt[:, trial_id]
            mask = (within > dev_end_for_t)

        y_cls = labels_to_class_index(y_pos_456)
        target = y_cls[:, trial_id]

        if mask.any():
            if token_loss_mode == "uniform":
                loss_mean = token_ce(logits[mask], target[mask])
                n = target[mask].numel()
                token_loss_sum = token_loss_sum + loss_mean * n
                token_count = token_count + n
            else:
                dev_end_bt = deviant_end_token_in_trial(y_pos_456, tone_T=int(tone_T), isi_T=int(isi_T))
                dev_end_for_t = dev_end_bt[:, trial_id]

                offset_T = max(0, int(round(int(tok_start_offset_ms) / int(token_ms))))
                anchor = dev_end_for_t + 1 + offset_T

                dist = (within.unsqueeze(0) - anchor).clamp(min=0)
                w = torch.exp(-dist.float() / float(token_tau))
                w = torch.clamp(w, min=float(token_w_min), max=1.0)

                loss_w = _weighted_token_ce(logits[mask], target[mask], w[mask])
                n = target[mask].numel()
                token_loss_sum = token_loss_sum + loss_w * n
                token_count = token_count + n

        m_end = (end_idx >= s) & (end_idx < e)
        if m_end.any():
            rel = (end_idx[m_end] - s).long()
            h_end = h_seq.index_select(dim=1, index=rel)
            collected_end_logits.append(model.classify_from_states(h_end))

        h = h.detach()

    if len(collected_end_logits) == 0:
        raise RuntimeError("No trial-end states collected. Check end_idx and T.")

    logits_end = torch.cat(collected_end_logits, dim=1)
    token_loss = token_loss_sum / (token_count + 1e-8)

    logits_all = None
    if return_full_logits:
        logits_all = torch.cat(logits_chunks, dim=1)

    return logits_end, token_loss, logits_all


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
# Train / Eval
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
    rt_p_thresh: float = 0.7,
    rt_k_consec: int = 3,
    debug: bool = False,
    debug_steps: int = 0,
    log_every: int = 50,
    token_loss_mode: str = "uniform",
    token_tau: float = 50.0,
    token_w_min: float = 0.05,
    debug_labels: bool = False,
    debug_labels_fatal: bool = False,
    debug_labels_dump: bool = False,
    debug_labels_dump_dir: Optional[Path] = None,
    debug_labels_max_dumps: int = 10,
    debug_labels_first_n_batches: int = 0,
    epoch_global: int = 0,
) -> dict:
    model.train()
    ce_end = nn.CrossEntropyLoss(reduction="mean")
    ce_tok = nn.CrossEntropyLoss(reduction="mean")

    total_end = 0.0
    total_tok = 0.0
    n_examples = 0

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []

    # ---- NEW: RT tracking on TRAIN ----
    rt_tokens_sum = 0.0
    rt_ms_sum = 0.0
    rt_n = 0
    rt_miss = 0

    n_steps = len(loader)
    t_epoch0 = time.time()
    ema_step = None
    ema_beta = 0.9

    step = 0
    for x, y in loader:
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

        # ---- IMPORTANT: return_full_logits=True so we can compute train RT ----
        logits_end, token_loss, logits_all = _run_block_through_tbptt(
            model=model,
            x=x,
            y_pos_456=y,
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
            return_full_logits=True,
        )

        y_cls = labels_to_class_index(y)
        end_loss = ce_end(logits_end.reshape(-1, 3), y_cls.reshape(-1))

        total_loss = end_loss + float(lambda_token) * token_loss
        total_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        optimizer.step()

        total_end += float(end_loss.item()) * B
        total_tok += float(token_loss.item()) * B

        with torch.no_grad():
            probs = torch.softmax(logits_end, dim=-1).detach().cpu().numpy().reshape(-1, 3)
            pred = probs.argmax(axis=1)
            yt = y_cls.detach().cpu().numpy().reshape(-1)
            y_true_all.append(yt)
            y_pred_all.append(pred)
            y_prob_all.append(probs)

        # ---- NEW: compute TRAIN RT ----
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

                rt_tokens_cpu, found_cpu = compute_rt_from_logits(
                    logits=logits_trial,
                    y_cls=y_cls_cpu,
                    dev_end=dev_end_cpu,
                    p_thresh=float(rt_p_thresh),
                    k_consec=int(rt_k_consec),
                )

                if found_cpu.any():
                    rt_vals_tokens = rt_tokens_cpu[found_cpu].float()
                    rt_tokens_sum += float(rt_vals_tokens.sum().item())

                    rt_vals_ms = rt_vals_tokens * float(token_ms)
                    rt_ms_sum += float(rt_vals_ms.sum().item())

                    rt_n += int(rt_vals_tokens.numel())

                rt_miss += int((~found_cpu).sum().item())

        if debug and step <= int(debug_steps):
            w1 = next(model.parameters()).detach().float().cpu()
            delta = (w1 - w0).abs().mean().item() if w0 is not None else float("nan")
            batch_acc = (torch.softmax(logits_end, dim=-1).argmax(dim=-1) == y_cls).float().mean().item()
            print(
                f"[debug step {step}] mean|Δparam|={delta:.6e} "
                f"end={end_loss.item():.4f} tok={float(token_loss.item()):.4f} "
                f"acc={batch_acc:.3f} logits_mean={logits_end.detach().float().mean().item():.4f}"
            )

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
                    rt_tokens_cpu, found_cpu = compute_rt_from_logits(
                        logits=logits_trial,
                        y_cls=y_cls_cpu,
                        dev_end=dev_end_cpu,
                        p_thresh=float(rt_p_thresh),
                        k_consec=int(rt_k_consec),
                    )
                    if found_cpu.any():
                        batch_mean_rt_tokens = float(rt_tokens_cpu[found_cpu].float().mean().item())
                        batch_mean_rt_ms = batch_mean_rt_tokens * float(token_ms)
                        batch_rt_msg = f"batch_meanRT={batch_mean_rt_tokens:.2f}tok/{batch_mean_rt_ms:.1f}ms"

            print(
                f"[train step {step:>4d}/{n_steps}] "
                f"dt={dt:.2f}s ema={ema_step:.2f}s "
                f"ETA_epoch={_fmt_hms(eta_epoch)} elapsed={_fmt_hms(elapsed)} "
                f"end={end_loss.item():.4f} tok={float(token_loss.item()):.4f} "
                f"acc={batch_acc:.3f} {batch_rt_msg}"
            )

    denom = max(1, n_examples)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.zeros((0, 3), dtype=np.float32)

    acc = float((y_true == y_pred).mean()) if y_true.size > 0 else float("nan")
    f1 = safe_f1_macro(y_true, y_pred) if y_true.size > 0 else float("nan")
    auc = safe_auc_ovr(y_true, y_prob, n_classes=3) if y_true.size > 0 else float("nan")

    mean_rt_tokens = (rt_tokens_sum / rt_n) if rt_n > 0 else float("nan")
    mean_rt_ms = (rt_ms_sum / rt_n) if rt_n > 0 else float("nan")

    return {
        "end_loss": total_end / denom,
        "token_loss": total_tok / denom,
        "total_loss": (total_end + float(lambda_token) * total_tok) / denom,
        "acc": acc,
        "f1_macro": float(f1),
        "auc_ovr": float(auc),
        "mean_rt_tokens": float(mean_rt_tokens),
        "mean_rt_ms": float(mean_rt_ms),
        "rt_found": int(rt_n),
        "rt_miss": int(rt_miss),
        "epoch_time_sec": time.time() - t_epoch0,
    }


@torch.no_grad()
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
    rt_p_thresh: float,
    rt_k_consec: int,
    debug_labels: bool = False,
    debug_labels_fatal: bool = False,
    debug_labels_dump: bool = False,
    debug_labels_dump_dir: Optional[Path] = None,
    debug_labels_max_dumps: int = 10,
    debug_labels_first_n_batches: int = 0,
    epoch_global: int = 0,
) -> dict:
    model.eval()
    ce_end = nn.CrossEntropyLoss(reduction="mean")
    ce_tok = nn.CrossEntropyLoss(reduction="mean")

    total_end = 0.0
    total_tok = 0.0
    n_examples = 0

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []

    # ---- NEW ----
    rt_tokens_sum = 0.0
    rt_ms_sum = 0.0
    rt_n = 0
    rt_miss = 0

    for x, y in loader:
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
        B, T, D = x.shape
        n_examples += B

        end_idx = infer_end_indices_from_T(T, trials_per_block=10).to(device)

        logits_end, token_loss, logits_all = _run_block_through_tbptt(
            model=model,
            x=x,
            y_pos_456=y,
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
            return_full_logits=True,
        )

        y_cls = labels_to_class_index(y)
        end_loss = ce_end(logits_end.reshape(-1, 3), y_cls.reshape(-1))

        total_end += float(end_loss.item()) * B
        total_tok += float(token_loss.item()) * B

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

                rt_tokens_cpu, found_cpu = compute_rt_from_logits(
                    logits=logits_trial_cpu,
                    y_cls=y_cls_cpu,
                    dev_end=dev_end_cpu,
                    p_thresh=float(rt_p_thresh),
                    k_consec=int(rt_k_consec),
                )

                if found_cpu.any():
                    rt_vals_tokens = rt_tokens_cpu[found_cpu].float()
                    rt_tokens_sum += float(rt_vals_tokens.sum().item())

                    rt_vals_ms = rt_vals_tokens * float(token_ms)
                    rt_ms_sum += float(rt_vals_ms.sum().item())

                    rt_n += int(rt_vals_tokens.numel())

                rt_miss += int((~found_cpu).sum().item())

    denom = max(1, n_examples)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.zeros((0, 3), dtype=np.float32)

    acc = float((y_true == y_pred).mean()) if y_true.size > 0 else float("nan")
    f1 = safe_f1_macro(y_true, y_pred) if y_true.size > 0 else float("nan")
    auc = safe_auc_ovr(y_true, y_prob, n_classes=3) if y_true.size > 0 else float("nan")

    mean_rt_tokens = (rt_tokens_sum / rt_n) if rt_n > 0 else float("nan")
    mean_rt_ms = (rt_ms_sum / rt_n) if rt_n > 0 else float("nan")

    return {
        "end_loss": total_end / denom,
        "token_loss": total_tok / denom,
        "total_loss": (total_end + float(lambda_token) * total_tok) / denom,
        "acc": acc,
        "f1_macro": float(f1),
        "auc_ovr": float(auc),
        "mean_rt_tokens": float(mean_rt_tokens),
        "mean_rt_ms": float(mean_rt_ms),
        "rt_found": int(rt_n),
        "rt_miss": int(rt_miss),
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

    if not _HAVE_PANDAS:
        print("[export_model_csv] pandas not installed; cannot write CSV.")
        return None

    import pandas as pd

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
            logits_end, token_loss, logits_all = _run_block_through_tbptt(
                model=model,
                x=x,
                y_pos_456=y_pos_456,
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
                return_full_logits=True,
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

            rt_tokens_cpu, found_cpu = compute_rt_from_logits(
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

    # ---- base dataset just to get input_dim + label sanity + splits ----
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

    train_idx, val_idx = split_indices(len(base_ds), args.val_split, args.seed)

    cfg = ModelConfig(
        input_dim=int(base_ds.input_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        layer_norm=bool(args.layer_norm),
    )
    model = PredictiveGRU(cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    # ---- resume (optional) ----
    start_epoch_global = 1
    best_val = float("inf")
    best_epoch = 0
    if getattr(args, "resume", ""):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "optim_state" in ckpt:
            optim.load_state_dict(ckpt["optim_state"])
        start_epoch_global = int(ckpt.get("epoch_global", ckpt.get("epoch", 0))) + 1
        best_val = float(ckpt.get("best_val", best_val))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        print(f"[resume] loaded: {args.resume} start_epoch_global={start_epoch_global}")

    # ---- logs ----
    jsonl_path = run_dir / "logs" / "metrics.jsonl"
    csv_path = run_dir / "logs" / "metrics.csv"
    header = [
        "epoch_global", "stage", "isi_ms", "token_ms",
        "train_total_loss", "train_end_loss", "train_token_loss", "train_acc", "train_f1_macro", "train_auc_ovr",
        "train_mean_rt_tokens", "train_mean_rt_ms", "train_rt_found", "train_rt_miss",
        "val_total_loss", "val_end_loss", "val_token_loss", "val_acc", "val_f1_macro", "val_auc_ovr",
        "val_mean_rt_tokens", "val_mean_rt_ms", "val_rt_found", "val_rt_miss",
        "best_val", "best_epoch",
        "best_target_isi_val", "best_target_isi_epoch", "best_target_isi",
        "time_elapsed_sec",
    ]


    patience = int(getattr(args, "early_stop_patience", 0))
    min_delta = float(getattr(args, "early_stop_min_delta", 0.0))

    # ---- target-ISI best tracking (e.g., 700) ----
    best_target_isi = int(getattr(args, "analysis_isi", 700))
    best_target_val = float("inf")
    best_target_epoch = 0
    best_target_path = run_dir / f"best_isi{best_target_isi}.pt"

    history: List[Dict[str, Any]] = []
    t_run0 = time.time()
    epoch_global = start_epoch_global - 1

    for stage, isi_ms in enumerate(args.isi_schedule, start=1):
        isi_ms = int(isi_ms)
        print(f"\n[curriculum] stage={stage}/{len(args.isi_schedule)}  isi_ms={isi_ms}")

        # ---- stage-local early stopping state (RESET EACH STAGE) ----
        stage_best_val = float("inf")
        stage_bad_count = 0

        # ---- stage-best snapshot (captured ONLY when improved) ----
        stage_best_state: Optional[Dict[str, torch.Tensor]] = None
        stage_best_optim: Optional[Dict[str, Any]] = None
        stage_best_epoch_global: Optional[int] = None

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

        # ---- epoch loop for this stage ----
        for _e in range(1, int(args.epochs_per_isi) + 1):
            epoch_global += 1

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
                tok_window_ms=int(getattr(args, "tok_window_ms", 0)),
                tok_start_offset_ms=int(getattr(args, "tok_start_offset_ms", 0)),
                rt_p_thresh=float(args.rt_p_thresh),
                rt_k_consec=int(args.rt_k_consec),
                debug=bool(getattr(args, "debug", False)),
                debug_steps=int(getattr(args, "debug_steps", 0)),
                log_every=int(args.log_every),
                token_loss_mode=str(args.token_loss_mode),
                token_tau=float(args.token_tau),
                token_w_min=float(args.token_w_min),
                debug_labels=bool(getattr(args, "debug_labels", False)),
                debug_labels_fatal=bool(getattr(args, "debug_labels_fatal", False)),
                debug_labels_dump=bool(getattr(args, "debug_labels_dump", False)),
                debug_labels_dump_dir=(run_dir / "logs" / "bad_batches"),
                debug_labels_max_dumps=int(getattr(args, "debug_labels_max_dumps", 10)),
                debug_labels_first_n_batches=int(getattr(args, "debug_labels_first_n_batches", 0)),
                epoch_global=int(epoch_global),
            )


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
                tok_window_ms=int(getattr(args, "tok_window_ms", 0)),
                tok_start_offset_ms=int(getattr(args, "tok_start_offset_ms", 0)),
                rt_p_thresh=float(args.rt_p_thresh),
                rt_k_consec=int(args.rt_k_consec),
                debug_labels=bool(getattr(args, "debug_labels", False)),
                debug_labels_fatal=bool(getattr(args, "debug_labels_fatal", False)),
                debug_labels_dump=bool(getattr(args, "debug_labels_dump", False)),
                debug_labels_dump_dir=(run_dir / "logs" / "bad_batches"),
                debug_labels_max_dumps=int(getattr(args, "debug_labels_max_dumps", 10)),
                debug_labels_first_n_batches=int(getattr(args, "debug_labels_first_n_batches", 0)),
                epoch_global=int(epoch_global),
            )

            elapsed = time.time() - t_run0

            # ---- GLOBAL best (across all stages): best.pt ----
            improved_global = (va["total_loss"] < (best_val - min_delta))
            if improved_global:
                best_val = float(va["total_loss"])
                best_epoch = int(epoch_global)

            # ---- TARGET-ISI best: best_isi{analysis_isi}.pt ----
            improved_target = False
            if int(isi_ms) == int(best_target_isi):
                improved_target = (va["total_loss"] < (best_target_val - min_delta))
                if improved_target:
                    best_target_val = float(va["total_loss"])
                    best_target_epoch = int(epoch_global)

            # ---- STAGE-local best + early stopping ----
            improved_stage = (va["total_loss"] < (stage_best_val - min_delta))
            if improved_stage:
                stage_best_val = float(va["total_loss"])
                stage_bad_count = 0

                # snapshot stage-best model+optim (cpu clone)
                stage_best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
                stage_best_optim = copy.deepcopy(optim.state_dict())
                stage_best_epoch_global = int(epoch_global)
            else:
                stage_bad_count += 1

            target_msg = ""
            if int(isi_ms) == int(best_target_isi):
                target_msg = f" target_best={best_target_val:.4f}@{best_target_epoch}"

            print(
                f"[epoch {epoch_global:04d}] (stage={stage} isi={isi_ms} token_ms={ds.token_ms}) "
                f"train: loss={tr['total_loss']:.4f} acc={tr['acc']:.4f} f1={tr['f1_macro']:.4f} auc={tr['auc_ovr']:.4f} "
                f"meanRT={tr['mean_rt_tokens']:.2f}tok/{tr['mean_rt_ms']:.1f}ms found={tr['rt_found']} miss={tr['rt_miss']} | "
                f"val: loss={va['total_loss']:.4f} acc={va['acc']:.4f} f1={va['f1_macro']:.4f} auc={va['auc_ovr']:.4f} "
                f"meanRT={va['mean_rt_tokens']:.2f}tok/{va['mean_rt_ms']:.1f}ms found={va['rt_found']} miss={va['rt_miss']} | "
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
                "train_acc": tr["acc"],
                "train_f1_macro": tr["f1_macro"],
                "train_auc_ovr": tr["auc_ovr"],
                "train_mean_rt_tokens": tr["mean_rt_tokens"],
                "train_mean_rt_ms": tr["mean_rt_ms"],
                "train_rt_found": tr["rt_found"],
                "train_rt_miss": tr["rt_miss"],

                "val_total_loss": va["total_loss"],
                "val_end_loss": va["end_loss"],
                "val_token_loss": va["token_loss"],
                "val_acc": va["acc"],
                "val_f1_macro": va["f1_macro"],
                "val_auc_ovr": va["auc_ovr"],
                "val_mean_rt_tokens": va["mean_rt_tokens"],
                "val_mean_rt_ms": va["mean_rt_ms"],
                "val_rt_found": va["rt_found"],
                "val_rt_miss": va["rt_miss"],

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

            # ---- stage-local early stop: break out of epoch loop ----
            if patience > 0 and stage_bad_count >= patience:
                print(
                    f"[early_stop] stage {stage} (isi={isi_ms}) patience reached: "
                    f"{stage_bad_count}/{patience}. End this stage."
                )
                break  # IMPORTANT: do NOT restore here; do it once after loop

        # ---- STAGE END: restore ONCE to stage-best before next stage ----
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

    # ---- final plots / summary ----
    plot_history(history, run_dir / "plots" / "dummy.png")
    print(f"[done] Saved run to: {run_dir.resolve()}")
    print("  - best.pt / last.pt")
    if best_target_path.exists():
        print(f"  - {best_target_path.name}  (best on isi={best_target_isi})")
    print("  - logs/metrics.jsonl + logs/metrics.csv")
    print("  - plots/acc.png f1.png auc.png loss.png")

    # ---- post-run analysis (optional): export model_trial.csv from best_isi{analysis_isi}.pt first ----
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
            print("[analysis] model_csv export failed; skip analysis.")
# -------------------------
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

    # RT criterion (eval only; can be overridden by sweeps)
    p.add_argument("--rt_p_thresh", type=float, default=0.7)
    p.add_argument("--rt_k_consec", type=int, default=3)

    # resume
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt (best.pt/last.pt)")

    # debug / sanity
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_steps", type=int, default=3)
    p.add_argument("--max_blocks", type=int, default=0)
    p.add_argument("--tok_window_ms", type=int, default=300)
    p.add_argument("--tok_start_offset_ms", type=int, default=0)
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


    args = p.parse_args()

    # parse isi schedule
    args.isi_schedule = parse_list_of_ints(args.isi_schedule)
    if not args.isi_schedule:
        raise ValueError("--isi_schedule 不能为空。例：0,50,300,700")

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

    # If not sweeping: run a single experiment in save_dir (as-is)
    if not args.sweep:
        sweep_info = {
            "sigma_other_noise": float(args.sigma_other_noise),
            "p_other_noise": float(args.p_other_noise),
            "sigma_silence_noise": float(args.sigma_silence_noise),
            "rt_p_thresh": float(args.rt_p_thresh),
            "rt_k_consec": int(args.rt_k_consec),
            "isi_schedule": list(args.isi_schedule),
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

    combos = list(itertools.product(sigmas_other, p_others, sigmas_sil, rt_ps, rt_ks, token_mss))
    print(f"[sweep] total combinations: {len(combos)}")
    print(f"[sweep] sigma_other: {sigmas_other}")
    print(f"[sweep] p_other: {p_others}")
    print(f"[sweep] sigma_silence: {sigmas_sil}")
    print(f"[sweep] rt_p_thresh: {rt_ps}")
    print(f"[sweep] rt_k_consec: {rt_ks}")
    print(f"[sweep] token_ms: {token_mss}")


    for i, (sigma_other, p_other, sigma_sil, rt_p, rt_k, token_ms) in enumerate(combos, start=1):
        run_args = copy.deepcopy(args)
        run_args.sigma_other_noise = float(sigma_other)
        run_args.p_other_noise = float(p_other)
        run_args.sigma_silence_noise = float(sigma_sil)
        run_args.rt_p_thresh = float(rt_p)
        run_args.rt_k_consec = int(rt_k)
        run_args.token_ms = int(token_ms)

        parts = {
            "i": i,
            "sig_other": sigma_other,
            "p_other": p_other,
            "sig_sil": sigma_sil,
            "rtp": rt_p,
            "rtk": rt_k,
            "tokms": token_ms,
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
            "isi_schedule": list(run_args.isi_schedule),
            "combo_index": i,
            "combo_total": len(combos),
        }

        run_curriculum_training(args=run_args, run_dir=run_dir, sweep_info=sweep_info)

    print(f"\n[sweep done] all runs saved under: {root.resolve()}")


if __name__ == "__main__":
    main()