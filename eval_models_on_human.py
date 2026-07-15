#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained PredictiveGRU checkpoints on the SAME trials as in human_trial.csv.

Outputs per model:
  - metrics.json
  - trial_level.csv (one row per human trial)
And global:
  - summary.csv / summary.json

Assumptions about human_trial.csv:
  Required columns (names can vary, script will try to auto-detect):
    - position: deviant position in {4,5,6}
    - rt_ms: human reaction time in ms (can be negative; NaN allowed)
    - isi_ms: ISI in ms (if missing, you can pass --isi_ms_default)
    - f_std: standard frequency in Hz
    - f_dev: deviant frequency in Hz

If your CSV contains the full 8-tone sequence already, you can adapt build_trial_freqs().
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# You already have these in your repo
from model import PredictiveGRU, ModelConfig  # type: ignore
from stimulus_encoding import (
    StimulusEncodingConfig,
    freq_to_bin_erb as shared_freq_to_bin_erb,
    make_erb_edges as shared_make_erb_edges,
    render_trial_tokens_from_freqs,
)
from rt_readout import (
    prepare_rt_readout,
    run_rt_readout_sweeps,
    prior_prob_from_position,
    prior_surprisal_from_position,
    build_participant_level_rt_eval,
    build_group_level_rt_eval,
)

# optional deps
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("This script requires pandas. Please install it in your env.") from e

try:
    from sklearn.metrics import f1_score, roc_auc_score
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

try:
    from scipy.stats import pearsonr, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# -------------------------
# Column detection helpers
# -------------------------
def _pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)

    pos = _pick_first_existing(cols, ["position", "pos", "deviant_pos", "dev_pos", "deviant_position"])
    rt = _pick_first_existing(cols, ["rt_ms", "human_rt_ms", "rt", "reaction_time_ms", "reaction_time", "RT_ms", "RT"])
    isi = _pick_first_existing(cols, ["isi_ms", "isi", "ISI", "ISI_ms"])
    f_std = _pick_first_existing(cols, ["f_std", "std_hz", "standard_hz", "f_standard", "standard_freq_hz", "standard_freq", "std_freq"])
    f_dev = _pick_first_existing(cols, ["f_dev", "dev_hz", "deviant_hz", "f_deviant", "deviant_freq_hz", "deviant_freq", "dev_freq"])
    sid = _pick_first_existing(cols, ["subject_id", "participant_id", "subj", "participant", "id", "ID"])
    run = _pick_first_existing(cols, ["run_id", "run", "run_name"])
    trial_order = _pick_first_existing(cols, ["trial_index_within_run", "trial_order", "trial_index", "trial", "trial_id", "order"])

    missing = [k for k, v in {
        "position": pos, "rt_ms": rt, "f_std": f_std, "f_dev": f_dev
    }.items() if v is None]
    if missing:
        raise ValueError(
            f"human_trial.csv missing required columns: {missing}. "
            f"Available columns: {cols}"
        )

    out = {"position": pos, "rt_ms": rt, "f_std": f_std, "f_dev": f_dev}
    if isi is not None:
        out["isi_ms"] = isi
    if sid is not None:
        out["subject_id"] = sid
    if run is not None:
        out["run_id"] = run
    if trial_order is not None:
        out["trial_order"] = trial_order
    return out


# -------------------------
# ERB binning + one-hot rendering (match your training)
# -------------------------
def hz_to_erb_rate_np(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * f_hz)


def make_erb_edges(f_min_hz: float, f_max_hz: float, n_bins: int) -> np.ndarray:
    return shared_make_erb_edges(f_min_hz=f_min_hz, f_max_hz=f_max_hz, n_bins=n_bins)


def freq_to_bin_erb(f_hz: float, edges_erb: np.ndarray) -> int:
    return shared_freq_to_bin_erb(f_hz=f_hz, edges_erb=edges_erb)


def labels_to_class_index(y_456: torch.Tensor) -> torch.Tensor:
    # {4,5,6} -> {0,1,2}
    return (y_456 - 4).long()


def deviant_end_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    # y_pos_456 shape (N,)
    dev_idx = (y_pos_456 - 1).long()  # {4,5,6}->{3,4,5}
    step = int(tone_T + isi_T)
    dev_end = dev_idx * step + int(tone_T) - 1
    return dev_end  # (N,)


def compute_rt_from_logits(
    logits: torch.Tensor,     # (N, trial_T_tokens, 3)
    y_cls: torch.Tensor,      # (N,) in {0,1,2}
    dev_end: torch.Tensor,    # (N,)
    p_thresh: float,
    k_consec: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RT in tokens after deviant end (t0 = dev_end + 1).
    Found if k consecutive tokens after t0 are:
      - predicted class == y
      - prob(y) >= p_thresh
    """
    y_cls = y_cls.long()
    if int(y_cls.min()) < 0 or int(y_cls.max()) > 2:
        raise ValueError(f"Invalid y_cls range: [{int(y_cls.min())},{int(y_cls.max())}]")

    probs = torch.softmax(logits, dim=-1)         # (N,T,3)
    pred = probs.argmax(dim=-1)                   # (N,T)
    y = y_cls[:, None].expand_as(pred)            # (N,T)
    correct = (pred == y)

    py = probs.gather(dim=-1, index=y_cls[:, None, None].expand(-1, probs.shape[1], 1)).squeeze(-1)  # (N,T)
    confident = py >= float(p_thresh)
    ok = correct & confident

    N, Tt, _ = logits.shape
    rt = torch.full((N,), -1, dtype=torch.long, device=logits.device)
    found = torch.zeros((N,), dtype=torch.bool, device=logits.device)

    K = int(max(1, k_consec))
    for i in range(N):
        t0 = int(dev_end[i].item()) + 1
        if t0 >= Tt:
            continue
        run = 0
        for t in range(t0, Tt):
            if bool(ok[i, t].item()):
                run += 1
                if run >= K:
                    first_t = t - K + 1
                    rt[i] = first_t - t0
                    found[i] = True
                    break
            else:
                run = 0

    return rt, found


def build_trial_freqs(f_std: float, f_dev: float, position_456: int) -> np.ndarray:
    """
    Reconstruct the 8-tone sequence given std/dev and deviant position.
    This matches the common paradigm:
      tones 1..8; tones 1-3 are standard; deviant at pos (4/5/6); others standard.

    If your human CSV actually has the full 8-tone list, replace this function.
    """
    pos = int(position_456)
    if pos not in (4, 5, 6):
        raise ValueError(f"position must be 4/5/6, got {pos}")
    seq = np.full((8,), float(f_std), dtype=np.float32)
    seq[pos - 1] = float(f_dev)
    return seq


def render_trials_onehot(
    df: pd.DataFrame,
    col: Dict[str, str],
    *,
    tone_ms: int,
    token_ms: int,
    isi_ms: int,
    f_min_hz: float,
    f_max_hz: float,
    n_bins: int,
    add_eos: bool,
    add_bos: bool,
    eos_mode: str,
    sigma_other_noise: float,
    p_other_noise: float,
    sigma_silence_noise: float,
    seed: int,
    encoding_mode: str = "onehot",
    sigma_rf: float = 1.0,
    rf_normalization: str = "peak",
    sigma_rf_noise: float = 0.0,
    rf_noise_per_token: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """
    Returns:
      X: (N, T_tokens, D)
      y_pos: (N,) in {4,5,6}
      trial_T_tokens, tone_T, isi_T
    """
    rng = np.random.default_rng(int(seed))

    tone_ms = int(tone_ms)
    token_ms = int(token_ms)
    isi_ms = int(isi_ms)

    if tone_ms % token_ms != 0:
        raise ValueError(f"tone_ms {tone_ms} not divisible by token_ms {token_ms}")
    if isi_ms % token_ms != 0:
        raise ValueError(f"isi_ms {isi_ms} not divisible by token_ms {token_ms}")

    cfg = StimulusEncodingConfig(
        tone_ms=int(tone_ms),
        isi_ms=int(isi_ms),
        token_ms=int(token_ms),
        f_min_hz=float(f_min_hz),
        f_max_hz=float(f_max_hz),
        n_bins=int(n_bins),
        add_eos=bool(add_eos),
        add_bos=bool(add_bos),
        eos_mode=str(eos_mode),
        sigma_other_noise=float(sigma_other_noise),
        p_other_noise=float(p_other_noise),
        sigma_silence_noise=float(sigma_silence_noise),
        encoding_mode=str(encoding_mode),
        sigma_rf=float(sigma_rf),
        rf_normalization=str(rf_normalization),
        sigma_rf_noise=float(sigma_rf_noise),
        rf_noise_per_token=bool(rf_noise_per_token),
    )
    tone_T = int(cfg.tone_T)
    isi_T = int(cfg.isi_T)
    trial_T_tokens = int(cfg.trial_T_tokens)
    edges_erb = make_erb_edges(float(f_min_hz), float(f_max_hz), int(n_bins))
    D = int(cfg.input_dim)

    N = int(df.shape[0])
    X = np.zeros((N, trial_T_tokens, D), dtype=np.float32)

    y_pos = df[col["position"]].astype(int).to_numpy()
    f_std = df[col["f_std"]].astype(float).to_numpy()
    f_dev = df[col["f_dev"]].astype(float).to_numpy()

    for i in range(N):
        freqs_8 = build_trial_freqs(float(f_std[i]), float(f_dev[i]), int(y_pos[i]))
        X[i] = render_trial_tokens_from_freqs(
            freqs_8=freqs_8,
            cfg=cfg,
            rng=rng,
            edges_erb=edges_erb,
        )

    return torch.from_numpy(X.copy()), torch.from_numpy(y_pos.copy()).long(), trial_T_tokens, tone_T, isi_T


def sort_human_trials_for_sequence(df: pd.DataFrame, col: Dict[str, str]) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["_row_order"] = np.arange(len(out), dtype=int)
    sort_cols: List[str] = []
    if "subject_id" in col:
        sort_cols.append(col["subject_id"])
    if "run_id" in col:
        sort_cols.append(col["run_id"])
    if "trial_order" in col:
        out[col["trial_order"]] = pd.to_numeric(out[col["trial_order"]], errors="coerce")
        sort_cols.append(col["trial_order"])
    sort_cols.append("_row_order")
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return out


def drop_leading_trials_per_participant(
    df: pd.DataFrame,
    col: Dict[str, str],
    n_drop: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    ordered = sort_human_trials_for_sequence(df, col)
    n_drop = max(0, int(n_drop))
    if n_drop == 0:
        return ordered, {
            "n_rows_input": int(len(df)),
            "n_rows_dropped_leading_per_participant": 0,
            "n_rows_kept_after_leading_drop": int(len(ordered)),
        }

    participant_group_cols: List[str] = []
    if "subject_id" in col:
        participant_group_cols.append(col["subject_id"])
    if not participant_group_cols:
        ordered["_single_participant"] = "all"
        participant_group_cols = ["_single_participant"]

    kept_groups: List[pd.DataFrame] = []
    dropped = 0
    for _, g in ordered.groupby(participant_group_cols, sort=False):
        g = g.copy().reset_index(drop=True)
        cur_drop = min(n_drop, len(g))
        dropped += cur_drop
        g = g.iloc[cur_drop:].copy().reset_index(drop=True)
        if not g.empty:
            kept_groups.append(g)

    if kept_groups:
        kept_df = pd.concat(kept_groups, ignore_index=True)
    else:
        kept_df = ordered.iloc[0:0].copy()

    return kept_df, {
        "n_rows_input": int(len(df)),
        "n_rows_dropped_leading_per_participant": int(dropped),
        "n_rows_kept_after_leading_drop": int(len(kept_df)),
    }


def build_block10_continuous_inputs(
    df: pd.DataFrame,
    col: Dict[str, str],
    *,
    tone_ms: int,
    token_ms: int,
    isi_ms: int,
    f_min_hz: float,
    f_max_hz: float,
    n_bins: int,
    add_eos: bool,
    add_bos: bool,
    eos_mode: str,
    sigma_other_noise: float,
    p_other_noise: float,
    sigma_silence_noise: float,
    seed: int,
    encoding_mode: str = "onehot",
    sigma_rf: float = 1.0,
    rf_normalization: str = "peak",
    sigma_rf_noise: float = 0.0,
    rf_noise_per_token: bool = True,
) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, int, int, int, Dict[str, int]]:
    ordered = sort_human_trials_for_sequence(df, col)
    cfg = StimulusEncodingConfig(
        tone_ms=int(tone_ms),
        isi_ms=int(isi_ms),
        token_ms=int(token_ms),
        f_min_hz=float(f_min_hz),
        f_max_hz=float(f_max_hz),
        n_bins=int(n_bins),
        add_eos=bool(add_eos),
        add_bos=bool(add_bos),
        eos_mode=str(eos_mode),
        sigma_other_noise=float(sigma_other_noise),
        p_other_noise=float(p_other_noise),
        sigma_silence_noise=float(sigma_silence_noise),
        encoding_mode=str(encoding_mode),
        sigma_rf=float(sigma_rf),
        rf_normalization=str(rf_normalization),
        sigma_rf_noise=float(sigma_rf_noise),
        rf_noise_per_token=bool(rf_noise_per_token),
    )
    tone_T = int(cfg.tone_T)
    isi_T = int(cfg.isi_T)
    trial_T_tokens = int(cfg.trial_T_tokens)
    edges_erb = make_erb_edges(float(f_min_hz), float(f_max_hz), int(n_bins))
    input_dim = int(cfg.input_dim)

    participant_group_cols: List[str] = []
    if "subject_id" in col:
        participant_group_cols.append(col["subject_id"])
    if not participant_group_cols:
        ordered["_single_participant"] = "all"
        participant_group_cols = ["_single_participant"]

    kept_groups: List[pd.DataFrame] = []
    rendered_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    dropped_incomplete = 0
    kept_trials = 0

    for _, g in ordered.groupby(participant_group_cols, sort=False):
        g = g.copy().reset_index(drop=True)
        n_full = len(g) // 10
        n_keep = n_full * 10
        dropped_incomplete += len(g) - n_keep
        if n_keep <= 0:
            continue
        g = g.iloc[:n_keep].copy().reset_index(drop=True)
        g["_block10_index"] = (np.arange(len(g)) // 10).astype(int)
        g["_trial10_index"] = (np.arange(len(g)) % 10).astype(int)
        kept_groups.append(g)
        kept_trials += len(g)

        for b in range(n_full):
            gb = g.iloc[b * 10:(b + 1) * 10]
            block = np.zeros((10 * trial_T_tokens, input_dim), dtype=np.float32)
            yb = gb[col["position"]].astype(int).to_numpy(dtype=np.int64)
            for tr, (_, row) in enumerate(gb.iterrows()):
                freqs_8 = build_trial_freqs(float(row[col["f_std"]]), float(row[col["f_dev"]]), int(row[col["position"]]))
                rng = np.random.default_rng(int(seed) + int(row["_row_order"]))
                xt = render_trial_tokens_from_freqs(
                    freqs_8=freqs_8,
                    cfg=cfg,
                    rng=rng,
                    edges_erb=edges_erb,
                )
                block[tr * trial_T_tokens:(tr + 1) * trial_T_tokens, :] = xt
            rendered_blocks.append(block)
            y_blocks.append(yb)

    if not kept_groups or not rendered_blocks:
        raise RuntimeError("No complete 10-trial blocks could be built from human CSV.")

    kept_df = pd.concat(kept_groups, ignore_index=True)
    X_blocks = torch.from_numpy(np.stack(rendered_blocks, axis=0))
    y_pos = torch.from_numpy(np.concatenate(y_blocks, axis=0)).long()
    audit = {
        "n_rows_input": int(len(df)),
        "n_rows_kept": int(kept_trials),
        "n_rows_dropped_incomplete": int(dropped_incomplete),
        "n_blocks": int(X_blocks.shape[0]),
    }
    return kept_df, X_blocks, y_pos, trial_T_tokens, tone_T, isi_T, audit


# -------------------------
# Model forward to get per-token logits and end logits
# -------------------------
@torch.no_grad()
def forward_full_logits(
    model: PredictiveGRU,
    x: torch.Tensor,   # (N,T,D)
    chunk_len: int,
) -> torch.Tensor:
    """
    Returns logits_all: (N,T,3)
    Uses model.forward_chunk + model.classify_tokens to avoid huge memory spikes.
    """
    model.eval()
    N, T, D = x.shape
    h = None
    chunks = []
    for s in range(0, T, int(chunk_len)):
        e = min(s + int(chunk_len), T)
        h_seq, h = model.forward_chunk(x[:, s:e, :], h0=h)
        logits = model.classify_tokens(h_seq)  # (N,L,3)
        chunks.append(logits)
        h = h.detach()
    return torch.cat(chunks, dim=1)


def _scale_hidden(hidden: Any, rho: float) -> Any:
    if hidden is None:
        return None
    scale = float(rho)
    if isinstance(hidden, tuple):
        return tuple(h * scale for h in hidden)
    return hidden * scale


def real_boundary_group_ids(df: pd.DataFrame, col: Dict[str, str]) -> Tuple[List[Tuple[Any, ...]], List[str]]:
    group_cols: List[str] = []
    if "subject_id" in col:
        group_cols.append(col["subject_id"])
    if "run_id" in col:
        group_cols.append(col["run_id"])
    block_col = next((c for c in ["block_id", "block_index", "block"] if c in df.columns), None)
    if block_col is not None:
        group_cols.append(block_col)
    if not group_cols:
        return [("all",)] * int(len(df)), ["<all>"]
    group_ids = [tuple(row[c] for c in group_cols) for _, row in df.iterrows()]
    return group_ids, group_cols


@torch.no_grad()
def forward_real_boundary_continuous_with_rho(
    model: PredictiveGRU,
    x_trials: torch.Tensor,  # (N,T,D)
    group_ids: List[Tuple[Any, ...]],
    chunk_len: int,
    rho: float,
) -> torch.Tensor:
    """
    Carry hidden within real participant/run/block groups and reset at group boundaries.
    This avoids blind 10-trial chunking across real run/block boundaries.
    """
    model.eval()
    if int(x_trials.shape[0]) != len(group_ids):
        raise ValueError(f"x_trials N={x_trials.shape[0]} does not match group_ids={len(group_ids)}")
    h = None
    prev_group: Optional[Tuple[Any, ...]] = None
    trial_logits: List[torch.Tensor] = []
    for i, group_id in enumerate(group_ids):
        if prev_group is None or group_id != prev_group:
            h = None
        else:
            h = _scale_hidden(h, float(rho))
        prev_group = group_id
        chunks = []
        x = x_trials[i:i + 1]
        for s in range(0, int(x.shape[1]), int(chunk_len)):
            e = min(s + int(chunk_len), int(x.shape[1]))
            h_seq, h = model.forward_chunk(x[:, s:e, :], h0=h)
            chunks.append(model.classify_tokens(h_seq))
            h = h.detach()
        trial_logits.append(torch.cat(chunks, dim=1).detach())
    return torch.cat(trial_logits, dim=0)


@torch.no_grad()
def forward_block10_continuous_with_rho(
    model: PredictiveGRU,
    x_blocks: torch.Tensor,  # (B,10*T,D)
    trial_T_tokens: int,
    chunk_len: int,
    rho: float,
) -> torch.Tensor:
    """
    Returns logits_all: (B,10*T,3)
    Within each 10-trial block, hidden state is carried across trial boundaries as:
      h0(n) = rho * h_end(n-1)
    rho=1.0 reproduces the old full-carryover behavior.
    rho=0.0 means reset at every trial boundary.
    """
    model.eval()
    B, total_T, _ = x_blocks.shape
    trial_T = int(trial_T_tokens)
    if trial_T <= 0 or total_T % trial_T != 0:
        raise ValueError(f"Invalid trial_T_tokens={trial_T} for x_blocks with total_T={total_T}")

    n_trials = total_T // trial_T
    trial_logits = []
    for tr in range(n_trials):
        xs = x_blocks[:, tr * trial_T:(tr + 1) * trial_T, :]
        h = None if tr == 0 else _scale_hidden(h, rho)
        chunks = []
        for s in range(0, trial_T, int(chunk_len)):
            e = min(s + int(chunk_len), trial_T)
            h_seq, h = model.forward_chunk(xs[:, s:e, :], h0=h)
            chunks.append(model.classify_tokens(h_seq))
            h = h.detach()
        trial_logits.append(torch.cat(chunks, dim=1))
    return torch.cat(trial_logits, dim=1)


def trial_end_logits_from_full(
    logits_all: torch.Tensor,  # (N,T,3)
    end_offset_from_trial_end: int = 0,
) -> torch.Tensor:
    idx = -1 - max(0, int(end_offset_from_trial_end))
    return logits_all[:, idx, :]


def safe_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not HAVE_SKLEARN:
        return float("nan")
    try:
        return float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        return float("nan")


def safe_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 3) -> float:
    if not HAVE_SKLEARN:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(n_classes))))
    except Exception:
        return float("nan")


def corr_pair(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Returns Pearson and Spearman correlations (and p-values if scipy installed).
    """
    out: Dict[str, float] = {}
    if x.size < 3:
        return {
            "n": float(x.size),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }

    if HAVE_SCIPY:
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        out["n"] = float(x.size)
        out["pearson_r"] = float(pr.statistic)
        out["pearson_p"] = float(pr.pvalue)
        out["spearman_r"] = float(sr.statistic)
        out["spearman_p"] = float(sr.pvalue)
        return out

    # fallback without p-values
    out["n"] = float(x.size)
    out["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    out["pearson_p"] = float("nan")
    # Spearman fallback: rank then Pearson
    xr = x.argsort().argsort().astype(float)
    yr = y.argsort().argsort().astype(float)
    out["spearman_r"] = float(np.corrcoef(xr, yr)[0, 1])
    out["spearman_p"] = float("nan")
    return out


def parse_list_of_floats(s: str) -> List[float]:
    if not str(s).strip():
        return []
    return [float(x) for x in str(s).replace(",", " ").split() if str(x).strip()]


def parse_list_of_ints(s: str) -> List[int]:
    if not str(s).strip():
        return []
    return [int(x) for x in str(s).replace(",", " ").split() if str(x).strip()]


def str2bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def split_readout_window_name(name: str) -> Tuple[str, str]:
    mapping = {
        "deviant_onset_to_deviant_offset": ("deviant_onset", "next_tone_onset"),
        "deviant_onset_to_next_tone_onset": ("deviant_onset", "next_tone_onset"),
        "deviant_onset_to_next_tone_offset": ("deviant_onset", "next_tone_offset"),
        "previous_standard_offset_to_next_tone_onset": ("previous_tone_offset", "next_tone_onset"),
        "deviant_onset_to_trial_end": ("deviant_onset", "trial_end"),
        "deviant_onset_to_second_next_tone_onset": ("deviant_onset", "trial_end"),
        "deviant_onset_to_second_next_tone_offset": ("deviant_onset", "trial_end"),
    }
    if name not in mapping:
        raise ValueError(f"Unknown cost_readout_window: {name}")
    return mapping[name]


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def make_readout_figures(
    out_dir: Path,
    summary_rows: List[Dict[str, Any]],
    condition_rows: List[Dict[str, Any]],
    participant_rows: List[Dict[str, Any]],
    first_metrics_path: Optional[Path],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] skip figures because matplotlib is unavailable: {e}")
        return
    import pandas as pd

    cond_df = pd.DataFrame(condition_rows)
    part_df = pd.DataFrame(participant_rows)
    if first_metrics_path is not None and first_metrics_path.exists():
        metrics = json.loads(first_metrics_path.read_text())
        traj_ms = metrics.get("window_metrics", {}).get("trajectory_rel_ms", [])
        traj_by_pos = metrics.get("window_metrics", {}).get("trajectory_by_position", {})
        if traj_ms and traj_by_pos:
            plt.figure(figsize=(6, 4))
            for pos in ["4", "5", "6"]:
                vals = traj_by_pos.get(pos) or traj_by_pos.get(int(pos)) or []
                if vals:
                    plt.plot(traj_ms[: len(vals)], vals, label=f"P{pos}")
            plt.xlabel("Time from deviant onset (ms)")
            plt.ylabel("p_correct")
            plt.title("p_correct trajectory by position")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "fig_p_correct_trajectory_by_position.png", dpi=150)
            plt.close()

    if not cond_df.empty:
        cost_df = cond_df[cond_df["readout_mode"] == "bayesian_cost"].copy()
        if not cost_df.empty:
            piv = cost_df.pivot_table(index="cost_threshold", columns="w", values="proportion_rt_floor_5ms", aggfunc="mean")
            if not piv.empty:
                plt.figure(figsize=(6, 4))
                plt.imshow(piv.values, aspect="auto", origin="lower")
                plt.xticks(range(len(piv.columns)), [f"{c:.4g}" for c in piv.columns], rotation=45)
                plt.yticks(range(len(piv.index)), [f"{r:.2f}" for r in piv.index])
                plt.xlabel("w")
                plt.ylabel("cost_threshold")
                plt.title("Floor rate heatmap")
                plt.colorbar(label="proportion_rt_floor_5ms")
                plt.tight_layout()
                plt.savefig(out_dir / "fig_floor_rate_heatmap.png", dpi=150)
                plt.close()

                if not part_df.empty:
                    piv_r2 = part_df[part_df["readout_mode"] == "bayesian_cost"].pivot_table(
                        index="cost_threshold", columns="w", values="a_R2", aggfunc="mean"
                    )
                    if not piv_r2.empty:
                        plt.figure(figsize=(6, 4))
                        plt.imshow(piv_r2.values, aspect="auto", origin="lower")
                        plt.xticks(range(len(piv_r2.columns)), [f"{c:.4g}" for c in piv_r2.columns], rotation=45)
                        plt.yticks(range(len(piv_r2.index)), [f"{r:.2f}" for r in piv_r2.index])
                        plt.xlabel("w")
                        plt.ylabel("cost_threshold")
                        plt.title("R2 heatmap")
                        plt.colorbar(label="mean_a_R2")
                        plt.tight_layout()
                        plt.savefig(out_dir / "fig_R2_heatmap.png", dpi=150)
                        plt.close()

        plt.figure(figsize=(7, 4))
        for mode, g in cond_df.groupby("readout_mode"):
            vals = pd.to_numeric(g["mean_rt_ms"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size > 0:
                plt.hist(vals, bins=min(20, vals.size), alpha=0.5, label=str(mode))
        plt.xlabel("Mean model RT (ms)")
        plt.ylabel("Count")
        plt.title("RT distribution by readout")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "fig_rt_distribution_by_readout.png", dpi=150)
        plt.close()

        by_pos = cond_df.groupby("readout_mode")[["mean_rt_P4", "mean_rt_P5", "mean_rt_P6"]].mean(numeric_only=True)
        if not by_pos.empty:
            plt.figure(figsize=(7, 4))
            x = np.arange(len(by_pos.index))
            plt.plot(x, by_pos["mean_rt_P4"], marker="o", label="P4")
            plt.plot(x, by_pos["mean_rt_P5"], marker="o", label="P5")
            plt.plot(x, by_pos["mean_rt_P6"], marker="o", label="P6")
            plt.xticks(x, by_pos.index, rotation=45)
            plt.ylabel("Mean RT (ms)")
            plt.title("RT by position")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "fig_rt_by_position.png", dpi=150)
            plt.close()

    if not part_df.empty:
        plt.figure(figsize=(5, 5))
        x = pd.to_numeric(part_df["c_R2_surprisal"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(part_df["a_R2"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        plt.scatter(x[m], y[m], alpha=0.7)
        if np.any(m):
            lim = float(max(np.nanmax(x[m]), np.nanmax(y[m]), 0.0))
            plt.plot([0, lim], [0, lim], "k--", lw=1)
        plt.xlabel("R2 prior surprisal")
        plt.ylabel("R2 RNN")
        plt.title("RNN vs prior R2")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_RNN_vs_prior_R2.png", dpi=150)
        plt.close()


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> PredictiveGRU:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        state = ckpt.get("model_state", {}) or {}
        args_dict = ckpt.get("args_dict", ckpt.get("args", {})) or {}
        if not state:
            raise RuntimeError(f"Checkpoint missing cfg and model_state: {ckpt_path}")

        gru_ih_keys = sorted([k for k in state.keys() if k.startswith("gru.weight_ih_l")])
        gru_hh_keys = sorted([k for k in state.keys() if k.startswith("gru.weight_hh_l")])
        if not gru_ih_keys or not gru_hh_keys:
            raise RuntimeError(f"Cannot infer GRU config from checkpoint: {ckpt_path}")

        input_dim = int(state[gru_ih_keys[0]].shape[1])
        hidden_dim = int(state[gru_hh_keys[0]].shape[1])
        num_layers = int(len(gru_ih_keys))
        num_classes = int(state["head.weight"].shape[0]) if "head.weight" in state else int(args_dict.get("num_classes", 3))
        use_stop_head = "stop_head.weight" in state
        use_event_head = "event_head.weight" in state
        use_next_tone_head = "next_tone_head.weight" in state
        use_response_head = "response_head.weight" in state
        add_tone_position_embedding = "tone_pos_embed.weight" in state
        if "ln.weight" in state:
            layer_norm = True
        elif "ln.bias" in state:
            layer_norm = True
        else:
            layer_norm = bool(args_dict.get("layer_norm", False))

        cfg_dict = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "next_tone_num_classes": int(args_dict.get("next_tone_num_classes", 2)),
            "num_layers": num_layers,
            "dropout": float(args_dict.get("dropout", 0.0)),
            "layer_norm": layer_norm,
            "hidden_noise_std": float(args_dict.get("hidden_noise_std", 0.0)),
            "use_stop_head": bool(use_stop_head),
            "use_event_head": bool(use_event_head),
            "use_next_tone_head": bool(use_next_tone_head),
            "use_response_head": bool(use_response_head),
            "add_tone_position_embedding": bool(add_tone_position_embedding),
            "tone_position_embed_dim": int(
                state["tone_pos_embed.weight"].shape[1]
                if "tone_pos_embed.weight" in state
                else args_dict.get("tone_position_embed_dim", 16)
            ),
        }

    cfg = ModelConfig(
        input_dim=int(cfg_dict["input_dim"]),
        hidden_dim=int(cfg_dict["hidden_dim"]),
        num_classes=int(cfg_dict.get("num_classes", 3)),
        next_tone_num_classes=int(cfg_dict.get("next_tone_num_classes", 2)),
        num_layers=int(cfg_dict["num_layers"]),
        dropout=float(cfg_dict.get("dropout", 0.0)),
        layer_norm=bool(cfg_dict.get("layer_norm", False)),
        hidden_noise_std=float(cfg_dict.get("hidden_noise_std", 0.0)),
        use_stop_head=bool(cfg_dict.get("use_stop_head", False)),
        use_event_head=bool(cfg_dict.get("use_event_head", False)),
        use_next_tone_head=bool(cfg_dict.get("use_next_tone_head", False)),
        use_response_head=bool(cfg_dict.get("use_response_head", False)),
        add_tone_position_embedding=bool(cfg_dict.get("add_tone_position_embedding", False)),
        tone_position_embed_dim=int(cfg_dict.get("tone_position_embed_dim", 16)),
    )
    model = PredictiveGRU(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str,
                    default=str(Path("derived") / "rt_readout_bayesian_cost_diagnostics"))

    # which models
    ap.add_argument("--ckpt_glob", type=str, default="",
                    help="Glob pattern, e.g. 'ckpt_sweep_fast/**/best.pt'")
    ap.add_argument("--ckpt_list", type=str, default="",
                    help="Comma-separated checkpoint paths")

    # filtering
    ap.add_argument("--isi_filter", type=int, default=None,
                    help="If set, evaluate only human rows with isi_ms==this (when column exists).")
    ap.add_argument("--isi_ms_default", type=int, default=100,
                    help="Used when human_csv has no isi_ms column.")

    # rendering params (match training)
    ap.add_argument("--tone_ms", type=int, default=50)
    ap.add_argument("--token_ms", type=int, default=10)
    ap.add_argument("--f_min_hz", type=float, default=1300.0)
    ap.add_argument("--f_max_hz", type=float, default=1700.0)
    ap.add_argument("--n_bins", type=int, default=128)
    ap.add_argument("--add_eos", action="store_true")
    ap.add_argument("--add_bos", action="store_true")
    ap.add_argument("--eos_mode", type=str, default="separate", choices=["separate", "mixed"])
    ap.add_argument("--encoding_mode", type=str, default="onehot", choices=["onehot", "gaussian_rf"])
    ap.add_argument("--sigma_rf", type=float, default=1.0)
    ap.add_argument("--rf_normalization", type=str, default="peak", choices=["peak", "sum", "none"])
    ap.add_argument("--sigma_rf_noise", type=float, default=0.0)
    ap.add_argument("--rf_noise_per_token", type=str2bool, default=True)

    # noise (should match how you trained that checkpoint; but for evaluation you can set it explicitly)
    ap.add_argument("--sigma_other_noise", type=float, default=0.0)
    ap.add_argument("--p_other_noise", type=float, default=1.0)
    ap.add_argument("--sigma_silence_noise", type=float, default=0.0)

    # RT criterion
    ap.add_argument("--rt_p_thresh", type=float, default=0.6)
    ap.add_argument("--rt_k_consec", type=int, default=1)
    ap.add_argument("--rt_readout_mode", type=str, default="both",
                    choices=[
                        "simple_threshold",
                        "simple_threshold_pmax",
                        "masked_logits_threshold",
                        "p_correct_argmax",
                        "p_correct_argmax_correct",
                        "p_correct_accumulator",
                        "baseline_corrected_simple_threshold",
                        "baseline_corrected_dynamic_margin",
                        "baseline_corrected_dynamic_pmax",
                        "baseline_corrected_ec",
                        "entropy_threshold",
                        "simple_threshold_logit_margin",
                        "msprt_threshold",
                        "bayesian_cost",
                        "expected_cost_threshold",
                        "bayes_cost_argmin",
                        "advisor_expected_cost_dp",
                        "advisor_bayes_stochastic",
                        "bayesian_online_cost",
                        "both",
                    ])
    ap.add_argument("--cost_w_list", type=str, default="0.00005 0.0001 0.0002 0.000333 0.0005 0.001")
    ap.add_argument("--cost_timeout_ms_list", type=str, default="")
    ap.add_argument("--cost_threshold_list", type=str, default="0.30 0.40 0.50 0.60 0.70 0.80")
    ap.add_argument("--p_correct_threshold_list", type=str, default="0.50 0.60 0.70 0.80 0.90 0.95")
    ap.add_argument("--entropy_threshold_list", type=str, default="1.05 1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10")
    ap.add_argument("--logit_margin_threshold_list", type=str, default="0.5 1.0 1.5 2.0 3.0 4.0")
    ap.add_argument("--msprt_threshold_list", type=str, default="1 2 3 4 5")
    ap.add_argument("--cost_k_consec_list", type=str, default="1 3 5")
    ap.add_argument("--bayes_error_cost", type=float, default=1.0)
    ap.add_argument("--bayes_time_cost_list", type=str, default="0.00005 0.0001 0.0002 0.000333 0.0005 0.001")
    ap.add_argument("--bayes_threshold_start_list", type=str, default="0.90 0.80 0.70 0.60")
    ap.add_argument("--bayes_threshold_min_list", type=str, default="0.20 0.30 0.40")
    ap.add_argument("--bayes_urgency_slope_list", type=str, default="0 0.0005 0.001 0.002")
    ap.add_argument("--bayes_k_consec_list", type=str, default="")
    ap.add_argument("--bayes_evidence_bound_list", type=str, default="2 4 6 8 10 15 20 25 30")
    ap.add_argument("--bayes_leak_list", type=str, default="0 0.01 0.03 0.05 0.10")
    ap.add_argument("--decision_not_before", type=str, default="window_start",
                    choices=["window_start", "deviant_onset", "p4_onset"])
    ap.add_argument("--cost_elapsed_reference", type=str, default="window_start",
                    choices=["window_start", "deviant_onset", "p4_onset"])
    ap.add_argument("--decision_min_elapsed_ms", type=float, default=0.0)
    ap.add_argument("--cost_readout_window", type=str, default="deviant_onset_to_next_tone_onset",
                    choices=[
                        "deviant_onset_to_deviant_offset",
                        "deviant_onset_to_next_tone_onset",
                        "deviant_onset_to_next_tone_offset",
                        "previous_standard_offset_to_next_tone_onset",
                        "deviant_onset_to_trial_end",
                        "deviant_onset_to_second_next_tone_onset",
                        "deviant_onset_to_second_next_tone_offset",
                    ])

    # forward
    ap.add_argument("--chunk_len", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_sequence_mode", type=str, default="trial_independent",
                    choices=["trial_independent", "block10_continuous", "real_boundary_continuous"],
                    help=(
                        "`trial_independent` resets hidden every trial; `block10_continuous` carries hidden within inferred 10-trial chunks; "
                        "`real_boundary_continuous` carries hidden within participant/run/block groups and resets at real boundaries."
                    ))
    ap.add_argument("--eval_context_rho", type=float, default=1.0,
                    help="For continuous modes: carry hidden across trial boundaries as h0(n)=rho*h_end(n-1). 1.0=full carryover, 0.0=reset each trial.")
    ap.add_argument("--drop_first_n_trials_per_participant", type=int, default=0,
                    help="Drop the first N trials for each participant before evaluation, regardless of eval_sequence_mode.")
    ap.add_argument("--no_trial_level_csv", action="store_true",
                    help="Do not write per-checkpoint trial_level.csv or global trial_level_rt_readout.csv.")
    ap.add_argument("--no_figures", action="store_true",
                    help="Do not generate summary figures.")

    args = ap.parse_args()

    def _same_or_both_nan(a, b) -> bool:
        try:
            if pd.isna(a) and pd.isna(b):
                return True
        except Exception:
            pass
        return a == b

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect ckpts
    ckpts: List[Path] = []
    if args.ckpt_list.strip():
        ckpts += [Path(p.strip()) for p in args.ckpt_list.split(",") if p.strip()]
    if args.ckpt_glob.strip():
        ckpts += sorted(Path(".").glob(args.ckpt_glob))
    ckpts = [p for p in ckpts if p.exists()]

    if not ckpts:
        raise RuntimeError("No checkpoints found. Use --ckpt_glob or --ckpt_list.")

    # load human trials
    df = pd.read_csv(args.human_csv)
    col = detect_columns(df)

    # optional isi filtering
    if args.isi_filter is not None and "isi_ms" in col:
        df[col["isi_ms"]] = pd.to_numeric(df[col["isi_ms"]], errors="coerce")
        df = df[df[col["isi_ms"]] == int(args.isi_filter)].copy()

    # clean
    df[col["position"]] = pd.to_numeric(df[col["position"]], errors="coerce")
    df[col["rt_ms"]] = pd.to_numeric(df[col["rt_ms"]], errors="coerce")
    df[col["f_std"]] = pd.to_numeric(df[col["f_std"]], errors="coerce")
    df[col["f_dev"]] = pd.to_numeric(df[col["f_dev"]], errors="coerce")

    df = df[df[col["position"]].isin([4, 5, 6])].copy()
    df = df[df[col["f_std"]].notna() & df[col["f_dev"]].notna()].copy()

    if df.empty:
        raise RuntimeError("No valid human trials after filtering/cleaning.")

    isi_ms_eval = int(args.isi_ms_default)
    if "isi_ms" in col:
        # if multiple ISIs remain, this script currently expects ONE ISI for rendering
        unique_isi = sorted(df[col["isi_ms"]].dropna().astype(int).unique().tolist())
        if len(unique_isi) == 1:
            isi_ms_eval = int(unique_isi[0])
        elif len(unique_isi) == 0:
            isi_ms_eval = int(args.isi_ms_default)
        else:
            raise RuntimeError(
                f"Multiple isi_ms values remain in human CSV: {unique_isi}. "
                f"Filter with --isi_filter or split the file."
            )

    df, leading_drop_audit = drop_leading_trials_per_participant(
        df=df,
        col=col,
        n_drop=int(args.drop_first_n_trials_per_participant),
    )
    if df.empty:
        raise RuntimeError("No valid human trials remain after leading-trial drop.")

    device = torch.device(args.device)
    sequence_audit: Dict[str, Any] = {
        "eval_sequence_mode": str(args.eval_sequence_mode),
        "eval_context_rho": float(args.eval_context_rho),
        "drop_first_n_trials_per_participant": int(args.drop_first_n_trials_per_participant),
    }
    sequence_audit.update(leading_drop_audit)

    if str(args.eval_sequence_mode) == "block10_continuous":
        df_eval, X_blocks, y_pos, trial_T_tokens, tone_T, isi_T, block_audit = build_block10_continuous_inputs(
            df=df,
            col=col,
            tone_ms=args.tone_ms,
            token_ms=args.token_ms,
            isi_ms=isi_ms_eval,
            f_min_hz=args.f_min_hz,
            f_max_hz=args.f_max_hz,
            n_bins=args.n_bins,
            add_eos=bool(args.add_eos),
            add_bos=bool(args.add_bos),
            eos_mode=str(args.eos_mode),
            sigma_other_noise=args.sigma_other_noise,
            p_other_noise=args.p_other_noise,
            sigma_silence_noise=args.sigma_silence_noise,
            encoding_mode=str(args.encoding_mode),
            sigma_rf=float(args.sigma_rf),
            rf_normalization=str(args.rf_normalization),
            sigma_rf_noise=float(args.sigma_rf_noise),
            rf_noise_per_token=bool(args.rf_noise_per_token),
            seed=args.seed,
        )
        sequence_audit.update(block_audit)
        X_blocks = X_blocks.to(device)
        N = int(len(df_eval))
    elif str(args.eval_sequence_mode) == "real_boundary_continuous":
        df_eval = df.copy().reset_index(drop=True)
        X, y_pos, trial_T_tokens, tone_T, isi_T = render_trials_onehot(
            df=df_eval,
            col=col,
            tone_ms=args.tone_ms,
            token_ms=args.token_ms,
            isi_ms=isi_ms_eval,
            f_min_hz=args.f_min_hz,
            f_max_hz=args.f_max_hz,
            n_bins=args.n_bins,
            add_eos=bool(args.add_eos),
            add_bos=bool(args.add_bos),
            eos_mode=str(args.eos_mode),
            sigma_other_noise=args.sigma_other_noise,
            p_other_noise=args.p_other_noise,
            sigma_silence_noise=args.sigma_silence_noise,
            encoding_mode=str(args.encoding_mode),
            sigma_rf=float(args.sigma_rf),
            rf_normalization=str(args.rf_normalization),
            sigma_rf_noise=float(args.sigma_rf_noise),
            rf_noise_per_token=bool(args.rf_noise_per_token),
            seed=args.seed,
        )
        real_group_ids, real_group_cols = real_boundary_group_ids(df_eval, col)
        sequence_audit.update({
            "real_boundary_group_cols": "+".join(real_group_cols),
            "n_real_boundary_groups": int(len(set(real_group_ids))),
        })
        X = X.to(device)
        N = X.shape[0]
    else:
        df_eval = df.copy().reset_index(drop=True)
        X, y_pos, trial_T_tokens, tone_T, isi_T = render_trials_onehot(
            df=df_eval,
            col=col,
            tone_ms=args.tone_ms,
            token_ms=args.token_ms,
            isi_ms=isi_ms_eval,
            f_min_hz=args.f_min_hz,
            f_max_hz=args.f_max_hz,
            n_bins=args.n_bins,
            add_eos=bool(args.add_eos),
            add_bos=bool(args.add_bos),
            eos_mode=str(args.eos_mode),
            sigma_other_noise=args.sigma_other_noise,
            p_other_noise=args.p_other_noise,
            sigma_silence_noise=args.sigma_silence_noise,
            encoding_mode=str(args.encoding_mode),
            sigma_rf=float(args.sigma_rf),
            rf_normalization=str(args.rf_normalization),
            sigma_rf_noise=float(args.sigma_rf_noise),
            rf_noise_per_token=bool(args.rf_noise_per_token),
            seed=args.seed,
        )
        X = X.to(device)
        N = X.shape[0]

    # targets
    y_cls = labels_to_class_index(y_pos)  # (N,)
    dev_end = deviant_end_token_in_trial(y_pos, tone_T=tone_T, isi_T=isi_T)  # (N,)

    summary_rows: List[Dict[str, Any]] = []
    all_trial_rows: List[Dict[str, Any]] = []
    all_condition_rows: List[Dict[str, Any]] = []

    for ckpt_path in ckpts:
        model_name = ckpt_path.parent.name if ckpt_path.name in ("best.pt", "last.pt") else ckpt_path.stem
        model_out = out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        model = load_checkpoint_model(ckpt_path, device=device)

        probs_end_all = []
        pred_all = []
        logits_all_list = []

        if str(args.eval_sequence_mode) == "block10_continuous":
            B = int(X_blocks.shape[0])
            trial_T = int(trial_T_tokens)
            for s in range(0, B, int(args.batch_size)):
                e = min(s + int(args.batch_size), B)
                xb = X_blocks[s:e]
                logits_cat = forward_block10_continuous_with_rho(
                    model,
                    xb,
                    trial_T_tokens=trial_T,
                    chunk_len=int(args.chunk_len),
                    rho=float(args.eval_context_rho),
                )  # (B,10*T,3)
                logits_trial = logits_cat.view(e - s, 10, trial_T, 3)
                logits_end = logits_trial[:, :, -1, :].reshape(-1, 3)
                logits_all = logits_trial.reshape(-1, trial_T, 3)
                logits_all_list.append(logits_all.detach().cpu())
                pb = torch.softmax(logits_end, dim=-1).detach().cpu().numpy()
                probs_end_all.append(pb)
                pred_all.append(pb.argmax(axis=1))
        elif str(args.eval_sequence_mode) == "real_boundary_continuous":
            logits_all = forward_real_boundary_continuous_with_rho(
                model,
                X,
                group_ids=real_group_ids,
                chunk_len=int(args.chunk_len),
                rho=float(args.eval_context_rho),
            )
            logits_all_list.append(logits_all.detach().cpu())
            logits_end = trial_end_logits_from_full(
                logits_all,
                end_offset_from_trial_end=(1 if bool(args.add_bos) else 0),
            )
            pb = torch.softmax(logits_end, dim=-1).detach().cpu().numpy()
            probs_end_all.append(pb)
            pred_all.append(pb.argmax(axis=1))
        else:
            for s in range(0, N, int(args.batch_size)):
                e = min(s + int(args.batch_size), N)
                xb = X[s:e]
                logits_all = forward_full_logits(model, xb, chunk_len=int(args.chunk_len))  # (B,T,3)
                logits_all_list.append(logits_all.detach().cpu())

                logits_end = trial_end_logits_from_full(
                    logits_all,
                    end_offset_from_trial_end=(1 if bool(args.add_bos) else 0),
                )  # (B,3)
                pb = torch.softmax(logits_end, dim=-1).detach().cpu().numpy()
                probs_end_all.append(pb)
                pred_all.append(pb.argmax(axis=1))

        probs_end = np.concatenate(probs_end_all, axis=0)  # (N,3)
        pred = np.concatenate(pred_all, axis=0)           # (N,)
        logits_all_cpu = torch.cat(logits_all_list, dim=0)  # (N,T,3)

        y_true = y_cls.detach().cpu().numpy().astype(int)
        acc = float((pred == y_true).mean())
        f1 = safe_f1_macro(y_true, pred)
        auc = safe_auc_ovr(y_true, probs_end, n_classes=3)

        readout_start, readout_end = split_readout_window_name(str(args.cost_readout_window))
        prepared = prepare_rt_readout(
            logits_trial=logits_all_cpu,
            y_cls=y_cls.detach().cpu(),
            y_pos_456=y_pos.detach().cpu(),
            tone_T=int(tone_T),
            isi_T=int(isi_T),
            token_ms=int(args.token_ms),
            readout_start=str(readout_start),
            readout_end=str(readout_end),
        )
        readout_trial_rows, readout_condition_rows = run_rt_readout_sweeps(
            prepared=prepared,
            rt_readout_mode=str(args.rt_readout_mode),
            p_threshold_list=parse_list_of_floats(args.entropy_threshold_list) if str(args.rt_readout_mode) == "entropy_threshold" else parse_list_of_floats(args.p_correct_threshold_list),
            logit_margin_threshold_list=parse_list_of_floats(args.logit_margin_threshold_list),
            msprt_threshold_list=parse_list_of_floats(args.msprt_threshold_list),
            cost_w_list=parse_list_of_floats(args.cost_w_list),
            cost_timeout_ms_list=parse_list_of_floats(args.cost_timeout_ms_list),
            cost_threshold_list=parse_list_of_floats(args.cost_threshold_list),
            k_consec_list=parse_list_of_ints(args.cost_k_consec_list),
            bayes_error_cost=float(args.bayes_error_cost),
            bayes_time_cost_list=parse_list_of_floats(args.bayes_time_cost_list),
            bayes_threshold_start_list=parse_list_of_floats(args.bayes_threshold_start_list),
            bayes_threshold_min_list=parse_list_of_floats(args.bayes_threshold_min_list),
            bayes_urgency_slope_list=parse_list_of_floats(args.bayes_urgency_slope_list),
            bayes_k_consec_list=parse_list_of_ints(args.bayes_k_consec_list),
            bayes_evidence_bound_list=parse_list_of_floats(args.bayes_evidence_bound_list),
            bayes_leak_list=parse_list_of_floats(args.bayes_leak_list),
            decision_not_before=str(args.decision_not_before),
            cost_elapsed_reference=str(args.cost_elapsed_reference),
            decision_min_elapsed_ms=float(args.decision_min_elapsed_ms),
        )

        human_rt = df_eval[col["rt_ms"]].to_numpy(dtype=float)
        participant_col = col.get("subject_id", "")
        participant_values = df_eval[participant_col].astype(str).to_numpy() if participant_col else np.array(["unknown"] * N, dtype=object)
        pos_vals = y_pos.detach().cpu().numpy().astype(int)
        prior_prob = prior_prob_from_position(pos_vals)
        prior_surprisal = prior_surprisal_from_position(pos_vals)
        ckpt_epoch = int(torch.load(str(ckpt_path), map_location="cpu", weights_only=False).get("epoch_global", -1))

        # save trial-level (legacy single-file view uses default simple rule when available)
        out_df = df_eval.copy()
        out_df["model_name"] = model_name
        out_df["model_ckpt"] = str(ckpt_path)
        out_df["y_cls"] = y_true
        out_df["pred_cls"] = pred
        out_df["p0"] = probs_end[:, 0]
        out_df["p1"] = probs_end[:, 1]
        out_df["p2"] = probs_end[:, 2]
        default_rt = next((r for r in readout_trial_rows if r["readout_mode"] == "simple_threshold" and float(r["p_threshold"]) == float(args.rt_p_thresh) and int(r["k_consec"]) == int(args.rt_k_consec)), None)
        if default_rt is not None:
            pass
        trial_csv: Optional[Path] = None
        if not bool(args.no_trial_level_csv):
            trial_csv = model_out / "trial_level.csv"
            out_df.to_csv(trial_csv, index=False)

        for row in readout_trial_rows:
            i = int(row["trial_index"])
            row.update({
                "participant": str(participant_values[i]),
                "run": str(model_name),
                "trial_index_global": int(i),
                "trial_type": f"std{int(round(float(df_eval.iloc[i][col['f_std']])))}_dev{int(round(float(df_eval.iloc[i][col['f_dev']]))) }",
                "human_rt_ms": float(human_rt[i]) if np.isfinite(human_rt[i]) else float("nan"),
                "prior_prob": float(prior_prob[i]),
                "prior_surprisal": float(prior_surprisal[i]),
                "checkpoint_path": str(ckpt_path.resolve()),
                "epoch": int(ckpt_epoch),
                "trial_index": int(i),
            })
        for row in readout_condition_rows:
            row.update({
                "checkpoint_path": str(ckpt_path.resolve()),
                "epoch": int(ckpt_epoch),
                "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
            })
        all_trial_rows.extend(readout_trial_rows)
        all_condition_rows.extend(readout_condition_rows)

        part_rows = build_participant_level_rt_eval(pd.DataFrame(readout_trial_rows))
        group_rows = build_group_level_rt_eval(part_rows)

        # Summarize the best readout condition per checkpoint using delta_R2 first.
        best_part_mean = float("nan")
        best_part_c_r2 = float("nan")
        best_part_delta_r2 = float("nan")
        best_part_found_rate = float("nan")
        best_part_mean_rt_ms = float("nan")
        best_part_floor5 = float("nan")
        best_cond_accuracy_at_rt = float("nan")
        decision_found_rate = float("nan")
        decision_acc_at_rt = float("nan")
        decision_acc_at_valid_rt = float("nan")
        best_cond_readout_mode = None
        best_cond_w = float("nan")
        best_cond_cost_threshold = float("nan")
        best_cond_p_threshold = float("nan")
        best_cond_k_consec = float("nan")
        corr = {"pearson_r": float("nan"), "n": float("nan")}
        if part_rows:
            def _score(row):
                delta = row.get("standalone_delta_R2", float("nan"))
                a_r2 = row.get("a_R2", float("nan"))
                return (
                    -1e9 if not np.isfinite(delta) else float(delta),
                    -1e9 if not np.isfinite(a_r2) else float(a_r2),
                )

            best = max(part_rows, key=_score)
            corr = {"pearson_r": float(best.get("a_r", float("nan"))), "n": float(best.get("n_trials", float("nan")))}
            best_part_mean = float(best.get("a_R2", float("nan")))
            best_part_c_r2 = float(best.get("c_R2_surprisal", float("nan")))
            best_part_delta_r2 = float(best.get("standalone_delta_R2", float("nan")))
            best_part_found_rate = float(best.get("found_rate", float("nan")))
            best_part_mean_rt_ms = float(best.get("mean_rt_ms", float("nan")))
            best_part_floor5 = float(best.get("floor_rate_5ms", float("nan")))
            best_cond_readout_mode = best.get("readout_mode")
            best_cond_w = float(best.get("w", float("nan")))
            best_cond_cost_threshold = float(best.get("cost_threshold", float("nan")))
            best_cond_p_threshold = float(best.get("p_threshold", float("nan")))
            best_cond_k_consec = float(best.get("k_consec", float("nan")))

            for cond in readout_condition_rows:
                if (
                    _same_or_both_nan(cond.get("readout_mode"), best.get("readout_mode"))
                    and _same_or_both_nan(cond.get("w"), best.get("w"))
                    and _same_or_both_nan(cond.get("timeout_ms"), best.get("timeout_ms"))
                    and _same_or_both_nan(cond.get("cost_threshold"), best.get("cost_threshold"))
                    and _same_or_both_nan(cond.get("p_threshold"), best.get("p_threshold"))
                    and _same_or_both_nan(cond.get("k_consec"), best.get("k_consec"))
                ):
                    best_cond_accuracy_at_rt = float(cond.get("accuracy_at_rt", float("nan")))
                    break

            matched_trials = [
                row for row in readout_trial_rows
                if (
                    _same_or_both_nan(row.get("readout_mode"), best.get("readout_mode"))
                    and _same_or_both_nan(row.get("w"), best.get("w"))
                    and _same_or_both_nan(row.get("timeout_ms"), best.get("timeout_ms"))
                    and _same_or_both_nan(row.get("cost_threshold"), best.get("cost_threshold"))
                    and _same_or_both_nan(row.get("p_threshold"), best.get("p_threshold"))
                    and _same_or_both_nan(row.get("k_consec"), best.get("k_consec"))
                )
            ]
            if matched_trials:
                found_arr = np.asarray([bool(row.get("found_flag", False)) for row in matched_trials], dtype=bool)
                correct_arr = np.asarray([bool(row.get("correct_at_rt", False)) for row in matched_trials], dtype=bool)
                decision_found_rate = float(np.mean(found_arr.astype(float)))
                decision_acc_at_rt = float(np.mean(correct_arr.astype(float)))
                decision_acc_at_valid_rt = float(np.mean(correct_arr[found_arr].astype(float))) if np.any(found_arr) else float("nan")

        metrics = {
            "model_name": model_name,
            "ckpt": str(ckpt_path),
            "N_trials": int(N),
            "isi_ms_eval": int(isi_ms_eval),
            "eval_sequence_mode": str(args.eval_sequence_mode),
            "tone_ms": int(args.tone_ms),
            "token_ms": int(args.token_ms),
            "n_bins": int(args.n_bins),
            "noise_eval": {
                "sigma_other_noise": float(args.sigma_other_noise),
                "p_other_noise": float(args.p_other_noise),
                "sigma_silence_noise": float(args.sigma_silence_noise),
            },
            "rt_rule": {
                "rt_readout_mode": str(args.rt_readout_mode),
                "p_threshold_list": parse_list_of_floats(args.p_correct_threshold_list),
                "entropy_threshold_list": parse_list_of_floats(args.entropy_threshold_list),
                "logit_margin_threshold_list": parse_list_of_floats(args.logit_margin_threshold_list),
                "msprt_threshold_list": parse_list_of_floats(args.msprt_threshold_list),
                "cost_w_list": parse_list_of_floats(args.cost_w_list),
                "cost_timeout_ms_list": parse_list_of_floats(args.cost_timeout_ms_list),
                "cost_threshold_list": parse_list_of_floats(args.cost_threshold_list),
                "k_consec_list": parse_list_of_ints(args.cost_k_consec_list),
                "bayes_error_cost": float(args.bayes_error_cost),
                "bayes_time_cost_list": parse_list_of_floats(args.bayes_time_cost_list),
                "bayes_threshold_start_list": parse_list_of_floats(args.bayes_threshold_start_list),
                "bayes_threshold_min_list": parse_list_of_floats(args.bayes_threshold_min_list),
                "bayes_urgency_slope_list": parse_list_of_floats(args.bayes_urgency_slope_list),
                "bayes_k_consec_list": parse_list_of_ints(args.bayes_k_consec_list),
                "decision_not_before": str(args.decision_not_before),
                "cost_elapsed_reference": str(args.cost_elapsed_reference),
                "decision_min_elapsed_ms": float(args.decision_min_elapsed_ms),
                "cost_readout_window": str(args.cost_readout_window),
            },
            "metrics": {
                "final_acc_all": acc,
                "f1_macro": float(f1),
                "auc_ovr": float(auc),
                "decision_found_rate": decision_found_rate,
                "decision_acc_at_rt": decision_acc_at_rt,
                "decision_acc_at_valid_rt": decision_acc_at_valid_rt,
            },
            "window_metrics": {
                **prepared.overall_metrics,
                "trajectory_rel_ms": prepared.trajectory_rel_ms,
                "trajectory_by_position": prepared.trajectory_by_position,
            },
            "window_metrics_by_position": prepared.by_position_metrics,
            "sequence_audit": sequence_audit,
            "readout_condition_summary_n": int(len(readout_condition_rows)),
            "participant_eval_n": int(len(part_rows)),
            "group_eval_n": int(len(group_rows)),
            "best_participant_mean_a_R2": best_part_mean,
            "best_participant_mean_c_R2": best_part_c_r2,
            "best_participant_mean_delta_R2": best_part_delta_r2,
            "decision_found_rate": decision_found_rate,
            "decision_acc_at_rt": decision_acc_at_rt,
            "decision_acc_at_valid_rt": decision_acc_at_valid_rt,
            "rt_corr_with_human": corr,
            "outputs": {"trial_level_csv": str(trial_csv) if trial_csv is not None else None},
        }

        (model_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        summary_rows.append({
            "model_name": model_name,
            "ckpt": str(ckpt_path),
            "N_trials": int(N),
            "isi_ms_eval": int(isi_ms_eval),
            "eval_sequence_mode": str(args.eval_sequence_mode),
            "final_acc_all": acc,
            "f1_macro": float(f1),
            "auc_ovr": float(auc),
            "readout_condition_summary_n": int(len(readout_condition_rows)),
            "best_participant_mean_a_R2": best_part_mean,
            "best_participant_mean_c_R2": best_part_c_r2,
            "best_participant_mean_delta_R2": best_part_delta_r2,
            "best_condition_found_rate": best_part_found_rate,
            "decision_found_rate": decision_found_rate,
            "decision_acc_at_rt": decision_acc_at_rt,
            "decision_acc_at_valid_rt": decision_acc_at_valid_rt,
            "best_condition_mean_rt_ms": best_part_mean_rt_ms,
            "best_condition_floor5_rate": best_part_floor5,
            "best_condition_acc_at_valid_rt": best_cond_accuracy_at_rt,
            "best_condition_readout_mode": best_cond_readout_mode,
            "best_condition_w": best_cond_w,
            "best_condition_cost_threshold": best_cond_cost_threshold,
            "best_condition_p_threshold": best_cond_p_threshold,
            "best_condition_k_consec": best_cond_k_consec,
            "rt_corr_n": int(corr.get("n", float("nan"))) if not math.isnan(corr.get("n", float("nan"))) else np.nan,
            "rt_corr_pearson_r": corr.get("pearson_r", float("nan")),
            "rt_corr_pearson_p": corr.get("pearson_p", float("nan")),
            "rt_corr_spearman_r": corr.get("spearman_r", float("nan")),
            "rt_corr_spearman_p": corr.get("spearman_p", float("nan")),
            "trial_level_csv": str(trial_csv) if trial_csv is not None else None,
        })

        print(
            f"[done] {model_name}: final_acc={acc:.3f} f1={float(f1):.3f} auc={float(auc):.3f} "
            f"best_a_R2={best_part_mean:.3f} best_c_R2={best_part_c_r2:.3f} "
            f"best_delta_R2={best_part_delta_r2:.3f} decision_acc={decision_acc_at_rt:.3f} "
            f"decision_acc_valid={decision_acc_at_valid_rt:.3f} found={decision_found_rate:.3f} "
            f"meanRT={best_part_mean_rt_ms:.1f} "
            f"pearson_r={corr.get('pearson_r', float('nan')):.3f}"
        )

    trial_level_path = out_dir / "trial_level_rt_readout.csv"
    cond_summary_path = out_dir / "readout_condition_summary.csv"
    participant_eval_path = out_dir / "participant_level_rt_eval.csv"
    group_eval_path = out_dir / "group_level_rt_eval.csv"
    eval_log_path = out_dir / "evaluation_log.txt"

    if not bool(args.no_trial_level_csv):
        write_csv_rows(trial_level_path, all_trial_rows)
    write_csv_rows(cond_summary_path, all_condition_rows)
    participant_rows = build_participant_level_rt_eval(pd.DataFrame(all_trial_rows)) if all_trial_rows else []
    group_rows = build_group_level_rt_eval(participant_rows)
    write_csv_rows(participant_eval_path, participant_rows)
    write_csv_rows(group_eval_path, group_rows)
    first_metrics_path = (out_dir / summary_rows[0]["model_name"] / "metrics.json") if summary_rows else None
    if not bool(args.no_figures):
        make_readout_figures(
            out_dir=out_dir,
            summary_rows=summary_rows,
            condition_rows=all_condition_rows,
            participant_rows=participant_rows,
            first_metrics_path=first_metrics_path,
        )

    log_lines = [
        f"checkpoints_evaluated={len(ckpts)}",
        f"rt_readout_mode={args.rt_readout_mode}",
        f"eval_sequence_mode={args.eval_sequence_mode}",
        f"cost_w_list={parse_list_of_floats(args.cost_w_list)}",
        f"cost_timeout_ms_list={parse_list_of_floats(args.cost_timeout_ms_list)}",
        f"cost_threshold_list={parse_list_of_floats(args.cost_threshold_list)}",
        f"p_threshold_list={parse_list_of_floats(args.p_correct_threshold_list)}",
        f"entropy_threshold_list={parse_list_of_floats(args.entropy_threshold_list)}",
        f"logit_margin_threshold_list={parse_list_of_floats(args.logit_margin_threshold_list)}",
        f"msprt_threshold_list={parse_list_of_floats(args.msprt_threshold_list)}",
        f"k_consec_list={parse_list_of_ints(args.cost_k_consec_list)}",
        f"bayes_error_cost={float(args.bayes_error_cost)}",
        f"bayes_time_cost_list={parse_list_of_floats(args.bayes_time_cost_list)}",
        f"bayes_threshold_start_list={parse_list_of_floats(args.bayes_threshold_start_list)}",
        f"bayes_threshold_min_list={parse_list_of_floats(args.bayes_threshold_min_list)}",
        f"bayes_urgency_slope_list={parse_list_of_floats(args.bayes_urgency_slope_list)}",
        f"bayes_k_consec_list={parse_list_of_ints(args.bayes_k_consec_list)}",
        f"decision_not_before={args.decision_not_before}",
        f"cost_elapsed_reference={args.cost_elapsed_reference}",
        f"decision_min_elapsed_ms={float(args.decision_min_elapsed_ms)}",
        "human_rt_trim_rule=-1500ms <= RT <= 1500ms",
        f"cost_readout_window={args.cost_readout_window}",
        "current training meanRT is treated as a diagnostic only; model RT comes from post-hoc readout sweep.",
    ]
    for k, v in sequence_audit.items():
        log_lines.append(f"sequence_audit_{k}={v}")
    if participant_rows:
        best_any = max(participant_rows, key=lambda r: (-1e9 if not np.isfinite(r.get('a_R2', float('nan'))) else r['a_R2']))
        log_lines.append(
            f"best_readout_by_mean_a_R2=mode:{best_any['readout_mode']} w:{best_any.get('w')} timeout:{best_any.get('timeout_ms')} "
            f"cost_thr:{best_any.get('cost_threshold')} p_thr:{best_any.get('p_threshold')} k:{best_any.get('k_consec')} a_R2:{best_any.get('a_R2')}"
        )
        non_floor = [r for r in participant_rows if np.isfinite(r.get("floor_rate_5ms", float("nan"))) and r.get("floor_rate_5ms", 1.0) < 0.2]
        if non_floor:
            best_nf = max(non_floor, key=lambda r: (-1e9 if not np.isfinite(r.get('a_R2', float('nan'))) else r['a_R2']))
            log_lines.append(
                f"best_non_floor_readout=mode:{best_nf['readout_mode']} w:{best_nf.get('w')} timeout:{best_nf.get('timeout_ms')} "
                f"cost_thr:{best_nf.get('cost_threshold')} p_thr:{best_nf.get('p_threshold')} k:{best_nf.get('k_consec')} "
                f"a_R2:{best_nf.get('a_R2')} floor5:{best_nf.get('floor_rate_5ms')}"
            )
        log_lines.append(f"any_negative_a_r={any(bool(r.get('warning_negative_a_r', False)) for r in participant_rows)}")
    if all_condition_rows:
        log_lines.append(f"any_rt_ordering_P4_gt_P5_gt_P6={any(bool(r.get('rt_ordering_P4_gt_P5_gt_P6', False)) for r in all_condition_rows)}")
        log_lines.append(f"any_p_correct_ordering_P6_gt_P5_gt_P4={any(bool(r.get('p_correct_ordering_P6_gt_P5_gt_P4', False)) for r in all_condition_rows)}")
    eval_log_path.write_text("\n".join(log_lines), encoding="utf-8")

    # save global summary
    summary_df = pd.DataFrame(summary_rows).sort_values(by="rt_corr_pearson_r", ascending=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"\n[all done] wrote: {out_dir/'summary.csv'} and per-model folders.")
    if not bool(args.no_trial_level_csv):
        print(f"  - {trial_level_path}")
    print(f"  - {cond_summary_path}")
    print(f"  - {participant_eval_path}")
    print(f"  - {group_eval_path}")


if __name__ == "__main__":
    main()
    
