#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_online.py

from __future__ import annotations

import argparse
import ast
import copy
import csv
import itertools
import json
import math
import shutil
import time
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model import PredictiveGRU, ModelConfig
from stimulus_encoding import (
    StimulusEncodingConfig,
    build_tone_encoding_vector,
    freq_to_bin_erb as shared_freq_to_bin_erb,
    make_erb_edges as shared_make_erb_edges,
)
from rt_readout import prepare_rt_readout, run_rt_readout_sweeps, aggregate_readout_trial_rows
from deviant_detection.common import (
    _HAVE_SKLEARN,
    _make_amp_autocast,
    _resolve_amp_dtype,
    compute_multiclass_metrics_from_probs,
    make_run_name,
    safe_auc_ovr,
    safe_f1_macro,
    set_all_seeds,
    str2bool,
    write_csv_row,
    write_jsonl,
)
from deviant_detection.data import (
    OnlineRenderDataset,
    PreRenderedBlockDataset,
    TrialwiseRenderDataset,
    compute_trial_gap_hz,
    make_gap_weight_tensor,
    unpack_batch_with_optional_gap,
)
from deviant_detection.stimuli import generate_stimuli_blocks, parse_exclude_pairs
from deviant_detection.token_geometry import (
    deviant_end_token_in_trial,
    deviant_onset_token_in_trial,
    first_possible_deviant_onset_token_in_trial,
    infer_end_indices_from_T,
    labels_to_class_index,
    make_strict_online_p4_mask,
    make_strict_p4_causal_target_probs,
    make_strict_p4_post_offset_mask,
    make_tone_event_targets_tokens,
    next_standard_onset_token_in_trial,
    next_tone_offset_token_in_trial,
    next_tone_onset_token_in_trial,
    previous_standard_offset_token_in_trial,
    previous_standard_onset_token_in_trial,
    second_next_tone_offset_token_in_trial,
    second_next_tone_onset_token_in_trial,
    supervision_num_classes,
    tone_offset_token_in_trial,
    tone_onset_token_in_trial,
    trial_end_token_in_trial,
)


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
    return shared_make_erb_edges(f_min_hz=f_min_hz, f_max_hz=f_max_hz, n_bins=n_bins)


def freq_to_bin_erb(
    f_hz: float,
    edges_erb: np.ndarray,
) -> int:
    return shared_freq_to_bin_erb(f_hz=f_hz, edges_erb=edges_erb)


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


def _normalize_token_loss_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m in ("uniform", "exp", "old"):
        return "old"
    if m == "windowed_correct_ce":
        return "windowed_correct_ce"
    if m in ("strict_p4_causal_ce", "causal_hazard_ce"):
        return "strict_p4_causal_ce"
    if m in ("event_deviance_ce", "tone_event_ce", "deviance_event_ce"):
        return "event_deviance_ce"
    raise ValueError(f"Unknown token_loss_mode: {mode}")


def compute_correct_ce_window_bounds_in_trial(
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    correct_ce_window: str,
    n_tones: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      start_tok: inclusive token index
      end_tok_exclusive: exclusive token index
      clipped: whether end had to be clipped to trial end
    """
    y_pos_456 = y_pos_456.long()
    if not torch.all((y_pos_456 >= 1) & (y_pos_456 <= int(n_tones))):
        raise ValueError(f"Tone positions must be in [1,{n_tones}], got min={int(y_pos_456.min())} max={int(y_pos_456.max())}")

    if correct_ce_window == "previous_standard_onset_to_next_tone_onset":
        start_tok = previous_standard_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    elif correct_ce_window == "first_possible_deviant_onset_to_next_tone_onset":
        start_tok = first_possible_deviant_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    elif correct_ce_window == "first_possible_deviant_onset_to_trial_end":
        start_tok = first_possible_deviant_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    else:
        start_tok = deviant_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    trial_end_exclusive = trial_end_token_in_trial(y_pos_456, trial_T_tokens=trial_T_tokens) + 1

    if correct_ce_window == "deviant_onset_to_deviant_offset":
        raw_end_exclusive = deviant_end_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T) + 1
    elif correct_ce_window == "deviant_onset_to_next_tone_onset":
        raw_end_exclusive = next_tone_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    elif correct_ce_window == "deviant_onset_to_trial_end":
        raw_end_exclusive = trial_end_exclusive
    elif correct_ce_window == "previous_standard_onset_to_next_tone_onset":
        raw_end_exclusive = next_tone_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    elif correct_ce_window == "first_possible_deviant_onset_to_next_tone_onset":
        raw_end_exclusive = next_tone_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    elif correct_ce_window == "first_possible_deviant_onset_to_trial_end":
        raw_end_exclusive = trial_end_exclusive
    elif correct_ce_window == "deviant_onset_to_next_tone_offset":
        raw_end_exclusive = next_tone_offset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T) + 1
    elif correct_ce_window == "deviant_onset_to_second_next_tone_onset":
        raw_end_exclusive = second_next_tone_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    elif correct_ce_window == "deviant_onset_to_second_next_tone_offset":
        raw_end_exclusive = second_next_tone_offset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T) + 1
    else:
        raise ValueError(f"Unknown correct_ce_window: {correct_ce_window}")

    end_tok_exclusive = torch.minimum(raw_end_exclusive, trial_end_exclusive)
    clipped = raw_end_exclusive > trial_end_exclusive
    if not torch.all(end_tok_exclusive >= start_tok):
        raise RuntimeError("Window end must be >= window start. Check token window definitions.")
    return start_tok, end_tok_exclusive, clipped


def make_correct_ce_window_mask_tokens(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    correct_ce_window: str,
) -> torch.Tensor:
    trial_id = (abs_t // trial_T_tokens).long()
    within = (abs_t % trial_T_tokens).long()
    start_bt, end_excl_bt, _clipped_bt = compute_correct_ce_window_bounds_in_trial(
        y_pos_456=y_pos_456,
        trial_T_tokens=trial_T_tokens,
        tone_T=tone_T,
        isi_T=isi_T,
        correct_ce_window=correct_ce_window,
    )
    start = start_bt[:, trial_id]
    end_excl = end_excl_bt[:, trial_id]
    return (within >= start) & (within < end_excl)


def _legacy_readout_window_to_bounds(readout_window: str) -> Tuple[str, str]:
    w = str(readout_window).strip().lower()
    if w == "deviant_onset_to_deviant_offset":
        return "deviant_onset", "deviant_offset"
    if w == "deviant_onset_to_next_tone_onset":
        return "deviant_onset", "next_tone_onset"
    if w == "deviant_onset_to_next_tone_offset":
        return "deviant_onset", "next_tone_offset"
    if w == "deviant_onset_to_second_next_tone_onset":
        return "deviant_onset", "trial_end"
    if w == "deviant_onset_to_second_next_tone_offset":
        return "deviant_onset", "trial_end"
    if w == "previous_standard_onset_to_next_tone_onset":
        return "previous_standard_onset", "next_tone_onset"
    if w == "previous_standard_offset_to_next_tone_onset":
        return "previous_standard_offset", "next_tone_onset"
    if w == "first_possible_deviant_onset_to_next_tone_onset":
        return "first_possible_deviant_onset", "next_tone_onset"
    if w == "first_possible_deviant_onset_to_trial_end":
        return "first_possible_deviant_onset", "trial_end"
    raise ValueError(f"Unknown readout_window: {readout_window}")


def get_correct_ce_window_debug_summary(
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    correct_ce_window: str,
) -> Dict[str, int]:
    y_one = y_pos_456.reshape(-1)[0:1].long()
    prev_on = previous_standard_onset_token_in_trial(y_one, tone_T=tone_T, isi_T=isi_T)
    prev_off = previous_standard_offset_token_in_trial(y_one, tone_T=tone_T, isi_T=isi_T)
    first_possible_on = first_possible_deviant_onset_token_in_trial(y_one, tone_T=tone_T, isi_T=isi_T)
    dev_on = deviant_onset_token_in_trial(y_one, tone_T=tone_T, isi_T=isi_T)
    dev_off = deviant_end_token_in_trial(y_one, tone_T=tone_T, isi_T=isi_T)
    next_on = next_tone_onset_token_in_trial(y_one, tone_T=tone_T, isi_T=isi_T)
    next_off = next_tone_offset_token_in_trial(y_one, tone_T=tone_T, isi_T=isi_T)
    second_next_on = second_next_tone_onset_token_in_trial(y_one, tone_T=tone_T, isi_T=isi_T)
    second_next_off = second_next_tone_offset_token_in_trial(y_one, tone_T=tone_T, isi_T=isi_T)
    win_start, win_end_excl, clipped = compute_correct_ce_window_bounds_in_trial(
        y_pos_456=y_one,
        trial_T_tokens=trial_T_tokens,
        tone_T=tone_T,
        isi_T=isi_T,
        correct_ce_window=correct_ce_window,
    )
    return {
        "example_deviant_position": int(y_one.item()),
        "example_previous_standard_onset_token": int(prev_on.item()),
        "example_previous_standard_offset_token": int(prev_off.item()),
        "example_first_possible_deviant_onset_token": int(first_possible_on.item()),
        "example_deviant_onset_token": int(dev_on.item()),
        "example_deviant_offset_token": int(dev_off.item()),
        "example_next_tone_onset_token": int(next_on.item()),
        "example_next_tone_offset_token": int(next_off.item()),
        "example_second_next_tone_onset_token": int(second_next_on.item()),
        "example_second_next_tone_offset_token": int(second_next_off.item()),
        "example_window_start": int(win_start.item()),
        "example_window_end_exclusive": int(win_end_excl.item()),
        "example_n_window_tokens": int((win_end_excl - win_start).item()),
        "example_window_clipped": int(clipped.item()),
    }


def compute_token_window_bounds_by_mode(
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_loss_mode: str,
    correct_ce_window: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_mode = _normalize_token_loss_mode(token_loss_mode)
    if normalized_mode == "old":
        start_bt = deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
        end_bt = next_standard_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
        clipped = torch.zeros_like(start_bt, dtype=torch.bool)
        return start_bt, end_bt, clipped
    return compute_correct_ce_window_bounds_in_trial(
        y_pos_456=y_pos_456,
        trial_T_tokens=trial_T_tokens,
        tone_T=tone_T,
        isi_T=isi_T,
        correct_ce_window=correct_ce_window,
    )


def debug_print_train_window_trials(
    *,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_ms: int,
    correct_ce_window: str,
    train_rt_diag_window: str,
    max_trials: int = 6,
) -> None:
    y_flat = y_pos_456.reshape(-1).long().detach().cpu()
    n_take = min(int(max_trials), int(y_flat.numel()))
    if n_take <= 0:
        return

    y_take = y_flat[:n_take]
    train_start, train_end_excl, _ = compute_correct_ce_window_bounds_in_trial(
        y_pos_456=y_take,
        trial_T_tokens=int(trial_T_tokens),
        tone_T=int(tone_T),
        isi_T=int(isi_T),
        correct_ce_window=str(correct_ce_window),
    )
    readout_start, readout_end = _legacy_readout_window_to_bounds(str(train_rt_diag_window))
    dummy_logits = torch.zeros((n_take, int(trial_T_tokens), 3), dtype=torch.float32)
    y_cls = labels_to_class_index(y_take)
    prepared = prepare_rt_readout(
        logits_trial=dummy_logits,
        y_cls=y_cls,
        y_pos_456=y_take,
        tone_T=int(tone_T),
        isi_T=int(isi_T),
        token_ms=int(token_ms),
        readout_start=str(readout_start),
        readout_end=str(readout_end),
        rt_reference="deviant_onset",
    )

    dev_on = deviant_onset_token_in_trial(y_take, tone_T=int(tone_T), isi_T=int(isi_T)).detach().cpu().numpy()
    dev_off = deviant_end_token_in_trial(y_take, tone_T=int(tone_T), isi_T=int(isi_T)).detach().cpu().numpy()
    tone4_on = tone_onset_token_in_trial(torch.full_like(y_take, 4), tone_T=int(tone_T), isi_T=int(isi_T)).detach().cpu().numpy()
    tone4_off = tone_offset_token_in_trial(torch.full_like(y_take, 4), tone_T=int(tone_T), isi_T=int(isi_T)).detach().cpu().numpy()
    tone5_on = tone_onset_token_in_trial(torch.full_like(y_take, 5), tone_T=int(tone_T), isi_T=int(isi_T)).detach().cpu().numpy()
    tone5_off = tone_offset_token_in_trial(torch.full_like(y_take, 5), tone_T=int(tone_T), isi_T=int(isi_T)).detach().cpu().numpy()

    print("[debug_train_window_trials] begin")
    for i in range(n_take):
        loc = int(y_take[i].item())
        train_first = int(train_start[i].item())
        train_last = int(train_end_excl[i].item() - 1)
        win_start = int(prepared.window_start[i])
        win_end_excl = int(prepared.window_end_exclusive[i])
        rt_ref_ms = float(prepared.rt_reference_time_ms[i])
        p4_ok_train = (loc != 4) or (train_first >= int(tone4_on[i]))
        p4_ok_rt = (loc != 4) or (win_start >= int(tone4_on[i]))
        print(
            "[debug_train_window_trials] "
            f"trial_id={i} y_pos_456={loc} loc={loc} correct_ce_window={correct_ce_window} "
            f"train_mask_first_token={train_first}({train_first * token_ms}ms) "
            f"train_mask_last_token={train_last}({train_last * token_ms}ms) "
            f"prepared.window_start={win_start}({prepared.window_start_time_ms[i]:.1f}ms) "
            f"prepared.window_end_exclusive={win_end_excl}({prepared.window_end_time_ms[i]:.1f}ms) "
            f"deviant_onset_idx={int(dev_on[i])}({int(dev_on[i]) * token_ms}ms) "
            f"deviant_end_idx={int(dev_off[i])}({int(dev_off[i]) * token_ms}ms) "
            f"tone4_onset_idx={int(tone4_on[i])}({int(tone4_on[i]) * token_ms}ms) "
            f"tone4_offset_idx={int(tone4_off[i])}({int(tone4_off[i]) * token_ms}ms) "
            f"tone5_onset_idx={int(tone5_on[i])}({int(tone5_on[i]) * token_ms}ms) "
            f"tone5_offset_idx={int(tone5_off[i])}({int(tone5_off[i]) * token_ms}ms) "
            f"rt_reference_time_ms={rt_ref_ms:.1f} "
            f"check_P4_train_mask_ge_tone4_on={p4_ok_train} "
            f"check_P4_prepared_start_ge_tone4_on={p4_ok_rt}"
        )
    print("[debug_train_window_trials] end")


def _nanmean_np(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _nanmedian_np(x: np.ndarray) -> float:
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


def _nanstd_np(x: np.ndarray) -> float:
    return float(np.nanstd(x)) if np.isfinite(x).any() else float("nan")


def _naniqr_np(x: np.ndarray) -> float:
    if not np.isfinite(x).any():
        return float("nan")
    q75, q25 = np.nanpercentile(x, [75, 25])
    return float(q75 - q25)


def compute_window_level_diagnostics(
    logits_all: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_ms: int,
    token_loss_mode: str,
    correct_ce_window: str,
) -> Dict[str, Any]:
    B, T, C = logits_all.shape
    if C != 3:
        raise ValueError(f"Expected logits_all (...,3), got shape={tuple(logits_all.shape)}")
    if T % 10 != 0:
        raise ValueError(f"T must be divisible by 10, got T={T}")
    Tt = int(trial_T_tokens)
    class_trial = logits_all.view(B, 10, Tt, 3)
    y_cls = labels_to_class_index(y_pos_456).long()
    probs = torch.softmax(class_trial, dim=-1)
    y_idx = y_cls.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Tt, 1)
    p_correct = probs.gather(dim=-1, index=y_idx).squeeze(-1)

    start_bt, end_bt, clipped_bt = compute_token_window_bounds_by_mode(
        y_pos_456=y_pos_456,
        trial_T_tokens=Tt,
        tone_T=tone_T,
        isi_T=isi_T,
        token_loss_mode=token_loss_mode,
        correct_ce_window=correct_ce_window,
    )
    n_tok_bt = (end_bt - start_bt).long()
    max_len = int(n_tok_bt.max().item()) if n_tok_bt.numel() > 0 else 0
    rel_idx = torch.arange(max_len, device=logits_all.device).view(1, 1, max_len)
    gather_idx = (start_bt.unsqueeze(-1) + rel_idx).clamp(max=Tt - 1)
    valid_mask = rel_idx < n_tok_bt.unsqueeze(-1)
    window_pc = p_correct.gather(dim=2, index=gather_idx)
    window_pc = torch.where(valid_mask, window_pc, torch.full_like(window_pc, float("nan")))
    gather_idx_probs = gather_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    window_probs = probs.gather(dim=2, index=gather_idx_probs)
    window_probs = torch.where(valid_mask.unsqueeze(-1), window_probs, torch.full_like(window_probs, float("nan")))

    n_minus_1 = (n_tok_bt - 1).clamp(min=0)
    idx_25 = torch.round(n_minus_1.float() * 0.25).long()
    idx_50 = torch.round(n_minus_1.float() * 0.50).long()
    idx_75 = torch.round(n_minus_1.float() * 0.75).long()
    idx_end = n_minus_1.long()

    def _gather_point(idx_bt: torch.Tensor) -> np.ndarray:
        vals = window_pc.gather(dim=2, index=idx_bt.unsqueeze(-1)).squeeze(-1)
        return vals.detach().cpu().numpy().reshape(-1).astype(np.float64)

    p_dev_on = _gather_point(torch.zeros_like(n_tok_bt))
    p_25 = _gather_point(idx_25)
    p_50 = _gather_point(idx_50)
    p_75 = _gather_point(idx_75)
    p_end = _gather_point(idx_end)

    valid_counts = valid_mask.sum(dim=2).clamp_min(1)
    mean_pc = torch.nan_to_num(window_pc, nan=0.0).sum(dim=2) / valid_counts
    mean_probs = torch.nan_to_num(window_probs, nan=0.0).sum(dim=2) / valid_counts.unsqueeze(-1)
    first_probs = window_probs[:, :, 0, :]
    end_probs = window_probs.gather(
        dim=2,
        index=idx_end.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
    ).squeeze(2)
    max_pc = torch.where(valid_mask, window_pc, torch.full_like(window_pc, float("-inf"))).max(dim=2).values
    token_loss_trial = -(torch.log(window_pc.clamp_min(1e-12)))
    token_loss_trial = torch.where(valid_mask, token_loss_trial, torch.zeros_like(token_loss_trial))
    token_loss_trial = token_loss_trial.sum(dim=2) / valid_counts
    if max_len > 1:
        dt_ms = float(token_ms)
        trap = 0.5 * (window_pc[:, :, 1:] + window_pc[:, :, :-1]) * dt_ms
        trap = torch.where(valid_mask[:, :, 1:] & valid_mask[:, :, :-1], trap, torch.zeros_like(trap))
        auc_pc = trap.sum(dim=2)
    else:
        auc_pc = torch.zeros_like(mean_pc)

    p_flat = window_pc.detach().cpu().numpy().reshape(-1, max_len).astype(np.float64)
    mean_flat = mean_pc.detach().cpu().numpy().reshape(-1).astype(np.float64)
    max_flat = max_pc.detach().cpu().numpy().reshape(-1).astype(np.float64)
    auc_flat = auc_pc.detach().cpu().numpy().reshape(-1).astype(np.float64)
    pos_flat = y_pos_456.detach().cpu().numpy().reshape(-1).astype(np.int64)
    y_true_flat = y_cls.detach().cpu().numpy().reshape(-1).astype(np.int64)
    prob_mean_flat = mean_probs.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    prob_first_flat = first_probs.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    prob_end_flat = end_probs.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    token_prob_flat = window_probs[valid_mask].detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    token_y_true_flat = y_cls.unsqueeze(-1).expand(-1, -1, max_len)[valid_mask].detach().cpu().numpy().reshape(-1).astype(np.int64)
    token_loss_trial_flat = token_loss_trial.detach().cpu().numpy().reshape(-1).astype(np.float64)
    n_tok_flat = n_tok_bt.detach().cpu().numpy().reshape(-1).astype(np.int64)
    clipped_flat = clipped_bt.detach().cpu().numpy().reshape(-1).astype(bool)
    early_len_flat = np.maximum(1, np.ceil(n_tok_flat.astype(np.float64) * 0.2).astype(np.int64))
    early_p_flat = np.full((p_flat.shape[0],), np.nan, dtype=np.float64)
    for i in range(p_flat.shape[0]):
        if n_tok_flat[i] > 0:
            early_p_flat[i] = _nanmean_np(p_flat[i, : early_len_flat[i]])

    rel_ms = np.arange(max_len, dtype=np.float64) * float(token_ms)
    thresholds = [0.50, 0.55, 0.60, 0.70, 0.90]
    crossing_stats: Dict[str, Dict[str, float]] = {}
    crossing_by_trial: Dict[str, np.ndarray] = {}
    crossing_idx_by_trial: Dict[str, np.ndarray] = {}
    for thr in thresholds:
        found = np.zeros((p_flat.shape[0],), dtype=bool)
        rt_ms = np.full((p_flat.shape[0],), np.nan, dtype=np.float64)
        rt_idx = np.full((p_flat.shape[0],), -1, dtype=np.int64)
        first_token = np.zeros((p_flat.shape[0],), dtype=bool)
        for i in range(p_flat.shape[0]):
            vals = p_flat[i, : n_tok_flat[i]]
            hits = np.where(vals >= thr)[0]
            if hits.size > 0:
                found[i] = True
                rt_idx[i] = int(hits[0])
                rt_ms[i] = float(hits[0]) * float(token_ms)
                first_token[i] = hits[0] == 0
        key = f"{thr:.2f}"
        crossing_by_trial[key] = rt_ms
        crossing_idx_by_trial[key] = rt_idx
        crossing_stats[key] = {
            "found_rate": float(found.mean()),
            "miss_rate": float((~found).mean()),
            "proportion_crossing_at_first_token": float(first_token.mean()),
            "median_rt_ms": _nanmedian_np(rt_ms),
            "std_rt_ms": _nanstd_np(rt_ms),
            "iqr_rt_ms": _naniqr_np(rt_ms),
        }

    traj_by_pos: Dict[int, np.ndarray] = {}
    by_position_rows: List[Dict[str, Any]] = []
    for pos in [4, 5, 6]:
        m = pos_flat == pos
        pos_end = _nanmean_np(p_end[m])
        pos_mean = _nanmean_np(mean_flat[m])
        pos_auc = _nanmean_np(auc_flat[m])
        rt050 = crossing_by_trial["0.50"][m]
        rt060 = crossing_by_trial["0.60"][m]
        idx050 = crossing_idx_by_trial["0.50"][m]
        idx055 = crossing_idx_by_trial["0.55"][m]
        idx060 = crossing_idx_by_trial["0.60"][m]
        idx070 = crossing_idx_by_trial["0.70"][m]
        idx090 = crossing_idx_by_trial["0.90"][m]
        y_pos_cls = y_true_flat[m]
        prob_pos = prob_end_flat[m]
        def _acc_at_cross_idx(idx_arr: np.ndarray, y_arr: np.ndarray, p_end_arr: np.ndarray) -> float:
            if idx_arr.size == 0:
                return float("nan")
            pred = np.argmax(p_end_arr, axis=1).astype(np.int64, copy=False)
            out = np.zeros((idx_arr.shape[0],), dtype=np.float64)
            found_mask = idx_arr >= 0
            out[found_mask] = (pred[found_mask] == y_arr[found_mask]).astype(np.float64)
            return float(out.mean())
        traj = np.nanmean(p_flat[m], axis=0) if np.any(m) else np.full((max_len,), np.nan, dtype=np.float64)
        traj_by_pos[pos] = traj
        by_position_rows.append(
            {
                "position": int(pos),
                "n_trials": int(np.sum(m)),
                "p_correct_at_window_end": pos_end,
                "mean_p_correct_in_window": pos_mean,
                "max_p_correct_in_window": _nanmean_np(max_flat[m]),
                "auc_p_correct_in_window": pos_auc,
                "first_crossing_050_ms": _nanmedian_np(rt050),
                "first_crossing_060_ms": _nanmedian_np(rt060),
                "found_rate_050": float(np.isfinite(rt050).mean()) if rt050.size > 0 else float("nan"),
                "acc_at_found_050": _acc_at_cross_idx(idx050, y_pos_cls, prob_pos),
                "acc_at_found_055": _acc_at_cross_idx(idx055, y_pos_cls, prob_pos),
                "found_rate_060": float(np.isfinite(rt060).mean()) if rt060.size > 0 else float("nan"),
                "acc_at_found_060": _acc_at_cross_idx(idx060, y_pos_cls, prob_pos),
                "found_rate_070": float(np.isfinite(crossing_by_trial["0.70"][m]).mean()) if np.any(m) else float("nan"),
                "acc_at_found_070": _acc_at_cross_idx(idx070, y_pos_cls, prob_pos),
                "found_rate_090": float(np.isfinite(crossing_by_trial["0.90"][m]).mean()) if np.any(m) else float("nan"),
                "acc_at_found_090": _acc_at_cross_idx(idx090, y_pos_cls, prob_pos),
                "proportion_rt_floor": float(np.nanmean(rt060 == 0.0)) if np.isfinite(rt060).any() else float("nan"),
            }
        )

    p_order_ok = bool(
        np.isfinite([row["p_correct_at_window_end"] for row in by_position_rows]).all()
        and by_position_rows[2]["p_correct_at_window_end"] > by_position_rows[1]["p_correct_at_window_end"] > by_position_rows[0]["p_correct_at_window_end"]
    )
    rt_order_ok = bool(
        np.isfinite([row["first_crossing_060_ms"] for row in by_position_rows]).all()
        and by_position_rows[0]["first_crossing_060_ms"] > by_position_rows[1]["first_crossing_060_ms"] > by_position_rows[2]["first_crossing_060_ms"]
    )

    summary = {
        "p_correct_at_dev_onset": _nanmean_np(p_dev_on),
        "p_correct_at_window_25pct": _nanmean_np(p_25),
        "p_correct_at_window_50pct": _nanmean_np(p_50),
        "p_correct_at_window_75pct": _nanmean_np(p_75),
        "p_correct_at_window_end": _nanmean_np(p_end),
        "mean_p_correct_in_window": _nanmean_np(mean_flat),
        "std_p_correct_in_window": _nanstd_np(mean_flat),
        "p_correct_p10": float(np.nanpercentile(mean_flat, 10)) if np.isfinite(mean_flat).any() else float("nan"),
        "p_correct_p50": float(np.nanpercentile(mean_flat, 50)) if np.isfinite(mean_flat).any() else float("nan"),
        "p_correct_p90": float(np.nanpercentile(mean_flat, 90)) if np.isfinite(mean_flat).any() else float("nan"),
        "early_p_correct_mean": _nanmean_np(early_p_flat),
        "max_p_correct_in_window": _nanmean_np(max_flat),
        "auc_p_correct_in_window": _nanmean_np(auc_flat),
        "window_prediction_mean": _nanmean_np(p_flat.reshape(-1)),
        "window_prediction_std": _nanstd_np(p_flat.reshape(-1)),
        "window_prediction_min": float(np.nanmin(p_flat)) if np.isfinite(p_flat).any() else float("nan"),
        "window_prediction_max": float(np.nanmax(p_flat)) if np.isfinite(p_flat).any() else float("nan"),
        "window_n_tokens_mean": float(np.mean(n_tok_flat)) if n_tok_flat.size > 0 else float("nan"),
        "window_n_tokens_min": int(np.min(n_tok_flat)) if n_tok_flat.size > 0 else 0,
        "window_n_tokens_max": int(np.max(n_tok_flat)) if n_tok_flat.size > 0 else 0,
        "window_clipped_rate": float(np.mean(clipped_flat.astype(np.float64))) if clipped_flat.size > 0 else 0.0,
        "p_correct_ordering_ok": int(p_order_ok),
        "rt_ordering_ok": int(rt_order_ok),
        "y_true_frac_class0": float(np.mean(y_true_flat == 0)) if y_true_flat.size > 0 else float("nan"),
        "y_true_frac_class1": float(np.mean(y_true_flat == 1)) if y_true_flat.size > 0 else float("nan"),
        "y_true_frac_class2": float(np.mean(y_true_flat == 2)) if y_true_flat.size > 0 else float("nan"),
        "trajectory_rel_ms": rel_ms,
        "trajectory_by_position": traj_by_pos,
        "by_position_rows": by_position_rows,
        "per_trial_position": pos_flat,
        "per_trial_y_true_cls": y_true_flat,
        "per_trial_p_end": p_end,
        "per_trial_mean": mean_flat,
        "per_trial_max": max_flat,
        "per_trial_auc": auc_flat,
        "per_trial_window_prob_mean": prob_mean_flat,
        "per_trial_window_prob_first": prob_first_flat,
        "per_trial_window_prob_end": prob_end_flat,
        "per_token_window_prob": token_prob_flat,
        "per_token_window_y_true_cls": token_y_true_flat,
        "per_trial_window_token_loss": token_loss_trial_flat,
        "per_trial_crossing_ms": crossing_by_trial,
        "per_trial_crossing_idx": crossing_idx_by_trial,
    }
    for thr_key, stats in crossing_stats.items():
        thr_label = thr_key.replace(".", "")
        summary[f"first_crossing_{thr_label}_ms"] = stats["median_rt_ms"]
        summary[f"found_rate_{thr_label}"] = stats["found_rate"]
        idx_arr = crossing_idx_by_trial[thr_key]
        pred = np.argmax(prob_end_flat, axis=1).astype(np.int64, copy=False) if prob_end_flat.size > 0 else np.array([], dtype=np.int64)
        if idx_arr.size > 0 and pred.size == idx_arr.size:
            acc_vec = np.zeros((idx_arr.shape[0],), dtype=np.float64)
            found_mask = idx_arr >= 0
            acc_vec[found_mask] = (pred[found_mask] == y_true_flat[found_mask]).astype(np.float64)
            summary[f"acc_at_found_{thr_label}"] = float(acc_vec.mean())
        else:
            summary[f"acc_at_found_{thr_label}"] = float("nan")
        summary[f"miss_rate_{thr_label}"] = stats["miss_rate"]
        summary[f"proportion_crossing_at_first_token_{thr_label}"] = stats["proportion_crossing_at_first_token"]
        summary[f"median_rt_ms_{thr_label}"] = stats["median_rt_ms"]
        summary[f"std_rt_ms_{thr_label}"] = stats["std_rt_ms"]
        summary[f"iqr_rt_ms_{thr_label}"] = stats["iqr_rt_ms"]
    return summary


def configure_trainable_parameters(model: nn.Module, freeze_variant: str) -> Dict[str, Any]:
    variant = str(freeze_variant).strip().lower()
    if variant in {"", "none"}:
        variant = "full_finetune"

    valid = {"full_finetune", "freeze_recurrent_core", "freeze_output_head"}
    if variant not in valid:
        raise ValueError(f"Unknown freeze_variant: {freeze_variant}")

    for _name, p in model.named_parameters():
        p.requires_grad = True

    core_prefixes = ("gru.", "ln.", "tone_pos_embed.", "tone_pos_proj.")
    head_prefixes = ("head.", "stop_head.", "event_head.", "next_tone_head.", "response_head.")

    def is_core(name: str) -> bool:
        return any(name.startswith(pref) for pref in core_prefixes)

    def is_head(name: str) -> bool:
        return any(name.startswith(pref) for pref in head_prefixes)

    if variant == "freeze_recurrent_core":
        for name, p in model.named_parameters():
            if is_core(name):
                p.requires_grad = False
    elif variant == "freeze_output_head":
        for name, p in model.named_parameters():
            if is_head(name):
                p.requires_grad = False

    trainable_names: List[str] = []
    frozen_names: List[str] = []
    trainable_n = 0
    frozen_n = 0
    core_trainable_n = 0
    head_trainable_n = 0
    other_trainable_n = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        if p.requires_grad:
            trainable_names.append(name)
            trainable_n += n
            if is_core(name):
                core_trainable_n += n
            elif is_head(name):
                head_trainable_n += n
            else:
                other_trainable_n += n
        else:
            frozen_names.append(name)
            frozen_n += n

    return {
        "freeze_variant": variant,
        "trainable_param_names": trainable_names,
        "frozen_param_names": frozen_names,
        "trainable_param_count": int(trainable_n),
        "frozen_param_count": int(frozen_n),
        "core_trainable_param_count": int(core_trainable_n),
        "head_trainable_param_count": int(head_trainable_n),
        "other_trainable_param_count": int(other_trainable_n),
    }


def compute_rf_ambiguity_diagnostics(
    freqs_blocks: torch.Tensor,
    y_pos_456: torch.Tensor,
    encoding_cfg: StimulusEncodingConfig,
) -> Dict[str, float]:
    x_np = freqs_blocks.detach().cpu().numpy().astype(np.float32)
    y_np = y_pos_456.detach().cpu().numpy().astype(np.int64)
    edges_erb = make_erb_edges(float(encoding_cfg.f_min_hz), float(encoding_cfg.f_max_hz), int(encoding_cfg.n_bins))
    cos_vals: List[float] = []
    l2_vals: List[float] = []
    overlap_vals: List[float] = []
    signal_power_vals: List[float] = []
    example_std_vec: Optional[np.ndarray] = None
    example_dev_vec: Optional[np.ndarray] = None
    for b in range(x_np.shape[0]):
        for tr in range(x_np.shape[1]):
            std_f = float(x_np[b, tr, 0])
            dev_pos = int(y_np[b, tr])
            dev_f = float(x_np[b, tr, dev_pos - 1])
            std_vec, _ = build_tone_encoding_vector(std_f, edges_erb, encoding_cfg)
            dev_vec, _ = build_tone_encoding_vector(dev_f, edges_erb, encoding_cfg)
            denom = (np.linalg.norm(std_vec) * np.linalg.norm(dev_vec)) + 1e-12
            cos_vals.append(float(np.dot(std_vec, dev_vec) / denom))
            l2_vals.append(float(np.linalg.norm(std_vec - dev_vec)))
            overlap_vals.append(float(np.minimum(std_vec, dev_vec).sum()))
            signal_power_vals.append(float(np.mean(std_vec ** 2)))
            if example_std_vec is None:
                example_std_vec = std_vec.astype(np.float32, copy=True)
                example_dev_vec = dev_vec.astype(np.float32, copy=True)
    sigma_rf_noise = float(getattr(encoding_cfg, "sigma_rf_noise", 0.0))
    if sigma_rf_noise > 0.0:
        noise_power = sigma_rf_noise ** 2
        input_snr_db = float(10.0 * np.log10(max(np.mean(signal_power_vals), 1e-12) / max(noise_power, 1e-12)))
    else:
        input_snr_db = float("nan")
    return {
        "sigma_rf": float(getattr(encoding_cfg, "sigma_rf", 1.0)),
        "sigma_rf_noise": sigma_rf_noise,
        "cosine_similarity_std_vs_dev": float(np.mean(cos_vals)) if cos_vals else float("nan"),
        "l2_distance_std_vs_dev": float(np.mean(l2_vals)) if l2_vals else float("nan"),
        "rf_overlap": float(np.mean(overlap_vals)) if overlap_vals else float("nan"),
        "input_snr_db": input_snr_db,
        "rf_curve_bin": list(range(int(encoding_cfg.n_bins))),
        "rf_curve_standard": ([] if example_std_vec is None else example_std_vec.astype(float).tolist()),
        "rf_curve_deviant": ([] if example_dev_vec is None else example_dev_vec.astype(float).tolist()),
    }


def append_rows_to_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_logits_all_chunked(
    model: PredictiveGRU,
    x: torch.Tensor,
    chunk_len: int,
) -> torch.Tensor:
    B, T, _ = x.shape
    h0 = None
    chunks: List[torch.Tensor] = []
    for s in range(0, T, int(chunk_len)):
        e = min(s + int(chunk_len), T)
        h_seq, h0 = model.forward_chunk(x[:, s:e, :], h0=h0)
        logits = model.classify_tokens(h_seq)
        chunks.append(logits.detach())
        h0 = (h0[0].detach(), h0[1].detach()) if isinstance(h0, tuple) else h0.detach()
    return torch.cat(chunks, dim=1)


def build_train_rt_diag_summary_lines(rows: List[Dict[str, Any]], epoch_global: int) -> List[str]:
    if not rows:
        return []
    wanted = [
        ("simple_threshold", 0.50, None, None, 1),
        ("simple_threshold", 0.90, None, None, 1),
        ("bayesian_cost", None, 5000.0, 0.50, 1),
    ]
    lines: List[str] = []
    for mode, pthr, timeout_ms, cthr, k in wanted:
        match = None
        for row in rows:
            if str(row.get("readout_mode")) != mode:
                continue
            if int(row.get("k_consec", -1)) != int(k):
                continue
            if mode == "simple_threshold":
                if abs(float(row.get("p_threshold", float("nan"))) - float(pthr)) < 1e-8:
                    match = row
                    break
            else:
                if abs(float(row.get("timeout_ms", float("nan"))) - float(timeout_ms)) < 1e-8 and abs(float(row.get("cost_threshold", float("nan"))) - float(cthr)) < 1e-8:
                    match = row
                    break
        if match is None:
            continue
        if mode == "simple_threshold":
            lines.append(
                f"[rt_diag] epoch={epoch_global:02d} mode=simple p={float(match['p_threshold']):.2f} k={int(match['k_consec'])} "
                f"found={float(match.get('found_rate', float('nan'))):.2f} floor0={float(match.get('proportion_rt_floor_0ms', float('nan'))):.2f} "
                f"floor5={float(match.get('proportion_rt_floor_5ms', float('nan'))):.2f} "
                f"meanRT={float(match.get('mean_rt_ms', float('nan'))):.1f}ms medianRT={float(match.get('median_rt_ms', float('nan'))):.1f}ms "
                f"P4={float(match.get('mean_rt_P4', float('nan'))):.1f} P5={float(match.get('mean_rt_P5', float('nan'))):.1f} P6={float(match.get('mean_rt_P6', float('nan'))):.1f} "
                f"ordering={bool(match.get('rt_ordering_P4_gt_P5_gt_P6', False))}"
            )
        else:
            lines.append(
                f"[rt_diag] epoch={epoch_global:02d} mode=cost timeout={int(round(float(match['timeout_ms'])))} cthr={float(match['cost_threshold']):.2f} k={int(match['k_consec'])} "
                f"found={float(match.get('found_rate', float('nan'))):.2f} floor0={float(match.get('proportion_rt_floor_0ms', float('nan'))):.2f} "
                f"floor5={float(match.get('proportion_rt_floor_5ms', float('nan'))):.2f} "
                f"meanRT={float(match.get('mean_rt_ms', float('nan'))):.1f}ms medianRT={float(match.get('median_rt_ms', float('nan'))):.1f}ms "
                f"P4={float(match.get('mean_rt_P4', float('nan'))):.1f} P5={float(match.get('mean_rt_P5', float('nan'))):.1f} P6={float(match.get('mean_rt_P6', float('nan'))):.1f} "
                f"ordering={bool(match.get('rt_ordering_P4_gt_P5_gt_P6', False))}"
            )
    return lines


def build_window_by_pos_summary_line(split_name: str, split_metrics: Dict[str, Any]) -> str:
    by_pos = split_metrics.get("window_diagnostics_by_position", []) or []
    if not by_pos:
        return f"[win_by_pos {split_name}] unavailable"
    rows = {int(r.get("position")): r for r in by_pos}
    parts: List[str] = []
    for pos in [4, 5, 6]:
        r = rows.get(pos, {})
        parts.append(
            f"P{pos} pEnd={float(r.get('p_correct_at_window_end', float('nan'))):.3f} "
            f"pPeak={float(r.get('max_p_correct_in_window', float('nan'))):.3f} "
            f"f@0.7={float(r.get('found_rate_070', float('nan'))):.3f} "
            f"f@0.9={float(r.get('found_rate_090', float('nan'))):.3f}"
        )
    return f"[win_by_pos {split_name}] " + " | ".join(parts)


def _threshold_key_from_prob(threshold: float) -> str:
    return f"{float(threshold):.2f}".replace(".", "")


def _init_pre_p4_probability_audit_accumulator() -> Dict[Tuple[str, int], Dict[str, Any]]:
    return {}


def make_pre_p4_uniform_mask_tokens(
    abs_t: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    within = (abs_t % int(trial_T_tokens)).long()
    tone3_offset_idx = int(tone_offset_token_in_trial(torch.tensor(3), tone_T=int(tone_T), isi_T=int(isi_T)).item())
    tone4_onset_idx = int(tone_onset_token_in_trial(torch.tensor(4), tone_T=int(tone_T), isi_T=int(isi_T)).item())
    return (within >= int(tone3_offset_idx)) & (within < int(tone4_onset_idx))


def compute_uniform_class_ce_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any().item()):
        return logits.new_tensor(0.0)
    log_probs = F.log_softmax(logits, dim=-1)
    loss_per_token = -log_probs.mean(dim=-1)
    return loss_per_token[mask].mean()


def make_pre_evidence_uniform_mask_tokens(
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    window: str,
) -> torch.Tensor:
    y_pos_456 = y_pos_456.long()
    tgrid = torch.arange(int(trial_T_tokens), device=y_pos_456.device).view(1, 1, -1)
    if str(window) == "trial_start_to_deviant_onset":
        end_tok = deviant_onset_token_in_trial(
            y_pos_456=y_pos_456,
            tone_T=int(tone_T),
            isi_T=int(isi_T),
        ).unsqueeze(-1)
    elif str(window) == "trial_start_to_p4_onset":
        p4 = torch.full_like(y_pos_456, 4)
        end_tok = tone_onset_token_in_trial(
            p4,
            tone_T=int(tone_T),
            isi_T=int(isi_T),
        ).unsqueeze(-1)
    else:
        raise ValueError(f"Unknown pre_evidence_uniform_kl_window: {window}")
    return tgrid < end_tok


def compute_uniform_class_kl_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any().item()):
        return logits.new_tensor(0.0)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    n_classes = int(logits.shape[-1])
    kl_per_token = (probs * (log_probs + math.log(float(n_classes)))).sum(dim=-1)
    return kl_per_token[mask].mean()


def _update_pre_p4_probability_audit_accumulator(
    acc: Dict[Tuple[str, int], Dict[str, Any]],
    logits_all: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_ms: int,
) -> None:
    logits_trial = logits_all.view(logits_all.shape[0], 10, int(trial_T_tokens), 3)
    probs_trial = torch.softmax(logits_trial, dim=-1)
    probs_flat = probs_trial.reshape(-1, int(trial_T_tokens), 3)
    y_flat = y_pos_456.reshape(-1).long()

    tone3_offset_idx = int(tone_offset_token_in_trial(torch.tensor(3), tone_T=int(tone_T), isi_T=int(isi_T)).item())
    tone4_onset_idx = int(tone_onset_token_in_trial(torch.tensor(4), tone_T=int(tone_T), isi_T=int(isi_T)).item())
    tone4_offset_idx = int(tone_offset_token_in_trial(torch.tensor(4), tone_T=int(tone_T), isi_T=int(isi_T)).item())
    tone5_offset_idx = int(tone_offset_token_in_trial(torch.tensor(5), tone_T=int(tone_T), isi_T=int(isi_T)).item())

    specs = [
        {
            "timepoint_label": "pre_P4_interval",
            "is_interval": True,
            "start_idx": int(max(0, tone3_offset_idx)),
            "end_exclusive_idx": int(min(int(trial_T_tokens), tone4_onset_idx)),
            "point_idx": None,
        },
        {
            "timepoint_label": "P4_onset",
            "is_interval": False,
            "start_idx": None,
            "end_exclusive_idx": None,
            "point_idx": int(min(max(0, tone4_onset_idx), int(trial_T_tokens) - 1)),
        },
        {
            "timepoint_label": "after_P4_offset",
            "is_interval": False,
            "start_idx": None,
            "end_exclusive_idx": None,
            "point_idx": int(min(max(0, tone4_offset_idx + 1), int(trial_T_tokens) - 1)),
        },
        {
            "timepoint_label": "after_P5_offset",
            "is_interval": False,
            "start_idx": None,
            "end_exclusive_idx": None,
            "point_idx": int(min(max(0, tone5_offset_idx + 1), int(trial_T_tokens) - 1)),
        },
    ]

    for pos in [4, 5, 6]:
        group_mask = (y_flat == int(pos))
        if not bool(group_mask.any().item()):
            continue
        group_probs = probs_flat[group_mask]
        y_cls = int(pos - 4)

        for spec in specs:
            key = (str(spec["timepoint_label"]), int(pos))
            if key not in acc:
                acc[key] = {
                    "timepoint_label": str(spec["timepoint_label"]),
                    "position": int(pos),
                    "group": f"P{int(pos)}",
                    "start_idx": spec["start_idx"],
                    "end_exclusive_idx": spec["end_exclusive_idx"],
                    "point_idx": spec["point_idx"],
                    "start_ms": (float(spec["start_idx"]) * float(token_ms)) if spec["start_idx"] is not None else float("nan"),
                    "end_exclusive_ms": (float(spec["end_exclusive_idx"]) * float(token_ms)) if spec["end_exclusive_idx"] is not None else float("nan"),
                    "point_ms": (float(spec["point_idx"]) * float(token_ms)) if spec["point_idx"] is not None else float("nan"),
                    "expected_uniform_prob": 1.0 / 3.0,
                    "n_trials": 0,
                    "n_values": 0,
                    "sum_p_class0": 0.0,
                    "sum_p_class1": 0.0,
                    "sum_p_class2": 0.0,
                    "sum_p_correct": 0.0,
                    "max_p_correct": float("-inf"),
                    "count_p_correct_ge_040": 0,
                    "count_p_correct_ge_050": 0,
                    "count_p_correct_ge_090": 0,
                }
            row = acc[key]
            row["n_trials"] += int(group_probs.shape[0])

            if bool(spec["is_interval"]):
                start_idx = int(spec["start_idx"])
                end_exclusive_idx = int(spec["end_exclusive_idx"])
                if end_exclusive_idx <= start_idx:
                    continue
                probs_sel = group_probs[:, start_idx:end_exclusive_idx, :]
                if probs_sel.numel() == 0:
                    continue
                p_correct = probs_sel[..., y_cls]
                row["n_values"] += int(p_correct.numel())
                row["sum_p_class0"] += float(probs_sel[..., 0].sum().item())
                row["sum_p_class1"] += float(probs_sel[..., 1].sum().item())
                row["sum_p_class2"] += float(probs_sel[..., 2].sum().item())
                row["sum_p_correct"] += float(p_correct.sum().item())
                row["max_p_correct"] = max(float(row["max_p_correct"]), float(p_correct.max().item()))
                row["count_p_correct_ge_040"] += int((p_correct >= 0.40).sum().item())
                row["count_p_correct_ge_050"] += int((p_correct >= 0.50).sum().item())
                row["count_p_correct_ge_090"] += int((p_correct >= 0.90).sum().item())
            else:
                point_idx = int(spec["point_idx"])
                probs_sel = group_probs[:, point_idx, :]
                if probs_sel.numel() == 0:
                    continue
                p_correct = probs_sel[:, y_cls]
                row["n_values"] += int(p_correct.numel())
                row["sum_p_class0"] += float(probs_sel[:, 0].sum().item())
                row["sum_p_class1"] += float(probs_sel[:, 1].sum().item())
                row["sum_p_class2"] += float(probs_sel[:, 2].sum().item())
                row["sum_p_correct"] += float(p_correct.sum().item())
                row["max_p_correct"] = max(float(row["max_p_correct"]), float(p_correct.max().item()))
                row["count_p_correct_ge_040"] += int((p_correct >= 0.40).sum().item())
                row["count_p_correct_ge_050"] += int((p_correct >= 0.50).sum().item())
                row["count_p_correct_ge_090"] += int((p_correct >= 0.90).sum().item())


def finalize_pre_p4_probability_audit(
    acc: Dict[Tuple[str, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (_key, row) in sorted(acc.items(), key=lambda kv: (str(kv[0][0]), int(kv[0][1]))):
        n_values = int(row.get("n_values", 0))
        denom = float(max(1, n_values))
        mean_p_class0 = float(row["sum_p_class0"]) / denom
        mean_p_class1 = float(row["sum_p_class1"]) / denom
        mean_p_class2 = float(row["sum_p_class2"]) / denom
        mean_p_correct = float(row["sum_p_correct"]) / denom
        max_p_correct = float(row["max_p_correct"]) if n_values > 0 else float("nan")
        prop_ge_040 = float(row["count_p_correct_ge_040"]) / denom
        prop_ge_050 = float(row["count_p_correct_ge_050"]) / denom
        prop_ge_090 = float(row["count_p_correct_ge_090"]) / denom
        uniform = float(row.get("expected_uniform_prob", 1.0 / 3.0))
        uniform_l1 = float(abs(mean_p_class0 - uniform) + abs(mean_p_class1 - uniform) + abs(mean_p_class2 - uniform))
        leakage = bool(
            str(row.get("timepoint_label")) == "pre_P4_interval"
            and (
                mean_p_correct >= 0.40
                or prop_ge_090 > 0.0
                or max_p_correct >= 0.90
            )
        )
        rows.append({
            "timepoint_label": str(row["timepoint_label"]),
            "position": int(row["position"]),
            "group": str(row["group"]),
            "n_trials": int(row["n_trials"]),
            "n_values": int(n_values),
            "start_idx": row.get("start_idx"),
            "end_exclusive_idx": row.get("end_exclusive_idx"),
            "point_idx": row.get("point_idx"),
            "start_ms": row.get("start_ms"),
            "end_exclusive_ms": row.get("end_exclusive_ms"),
            "point_ms": row.get("point_ms"),
            "mean_p_class0": mean_p_class0,
            "mean_p_class1": mean_p_class1,
            "mean_p_class2": mean_p_class2,
            "mean_p_correct": mean_p_correct,
            "max_p_correct": max_p_correct,
            "prop_p_correct_ge_040": prop_ge_040,
            "prop_p_correct_ge_050": prop_ge_050,
            "prop_p_correct_ge_090": prop_ge_090,
            "expected_uniform_prob": uniform,
            "uniform_l1_distance": uniform_l1,
            "leakage_or_memorization": leakage,
        })
    return rows


def build_pre_p4_probability_audit_summary_line(split_name: str, split_metrics: Dict[str, Any]) -> str:
    rows = split_metrics.get("pre_p4_probability_audit_rows", []) or []
    if not rows:
        return f"[pre_p4_prob {split_name}] unavailable"
    def _norm_label(x: Any) -> str:
        return str(x).strip().lower().replace("-", "_")
    wanted = [
        r for r in rows
        if _norm_label(r.get("timepoint_label")) in {"pre_p4_interval", "prep4_interval", "pre_p4"}
    ]
    if not wanted:
        labels_dbg = sorted({_norm_label(r.get("timepoint_label")) for r in rows})
        return f"[pre_p4_prob {split_name}] unavailable labels={labels_dbg}"
    parts: List[str] = []
    by_pos = {int(r.get("position")): r for r in wanted}
    for pos in [4, 5, 6]:
        r = by_pos.get(pos, {})
        parts.append(
            f"P{pos} probs=[{float(r.get('mean_p_class0', float('nan'))):.3f},"
            f"{float(r.get('mean_p_class1', float('nan'))):.3f},"
            f"{float(r.get('mean_p_class2', float('nan'))):.3f}] "
            f"pCorr={float(r.get('mean_p_correct', float('nan'))):.3f} "
            f"ge0.9={float(r.get('prop_p_correct_ge_090', float('nan'))):.3f}"
        )
    return f"[pre_p4_prob {split_name}] " + " | ".join(parts)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(min(1.0, max(0.0, x)))


def _serialize_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _serialize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_jsonable(v) for v in value]
    return value


def _extract_epoch_trajectory_payload(split_metrics: Dict[str, Any]) -> Dict[str, Any]:
    wd = split_metrics.get("window_diagnostics", {}) or {}
    by_pos = split_metrics.get("window_diagnostics_by_position", []) or []
    payload: Dict[str, Any] = {
        "trajectory_rel_ms": _serialize_jsonable(wd.get("trajectory_rel_ms", [])),
        "trajectory_by_position": _serialize_jsonable(wd.get("trajectory_by_position", {})),
        "p_correct_at_dev_onset": _safe_float(wd.get("p_correct_at_dev_onset")),
        "p_correct_at_window_end": _safe_float(wd.get("p_correct_at_window_end")),
        "window_p_correct_mean": _safe_float(wd.get("window_p_correct_mean", wd.get("mean_p_correct_in_window"))),
        "window_p_correct_max": _safe_float(wd.get("window_p_correct_max", wd.get("max_p_correct_in_window"))),
        "window_p_correct_auc": _safe_float(wd.get("window_p_correct_auc", wd.get("auc_p_correct_in_window"))),
        "window_n_tokens_mean": _safe_float(wd.get("window_n_tokens_mean")),
        "by_position_rows": _serialize_jsonable(by_pos),
    }
    traj_by_pos = wd.get("trajectory_by_position", {}) or {}
    rel_ms = wd.get("trajectory_rel_ms", [])
    if rel_ms is None:
        rel_ms = []
    if isinstance(rel_ms, str):
        try:
            rel_ms = json.loads(rel_ms)
        except Exception:
            try:
                rel_ms = ast.literal_eval(rel_ms)
            except Exception:
                rel_ms = []
    rel_ms_np = np.asarray(rel_ms, dtype=float) if isinstance(rel_ms, (list, tuple, np.ndarray)) else np.array([], dtype=float)
    for pos_key, series in traj_by_pos.items():
        try:
            arr = np.asarray(series, dtype=float)
        except Exception:
            continue
        for thr in [0.50, 0.80, 0.90]:
            idx = np.where(arr >= thr)[0]
            t_val = float(rel_ms_np[idx[0]]) if (idx.size > 0 and rel_ms_np.size == arr.size) else float("nan")
            payload[f"time_to_p{int(round(thr * 100))}_ms_{pos_key}"] = t_val
    return payload


def compute_rt_candidate_score_for_row(
    rt_row: Dict[str, Any],
    val_metrics: Dict[str, Any],
    min_win_auc: float,
) -> Dict[str, Any]:
    win_auc = _safe_float(val_metrics.get("window_auc"))
    val_tok = _safe_float(val_metrics.get("token_loss"))
    found_rate = _clip01(_safe_float(rt_row.get("found_rate"), 0.0))
    floor0 = _safe_float(rt_row.get("proportion_rt_floor_0ms"), float("nan"))
    floor5 = _safe_float(rt_row.get("proportion_rt_floor_5ms"), float("nan"))
    if found_rate <= 0.0:
        floor_metric = 0.0
    else:
        floors = [v for v in [floor0, floor5] if np.isfinite(v)]
        floor_metric = max(floors) if floors else float("nan")
    floor_metric = 0.0 if not np.isfinite(floor_metric) else _clip01(floor_metric)
    nonfloor_component = _clip01(1.0 - floor_metric) if np.isfinite(floor_metric) else 0.0
    miss_penalty = _clip01(1.0 - found_rate) if np.isfinite(found_rate) else 1.0
    if np.isfinite(win_auc):
        valid_win_auc_component = _clip01((win_auc - 0.5) / 0.5)
    else:
        valid_win_auc_component = 0.0
    mean_rt = _safe_float(rt_row.get("mean_rt_ms"), float("nan"))
    if not np.isfinite(mean_rt):
        rt_range_component = 0.0
    elif mean_rt < 5.0:
        rt_range_component = 0.0
    elif 50.0 <= mean_rt <= 500.0:
        rt_range_component = 1.0
    elif mean_rt < 50.0:
        rt_range_component = _clip01(mean_rt / 50.0)
    elif mean_rt <= 1000.0:
        rt_range_component = _clip01(1.0 - ((mean_rt - 500.0) / 500.0))
    else:
        rt_range_component = 0.0

    eligible = True
    notes: List[str] = []
    if np.isfinite(win_auc) and win_auc < float(min_win_auc):
        eligible = False
        notes.append(f"win_auc<{float(min_win_auc):.2f}")
    if np.isfinite(val_tok) and np.isfinite(win_auc) and val_tok >= 1.08 and win_auc <= 0.60:
        eligible = False
        notes.append("val_tok_random_and_win_auc_low")
    if not np.isfinite(found_rate) or found_rate <= 0.0:
        eligible = False
        notes.append("found_rate_zero")

    score = (
        2.0 * float(valid_win_auc_component)
        + 1.0 * float(found_rate if np.isfinite(found_rate) else 0.0)
        + 1.0 * float(nonfloor_component if np.isfinite(nonfloor_component) else 0.0)
        + 0.5 * float(rt_range_component)
        - 1.0 * float(floor_metric if np.isfinite(floor_metric) else 0.0)
        - 1.0 * float(miss_penalty if np.isfinite(miss_penalty) else 1.0)
    )
    if not eligible:
        score = float("-inf")
    return {
        "rt_candidate_score": float(score),
        "rt_candidate_eligible": bool(eligible),
        "rt_candidate_notes": ";".join(notes),
        "found_rate": float(found_rate) if np.isfinite(found_rate) else float("nan"),
        "floor0": float(floor0),
        "floor5": float(floor5),
        "meanRT": float(mean_rt),
    }


def choose_rt_candidate_row(
    rt_rows: List[Dict[str, Any]],
    val_metrics: Dict[str, Any],
    prefer_mode: str,
    prefer_threshold: float,
    min_win_auc: float,
) -> Dict[str, Any]:
    if not rt_rows:
        return {"selected_row": None, "selected_score": float("-inf"), "selected_meta": None}
    rows = list(rt_rows)
    if str(prefer_mode) == "simple":
        simple_rows = [r for r in rows if str(r.get("readout_mode")) == "simple_threshold"]
        if simple_rows:
            pref = [r for r in simple_rows if np.isfinite(_safe_float(r.get("p_threshold"))) and abs(_safe_float(r.get("p_threshold")) - float(prefer_threshold)) < 1e-8]
            if pref:
                rows = pref
            else:
                simple_rows_sorted = sorted(simple_rows, key=lambda r: _safe_float(r.get("p_threshold"), float("-inf")))
                rows = [simple_rows_sorted[-1]]
        else:
            rows = [] if str(prefer_mode) != "any" else rows
    elif str(prefer_mode) == "cost":
        cost_rows = [r for r in rows if str(r.get("readout_mode")) == "bayesian_cost"]
        rows = cost_rows if cost_rows else ([] if str(prefer_mode) != "any" else rows)

    candidate_pool = rows if rows else rt_rows
    scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for row in candidate_pool:
        meta = compute_rt_candidate_score_for_row(row, val_metrics=val_metrics, min_win_auc=float(min_win_auc))
        scored.append((float(meta["rt_candidate_score"]), row, meta))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_row, best_meta = scored[0]
    return {"selected_row": best_row, "selected_score": best_score, "selected_meta": best_meta}


def _choose_behavior_reference_rt_row(
    rt_rows: List[Dict[str, Any]],
    preferred_mode: str = "simple_threshold",
    preferred_p: float = 0.50,
    preferred_k: int = 1,
) -> Optional[Dict[str, Any]]:
    if not rt_rows:
        return None
    rows = list(rt_rows)
    preferred = [r for r in rows if str(r.get("readout_mode")) == str(preferred_mode)]
    if preferred:
        rows = preferred
    rows.sort(
        key=lambda r: (
            abs(_safe_float(r.get("p_threshold"), preferred_p) - float(preferred_p)),
            abs(int(r.get("k_consec", preferred_k)) - int(preferred_k)),
            -_safe_float(r.get("found_rate"), 0.0),
        )
    )
    return rows[0] if rows else None


def _epoch_behavior_diagnostics(
    val_metrics: Dict[str, Any],
    rt_rows: List[Dict[str, Any]],
    token_ms: int,
) -> Dict[str, Any]:
    rt_ref = _choose_behavior_reference_rt_row(rt_rows)
    mean_rt_ms = _safe_float((rt_ref or {}).get("mean_rt_ms"), _safe_float(val_metrics.get("mean_rt_ms")))
    mean_rt_tokens = _safe_float(val_metrics.get("mean_rt_tokens"))
    if (not np.isfinite(mean_rt_tokens)) and np.isfinite(mean_rt_ms):
        mean_rt_tokens = float(mean_rt_ms) / max(float(token_ms), 1e-12)
    floor0 = _safe_float((rt_ref or {}).get("proportion_rt_floor_0ms"))
    floor5 = _safe_float((rt_ref or {}).get("proportion_rt_floor_5ms"))
    p4 = _safe_float((rt_ref or {}).get("mean_rt_P4"))
    p5 = _safe_float((rt_ref or {}).get("mean_rt_P5"))
    p6 = _safe_float((rt_ref or {}).get("mean_rt_P6"))
    ordering = bool((rt_ref or {}).get("rt_ordering_P4_gt_P5_gt_P6", False))
    val_win_acc = _safe_float(val_metrics.get("window_acc"))
    val_loss = _safe_float(val_metrics.get("total_loss"))
    online_loss_mean = _safe_float(val_metrics.get("online_loss_mean"), _safe_float(val_metrics.get("token_loss")))
    p50 = _safe_float(val_metrics.get("window_p_correct_p50", val_metrics.get("p_correct_p50")))
    p90 = _safe_float(val_metrics.get("window_p_correct_p90", val_metrics.get("p_correct_p90")))
    early_p = _safe_float(val_metrics.get("early_p_correct_mean"))

    reasons: List[str] = []
    if np.isfinite(mean_rt_ms) and mean_rt_ms < 30.0:
        reasons.append("meanRT_ms_lt_30")
    if np.isfinite(mean_rt_tokens) and mean_rt_tokens < 6.0:
        reasons.append("meanRT_tokens_lt_6")
    if np.isfinite(floor0) and floor0 > 0.70:
        reasons.append("floor0_gt_0.70")
    if np.isfinite(val_win_acc) and val_win_acc >= 0.98 and np.isfinite(mean_rt_ms) and mean_rt_ms < 50.0:
        reasons.append("win_acc_ge_0.98_and_meanRT_ms_lt_50")
    if np.isfinite(val_win_acc) and val_win_acc >= 0.98 and np.isfinite(mean_rt_tokens) and mean_rt_tokens < 10.0:
        reasons.append("win_acc_ge_0.98_and_meanRT_tokens_lt_10")
    if all(np.isfinite(v) for v in [p4, p5, p6]) and all(abs(v) <= 1e-6 for v in [p4, p5, p6]):
        reasons.append("P4_P5_P6_all_zero")
    if ((np.isfinite(early_p) and early_p > 0.98) or (np.isfinite(p90) and p90 > 0.98)) and np.isfinite(mean_rt_ms) and mean_rt_ms < 50.0:
        reasons.append("early_p_correct_or_p90_saturated_with_fast_RT")
    if np.isfinite(val_win_acc) and val_win_acc > 0.40 and (not np.isfinite(mean_rt_ms)) and (not np.isfinite(mean_rt_tokens)):
        reasons.append("rt_missing_for_above_chance_epoch")
    if np.isfinite(val_win_acc) and val_win_acc <= 0.40:
        reasons.append("val_win_acc_le_0.40")

    collapse_invalid = bool(reasons)
    strong_valid = (not collapse_invalid) and np.isfinite(val_win_acc) and (val_win_acc > 0.40)
    fallback_score = (
        1 if (np.isfinite(mean_rt_ms) and mean_rt_ms >= 30.0) else 0,
        1 if (np.isfinite(val_win_acc) and val_win_acc > 0.40) else 0,
        0 if collapse_invalid else 1,
        _safe_float(val_win_acc, -1e9),
        _safe_float(mean_rt_ms, -1e9),
        -_safe_float(val_loss, 1e9),
    )
    return {
        "rt_row": rt_ref,
        "meanRT_ms": float(mean_rt_ms),
        "batch_meanRT_tokens": float(mean_rt_tokens),
        "batch_meanRT_ms": float(mean_rt_ms),
        "floor0": float(floor0),
        "floor5": float(floor5),
        "P4": float(p4),
        "P5": float(p5),
        "P6": float(p6),
        "ordering": bool(ordering),
        "val_win_acc": float(val_win_acc),
        "val_loss": float(val_loss),
        "online_loss_mean": float(online_loss_mean),
        "p_correct_p50": float(p50),
        "p_correct_p90": float(p90),
        "early_p_correct_mean": float(early_p),
        "collapse_invalid": bool(collapse_invalid),
        "collapse_reason": ";".join(reasons),
        "strong_valid": bool(strong_valid),
        "fallback_score": fallback_score,
    }


@torch.no_grad()
def run_validation_rt_readout_diagnostics(
    model: PredictiveGRU,
    loader,
    device: torch.device,
    chunk_len: int,
    tone_T: int,
    isi_T: int,
    trial_T_tokens: int,
    token_ms: int,
    readout_window: str,
    rt_readout_mode: str,
    p_thresholds: Sequence[float],
    cost_weights: Sequence[float],
    cost_timeouts_ms: Sequence[float],
    cost_thresholds: Sequence[float],
    k_consec_list: Sequence[int],
    max_trials: int,
    epoch_global: int,
    isi_ms: int,
    checkpoint_label: str,
    supervision_mode: str = "post_deviant",
) -> List[Dict[str, Any]]:
    model.eval()
    trial_rows: List[Dict[str, Any]] = []
    seen_trials = 0
    max_trials = int(max(0, max_trials))
    for batch in loader:
        if max_trials > 0 and seen_trials >= max_trials:
            break
        x, y, _gap_hz = unpack_batch_with_optional_gap(batch)
        x = x.to(device, non_blocking=True)
        y = y.long().to(device, non_blocking=True)
        logits_all = collect_logits_all_chunked(model=model, x=x, chunk_len=int(chunk_len))
        B, T, _ = logits_all.shape
        logits_trial = logits_all.view(B, 10, int(trial_T_tokens), int(logits_all.shape[-1]))
        n_take_trials = B * 10
        if max_trials > 0:
            n_take_trials = min(n_take_trials, max_trials - seen_trials)
        flat_logits = logits_trial.reshape(B * 10, int(trial_T_tokens), int(logits_all.shape[-1]))[:n_take_trials].detach().cpu()
        flat_y_pos = y.reshape(B * 10)[:n_take_trials].detach().cpu()
        flat_y_cls = labels_to_class_index(flat_y_pos).detach().cpu()
        readout_start, readout_end = _legacy_readout_window_to_bounds(str(readout_window))
        prepared = prepare_rt_readout(
            logits_trial=flat_logits,
            y_cls=flat_y_cls,
            y_pos_456=flat_y_pos,
            tone_T=int(tone_T),
            isi_T=int(isi_T),
            token_ms=int(token_ms),
            readout_start=str(readout_start),
            readout_end=str(readout_end),
            rt_reference="deviant_onset",
            supervision_mode=str(supervision_mode),
        )
        batch_trial_rows, _ = run_rt_readout_sweeps(
            prepared=prepared,
            rt_readout_mode=str(rt_readout_mode),
            p_threshold_list=p_thresholds,
            cost_w_list=cost_weights,
            cost_timeout_ms_list=cost_timeouts_ms,
            cost_threshold_list=cost_thresholds,
            k_consec_list=k_consec_list,
        )
        for row in batch_trial_rows:
            row.update({
                "epoch": int(epoch_global),
                "epoch_global": int(epoch_global),
                "isi_ms": int(isi_ms),
                "token_ms": int(token_ms),
                "checkpoint_label": str(checkpoint_label),
            })
        trial_rows.extend(batch_trial_rows)
        seen_trials += n_take_trials
    cond_rows = aggregate_readout_trial_rows(trial_rows)
    for row in cond_rows:
        row.update({
            "epoch": int(epoch_global),
            "epoch_global": int(epoch_global),
            "isi_ms": int(isi_ms),
            "token_ms": int(token_ms),
            "checkpoint_label": str(checkpoint_label),
        })
    return cond_rows


def finalize_window_epoch_diagnostics(
    batch_diags: List[Dict[str, Any]],
    token_ms: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not batch_diags:
        return {}, []
    pos = np.concatenate([d["per_trial_position"] for d in batch_diags], axis=0)
    y_true_cls = np.concatenate([d["per_trial_y_true_cls"] for d in batch_diags], axis=0)
    p_end = np.concatenate([d["per_trial_p_end"] for d in batch_diags], axis=0)
    p_mean = np.concatenate([d["per_trial_mean"] for d in batch_diags], axis=0)
    p_max = np.concatenate([d["per_trial_max"] for d in batch_diags], axis=0)
    p_auc = np.concatenate([d["per_trial_auc"] for d in batch_diags], axis=0)
    prob_mean = np.concatenate([d["per_trial_window_prob_mean"] for d in batch_diags], axis=0)
    prob_first = np.concatenate([d["per_trial_window_prob_first"] for d in batch_diags], axis=0)
    prob_end = np.concatenate([d["per_trial_window_prob_end"] for d in batch_diags], axis=0)
    token_prob = np.concatenate([d["per_token_window_prob"] for d in batch_diags], axis=0)
    token_y_true_cls = np.concatenate([d["per_token_window_y_true_cls"] for d in batch_diags], axis=0)
    token_loss_trial = np.concatenate([d["per_trial_window_token_loss"] for d in batch_diags], axis=0)
    crossing = {
        k: np.concatenate([d["per_trial_crossing_ms"][k] for d in batch_diags], axis=0)
        for k in ["0.50", "0.55", "0.60", "0.70", "0.90"]
    }
    crossing_idx = {
        k: np.concatenate([d["per_trial_crossing_idx"][k] for d in batch_diags], axis=0)
        for k in ["0.50", "0.55", "0.60", "0.70", "0.90"]
    }
    rel_ms = batch_diags[0]["trajectory_rel_ms"]
    mean_metrics = compute_multiclass_metrics_from_probs(y_true_cls, prob_mean, n_classes=3)
    first_metrics = compute_multiclass_metrics_from_probs(y_true_cls, prob_first, n_classes=3)
    end_metrics = compute_multiclass_metrics_from_probs(y_true_cls, prob_end, n_classes=3)
    token_metrics = compute_multiclass_metrics_from_probs(token_y_true_cls, token_prob, n_classes=3)

    traj_by_pos: Dict[int, np.ndarray] = {}
    by_pos_rows: List[Dict[str, Any]] = []
    for p in [4, 5, 6]:
        trajs = [d["trajectory_by_position"][p] for d in batch_diags if p in d["trajectory_by_position"]]
        if trajs:
            stack = np.stack(trajs, axis=0)
            traj = np.nanmean(stack, axis=0)
        else:
            traj = np.full((len(rel_ms),), np.nan, dtype=np.float64)
        traj_by_pos[p] = traj
        m = pos == p
        rt050 = crossing["0.50"][m]
        rt060 = crossing["0.60"][m]
        idx050 = crossing_idx["0.50"][m]
        idx055 = crossing_idx["0.55"][m]
        idx060 = crossing_idx["0.60"][m]
        idx070 = crossing_idx["0.70"][m]
        idx090 = crossing_idx["0.90"][m]
        pos_prob = prob_end[m]
        pos_pred = np.argmax(pos_prob, axis=1) if pos_prob.size > 0 else np.array([], dtype=np.int64)
        pos_acc = float(np.mean(pos_pred == y_true_cls[m])) if np.any(m) else float("nan")
        def _acc_at_cross_idx(idx_arr: np.ndarray, y_arr: np.ndarray, p_end_arr: np.ndarray) -> float:
            if idx_arr.size == 0:
                return float("nan")
            pred = np.argmax(p_end_arr, axis=1).astype(np.int64, copy=False)
            out = np.zeros((idx_arr.shape[0],), dtype=np.float64)
            found_mask = idx_arr >= 0
            out[found_mask] = (pred[found_mask] == y_arr[found_mask]).astype(np.float64)
            return float(out.mean())
        by_pos_rows.append(
            {
                "position": int(p),
                "n_trials": int(np.sum(m)),
                "window_acc": pos_acc,
                "window_f1_class": float(end_metrics["f1_per_class"][int(p - 4)]),
                "window_auc_class_ovr": float(end_metrics["auc_per_class"][int(p - 4)]),
                "p_correct_at_window_end": _nanmean_np(p_end[m]),
                "mean_p_correct_in_window": _nanmean_np(p_mean[m]),
                "max_p_correct_in_window": _nanmean_np(p_max[m]),
                "auc_p_correct_in_window": _nanmean_np(p_auc[m]),
                "window_token_loss": _nanmean_np(token_loss_trial[m]),
                "first_crossing_050_ms": _nanmedian_np(rt050),
                "first_crossing_060_ms": _nanmedian_np(rt060),
                "found_rate_050": float(np.isfinite(rt050).mean()) if rt050.size > 0 else float("nan"),
                "acc_at_found_050": _acc_at_cross_idx(idx050, y_true_cls[m], pos_prob),
                "acc_at_found_055": _acc_at_cross_idx(idx055, y_true_cls[m], pos_prob),
                "found_rate_060": float(np.isfinite(rt060).mean()) if rt060.size > 0 else float("nan"),
                "acc_at_found_060": _acc_at_cross_idx(idx060, y_true_cls[m], pos_prob),
                "found_rate_070": float(np.isfinite(crossing["0.70"][m]).mean()) if rt050.size > 0 else float("nan"),
                "acc_at_found_070": _acc_at_cross_idx(idx070, y_true_cls[m], pos_prob),
                "found_rate_090": float(np.isfinite(crossing["0.90"][m]).mean()) if rt050.size > 0 else float("nan"),
                "acc_at_found_090": _acc_at_cross_idx(idx090, y_true_cls[m], pos_prob),
                "proportion_rt_floor": float(np.nanmean(rt060 == 0.0)) if np.isfinite(rt060).any() else float("nan"),
            }
        )

    summary = {
        "window_acc": float(end_metrics["acc"]),
        "window_f1": float(end_metrics["f1_macro"]),
        "window_auc": float(end_metrics["auc_ovr"]),
        "window_acc_first_token": float(first_metrics["acc"]),
        "window_acc_last_token": float(end_metrics["acc"]),
        "window_acc_mean_token": float(token_metrics["acc"]),
        "window_auc_mean_token": float(token_metrics["auc_ovr"]),
        "window_acc_prob_mean": float(mean_metrics["acc"]),
        "window_f1_prob_mean": float(mean_metrics["f1_macro"]),
        "window_auc_prob_mean": float(mean_metrics["auc_ovr"]),
        "p_correct_at_dev_onset": float(np.mean([d["p_correct_at_dev_onset"] for d in batch_diags])),
        "p_correct_at_window_25pct": float(np.mean([d["p_correct_at_window_25pct"] for d in batch_diags])),
        "p_correct_at_window_50pct": float(np.mean([d["p_correct_at_window_50pct"] for d in batch_diags])),
        "p_correct_at_window_75pct": float(np.mean([d["p_correct_at_window_75pct"] for d in batch_diags])),
        "p_correct_at_window_end": _nanmean_np(p_end),
        "window_p_correct_mean": _nanmean_np(p_mean),
        "window_p_correct_end": _nanmean_np(p_end),
        "window_p_correct_max": float(np.mean([d["max_p_correct_in_window"] for d in batch_diags])),
        "window_p_correct_auc": _nanmean_np(p_auc),
        "window_p_correct_std": _nanmean_np([d.get("std_p_correct_in_window", float("nan")) for d in batch_diags]),
        "window_p_correct_p10": _nanmean_np([d.get("p_correct_p10", float("nan")) for d in batch_diags]),
        "window_p_correct_p50": _nanmean_np([d.get("p_correct_p50", float("nan")) for d in batch_diags]),
        "window_p_correct_p90": _nanmean_np([d.get("p_correct_p90", float("nan")) for d in batch_diags]),
        "early_p_correct_mean": _nanmean_np([d.get("early_p_correct_mean", float("nan")) for d in batch_diags]),
        "window_token_loss": _nanmean_np(token_loss_trial),
        "mean_p_correct_in_window": _nanmean_np(p_mean),
        "max_p_correct_in_window": float(np.mean([d["max_p_correct_in_window"] for d in batch_diags])),
        "auc_p_correct_in_window": _nanmean_np(p_auc),
        "window_prediction_mean": _nanmean_np([d.get("window_prediction_mean", float("nan")) for d in batch_diags]),
        "window_prediction_std": _nanmean_np([d.get("window_prediction_std", float("nan")) for d in batch_diags]),
        "window_prediction_min": _nanmean_np([d.get("window_prediction_min", float("nan")) for d in batch_diags]),
        "window_prediction_max": _nanmean_np([d.get("window_prediction_max", float("nan")) for d in batch_diags]),
        "window_y_true_frac_class0": _nanmean_np([d.get("y_true_frac_class0", float("nan")) for d in batch_diags]),
        "window_y_true_frac_class1": _nanmean_np([d.get("y_true_frac_class1", float("nan")) for d in batch_diags]),
        "window_y_true_frac_class2": _nanmean_np([d.get("y_true_frac_class2", float("nan")) for d in batch_diags]),
        "trajectory_rel_ms": rel_ms,
        "trajectory_by_position": traj_by_pos,
        "p_correct_ordering_ok": int(
            np.isfinite([r["p_correct_at_window_end"] for r in by_pos_rows]).all()
            and by_pos_rows[2]["p_correct_at_window_end"] > by_pos_rows[1]["p_correct_at_window_end"] > by_pos_rows[0]["p_correct_at_window_end"]
        ),
        "rt_ordering_ok": int(
            np.isfinite([r["first_crossing_060_ms"] for r in by_pos_rows]).all()
            and by_pos_rows[0]["first_crossing_060_ms"] > by_pos_rows[1]["first_crossing_060_ms"] > by_pos_rows[2]["first_crossing_060_ms"]
        ),
    }
    for thr_key, arr in crossing.items():
        lab = thr_key.replace(".", "")
        summary[f"first_crossing_{lab}_ms"] = _nanmedian_np(arr)
        summary[f"found_rate_{lab}"] = float(np.isfinite(arr).mean()) if arr.size > 0 else float("nan")
        idx_arr = crossing_idx[thr_key]
        pred = np.argmax(prob_end, axis=1).astype(np.int64, copy=False) if prob_end.size > 0 else np.array([], dtype=np.int64)
        if idx_arr.size > 0 and pred.size == idx_arr.size:
            acc_vec = np.zeros((idx_arr.shape[0],), dtype=np.float64)
            found_mask = idx_arr >= 0
            acc_vec[found_mask] = (pred[found_mask] == y_true_cls[found_mask]).astype(np.float64)
            summary[f"acc_at_found_{lab}"] = float(acc_vec.mean())
        else:
            summary[f"acc_at_found_{lab}"] = float("nan")
        summary[f"miss_rate_{lab}"] = float(np.isnan(arr).mean()) if arr.size > 0 else float("nan")
        summary[f"proportion_crossing_at_first_token_{lab}"] = float(np.nanmean(arr == 0.0)) if np.isfinite(arr).any() else float("nan")
        summary[f"median_rt_ms_{lab}"] = _nanmedian_np(arr)
        summary[f"std_rt_ms_{lab}"] = _nanstd_np(arr)
        summary[f"iqr_rt_ms_{lab}"] = _naniqr_np(arr)
    for pos_row in by_pos_rows:
        pos_label = int(pos_row["position"])
        summary[f"window_acc_P{pos_label}"] = float(pos_row["window_acc"])
        summary[f"window_f1_P{pos_label}"] = float(pos_row["window_f1_class"])
        summary[f"window_auc_P{pos_label}"] = float(pos_row["window_auc_class_ovr"])
        summary[f"window_p_correct_mean_P{pos_label}"] = float(pos_row["mean_p_correct_in_window"])
        summary[f"window_p_correct_end_P{pos_label}"] = float(pos_row["p_correct_at_window_end"])
        summary[f"window_token_loss_P{pos_label}"] = float(pos_row["window_token_loss"])
        for lab in ["050", "055", "060", "070", "090"]:
            summary[f"acc_at_found_{lab}_P{pos_label}"] = float(pos_row.get(f"acc_at_found_{lab}", float("nan")))
    return summary, by_pos_rows


def token_supervision_reference_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
    reference: str = "deviant_offset",  # "deviant_onset" | "deviant_offset"
) -> torch.Tensor:
    if reference == "deviant_onset":
        return deviant_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    if reference == "deviant_offset":
        return deviant_end_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)
    raise ValueError(f"Unknown token supervision reference: {reference}")


def make_post_deviant_window_mask_tokens(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_ms: int,
    window_ms: int,
    start_offset_ms: int = 0,
    reference: str = "deviant_offset",  # "deviant_onset" | "deviant_offset"
    include_anchor_token: bool = False,
) -> torch.Tensor:
    trial_id = (abs_t // trial_T_tokens).long()
    within = (abs_t % trial_T_tokens).long()

    ref_bt = token_supervision_reference_token_in_trial(
        y_pos_456=y_pos_456,
        tone_T=tone_T,
        isi_T=isi_T,
        reference=reference,
    )
    ref_for_t = ref_bt[:, trial_id]

    offset_T = max(0, int(round(start_offset_ms / token_ms)))
    win_T = max(1, int(round(window_ms / token_ms)))

    start = ref_for_t + offset_T + (0 if bool(include_anchor_token) else 1)
    end = start + win_T
    return (within >= start) & (within < end)


def make_deviant_to_next_standard_mask_tokens(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    trial_id = (abs_t // trial_T_tokens).long()
    within = (abs_t % trial_T_tokens).long()

    dev_on_bt = deviant_onset_token_in_trial(
        y_pos_456=y_pos_456,
        tone_T=tone_T,
        isi_T=isi_T,
    )
    next_std_on_bt = next_standard_onset_token_in_trial(
        y_pos_456=y_pos_456,
        tone_T=tone_T,
        isi_T=isi_T,
    )
    dev_on = dev_on_bt[:, trial_id]
    next_std_on = next_std_on_bt[:, trial_id]
    return (within >= dev_on) & (within < next_std_on)


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


# =========================================================================
# NEW: Response-head RT, readiness ranking loss, tone-position helpers
# =========================================================================

def compute_response_head_rt(
    response_logits: torch.Tensor,       # (B,10,Tt,1)
    dev_onset: torch.Tensor,             # (B,10)
    theta: float,
    earliest_tokens: int = 0,
    max_tokens: int = 600,
    token_ms: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic RT from response head: first token where sigmoid(logit) > theta.

    Returns:
      rt_ms: (B,10) float, RT from deviant onset in ms, NaN if no stop
      found: (B,10) bool
      decision_token: (B,10) long
    """
    B, N, Tt, _ = response_logits.shape
    device = response_logits.device
    p_respond = torch.sigmoid(response_logits.squeeze(-1))  # (B,10,Tt)

    rt_ms = torch.full((B, N), float("nan"), device=device)
    found = torch.zeros((B, N), dtype=torch.bool, device=device)
    decision_token = torch.full((B, N), -1, dtype=torch.long, device=device)

    for b in range(B):
        for n in range(N):
            on = int(dev_onset[b, n].item())
            first = on + int(earliest_tokens)
            last = min(on + int(max_tokens), Tt)
            for t in range(first, last):
                if p_respond[b, n, t] > theta:
                    rt_ms[b, n] = float((t - on) * int(token_ms))
                    found[b, n] = True
                    decision_token[b, n] = t
                    break
    return rt_ms, found, decision_token


def compute_readiness_ranking_loss(
    response_logits: torch.Tensor,       # (B,10,Tt,1)
    y_pos_456: torch.Tensor,             # (B,10) in {4,5,6}
    dev_onset: torch.Tensor,             # (B,10)
    margin: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Ordinal ranking loss: pre-deviant readiness P4 < P5 < P6.

    readiness = p_respond at the last token before deviant onset.
    Loss = max(0, margin - (R_P5 - R_P4)) + max(0, margin - (R_P6 - R_P5))
    """
    B, N, Tt, _ = response_logits.shape
    device = response_logits.device
    p_respond = torch.sigmoid(response_logits.squeeze(-1))  # (B,10,Tt)

    # Readiness: p_respond at token just before deviant onset
    readiness = torch.zeros(B, N, device=device)
    valid = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        for n in range(N):
            on = int(dev_onset[b, n].item())
            if on > 0:
                readiness[b, n] = p_respond[b, n, on - 1]
                valid[b, n] = True

    # Per-position mean readiness (only across valid trials in this batch)
    pos_mask = {4: (y_pos_456 == 4) & valid, 5: (y_pos_456 == 5) & valid, 6: (y_pos_456 == 6) & valid}
    mean_readiness = {}
    for p in [4, 5, 6]:
        if pos_mask[p].any():
            mean_readiness[p] = readiness[pos_mask[p]].mean()
        else:
            mean_readiness[p] = None

    loss = torch.tensor(0.0, device=device)
    details = {"R4": 0.0, "R5": 0.0, "R6": 0.0, "rank_loss": 0.0, "rank_applied": False}

    if (mean_readiness[4] is not None and mean_readiness[5] is not None and
        mean_readiness[6] is not None):
        details["R4"] = float(mean_readiness[4].item())
        details["R5"] = float(mean_readiness[5].item())
        details["R6"] = float(mean_readiness[6].item())
        loss_45 = torch.clamp(margin - (mean_readiness[5] - mean_readiness[4]), min=0)
        loss_56 = torch.clamp(margin - (mean_readiness[6] - mean_readiness[5]), min=0)
        loss = loss_45 + loss_56
        details["rank_loss"] = float(loss.item())
        details["rank_applied"] = True

    return loss, details


def compute_rt_ranking_loss(
    response_logits: torch.Tensor,       # (B,10,Tt,1)
    y_pos_456: torch.Tensor,             # (B,10) in {4,5,6}
    dev_onset: torch.Tensor,             # (B,10)
    earliest_tokens: int = 6,
    token_ms: int = 5,
    margin: float = 50.0,                # margin in ms
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Ordinal ranking loss on expected RT: RT_P4 > RT_P5 > RT_P6.

    Computes expected RT under the response-head stop policy for each trial,
    then averages per position and penalizes if RT_P4 <= RT_P5 or RT_P5 <= RT_P6.
    """
    B, N, Tt, _ = response_logits.shape
    device = response_logits.device
    p_respond = torch.sigmoid(response_logits.squeeze(-1))  # (B,10,Tt)

    # Eligibility mask from dev_onset + earliest_tokens (vectorized)
    t_idx = torch.arange(Tt, device=device).view(1, 1, Tt)
    first_tok = (dev_onset.unsqueeze(-1) + int(earliest_tokens))
    mask = (t_idx >= first_tok).float()
    p_resp = p_respond * mask

    # Expected stop time under first-stop distribution
    q = (1.0 - p_resp).clamp(1e-6, 1.0)
    survive = torch.cumprod(q, dim=-1)
    survive_s = torch.cat([torch.ones_like(survive[..., :1]), survive[..., :-1]], dim=-1)
    p_first_stop = p_resp * survive_s * mask

    tgrid = torch.arange(Tt, device=device).view(1, 1, Tt).float()
    rt_from_on = ((tgrid - dev_onset.unsqueeze(-1).float()) * float(token_ms)).clamp(min=0)
    expected_rt = (p_first_stop * rt_from_on).sum(dim=-1)  # (B, 10)
    # For trials with near-zero p_resp, expected_rt ≈ Tt*token_ms; clamp
    expected_rt = expected_rt + (1.0 - p_first_stop.sum(dim=-1)) * float(Tt * token_ms)

    # Per-position mean expected RT
    pos_mask = {4: y_pos_456 == 4, 5: y_pos_456 == 5, 6: y_pos_456 == 6}
    mean_rt = {}
    for p in [4, 5, 6]:
        if pos_mask[p].any():
            mean_rt[p] = expected_rt[pos_mask[p]].mean()
        else:
            mean_rt[p] = None

    loss = torch.tensor(0.0, device=device)
    details = {"RT4": 0.0, "RT5": 0.0, "RT6": 0.0, "rt_rank_loss": 0.0, "rt_rank_applied": False}

    if (mean_rt[4] is not None and mean_rt[5] is not None and mean_rt[6] is not None):
        details["RT4"] = float(mean_rt[4].item())
        details["RT5"] = float(mean_rt[5].item())
        details["RT6"] = float(mean_rt[6].item())
        # We want RT_P4 > RT_P5 > RT_P6
        loss_45 = torch.clamp(margin - (mean_rt[4] - mean_rt[5]), min=0)
        loss_56 = torch.clamp(margin - (mean_rt[5] - mean_rt[6]), min=0)
        loss = loss_45 + loss_56
        details["rt_rank_loss"] = float(loss.item())
        details["rt_rank_applied"] = True

    return loss, details


def make_tone_position_tensor(
    trial_T_tokens: int, tone_T: int, isi_T: int, n_tones: int = 8,
    batch_size: int = 1, n_trials: int = 10, device=None,
) -> torch.Tensor:
    """Create tone-position index tensor (0..7) for each token in each trial.

    Returns: (B, N*trial_T_tokens) or (1, N*trial_T_tokens) with values 0..7.
    ISI tokens get the same position as the preceding tone.
    """
    seq = []
    for tone_idx in range(n_tones):
        seq.extend([tone_idx] * tone_T)
        if tone_idx < n_tones - 1:
            seq.extend([tone_idx] * isi_T)
    trial_pos = torch.tensor(seq, dtype=torch.long, device=device)  # (trial_T_tokens,)
    block_pos = trial_pos.repeat(n_trials)  # (N*trial_T_tokens,)
    return block_pos.unsqueeze(0).expand(batch_size, -1)  # (B, N*trial_T_tokens)


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


def compute_strict_p4_decision_tokens_from_logits(
    logits: torch.Tensor,          # (B,10,trial_T_tokens,3)
    tone_T: int,
    isi_T: int,
    p_thresh: float,
    k_consec: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict-online diagnostic: first stable P4/P5/P6 decision from fixed P4."""
    B, N, Tt, C = logits.shape
    assert N == 10 and C >= 3
    probs = torch.softmax(logits[..., :3], dim=-1)
    pred = probs.argmax(dim=-1)
    pmax = probs.max(dim=-1).values
    p4 = int(3 * (int(tone_T) + int(isi_T)))
    decision_token = torch.full((B, N), -1, dtype=torch.long, device=logits.device)
    found = torch.zeros((B, N), dtype=torch.bool, device=logits.device)
    pred_at_rt = torch.full((B, N), -1, dtype=torch.long, device=logits.device)
    K = int(max(1, k_consec))
    if p4 >= Tt:
        return decision_token, found, pred_at_rt
    for b in range(B):
        for tr in range(N):
            for t in range(p4, Tt - K + 1):
                cls0 = int(pred[b, tr, t].item())
                ok = True
                for j in range(K):
                    if int(pred[b, tr, t + j].item()) != cls0 or float(pmax[b, tr, t + j].item()) < float(p_thresh):
                        ok = False
                        break
                if ok:
                    decision_token[b, tr] = int(t)
                    found[b, tr] = True
                    pred_at_rt[b, tr] = cls0
                    break
    return decision_token, found, pred_at_rt


def compute_strict_p4_argmin_cost_decision_tokens_from_logits(
    logits: torch.Tensor,          # (B,10,trial_T_tokens,C>=3)
    y_pos_456: torch.Tensor,       # (B,10)
    tone_T: int,
    isi_T: int,
    token_ms: int,
    time_cost_w: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict-online expected-cost argmin readout from fixed P4 to trial end."""
    B, N, Tt, C = logits.shape
    assert N == 10 and C >= 3
    probs = torch.softmax(logits, dim=-1)[..., :3]
    pred = probs.argmax(dim=-1)
    y_cls = labels_to_class_index(y_pos_456).to(logits.device).long()
    gather_idx = y_cls.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Tt, 1)
    p_correct = probs.gather(dim=-1, index=gather_idx).squeeze(-1)
    p4 = int(3 * (int(tone_T) + int(isi_T)))
    decision_token = torch.full((B, N), -1, dtype=torch.long, device=logits.device)
    found = torch.zeros((B, N), dtype=torch.bool, device=logits.device)
    pred_at_rt = torch.full((B, N), -1, dtype=torch.long, device=logits.device)
    if p4 >= Tt:
        return decision_token, found, pred_at_rt
    elapsed = torch.arange(Tt - p4, device=logits.device, dtype=p_correct.dtype) * float(token_ms)
    window_pc = p_correct[:, :, p4:Tt]
    expected_cost = (1.0 - window_pc) + window_pc * float(time_cost_w) * elapsed.view(1, 1, -1)
    finite = torch.isfinite(expected_cost)
    expected_cost = torch.where(finite, expected_cost, torch.full_like(expected_cost, float("inf")))
    idx = expected_cost.argmin(dim=-1)
    has_valid = finite.any(dim=-1)
    decision_token[has_valid] = idx[has_valid].long() + int(p4)
    found[has_valid] = True
    pred_at_rt[has_valid] = pred.gather(dim=-1, index=decision_token.clamp_min(0).unsqueeze(-1)).squeeze(-1)[has_valid]
    return decision_token, found, pred_at_rt


def make_online_eligibility_mask_tokens(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    start_mode: str = "deviant_start",   # trial_start | deviant_start | deviant_onset | deviant_end
    end_mode: str = "trial_end",         # trial_end | stimulus_end | sequence_end | next_tone_onset | next_tone_offset | second_next_tone_onset | second_next_tone_offset
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
    elif end_mode == "next_tone_onset":
        hi = next_tone_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)[:, trial_id] - 1
    elif end_mode == "next_tone_offset":
        hi = next_tone_offset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)[:, trial_id]
    elif end_mode == "second_next_tone_onset":
        hi = second_next_tone_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)[:, trial_id] - 1
    elif end_mode == "second_next_tone_offset":
        hi = second_next_tone_offset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)[:, trial_id]
    else:
        raise ValueError(f"Unknown online_supervision_end: {end_mode}")

    hi = torch.maximum(hi, lo)
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
    pre_devend_cost_weight: float = 0.0,
    pre_devend_cost_mode: str = "none",
    pre_devend_cost_margin_ms: float = 0.0,
    pre_devend_cost_scale_ms: float = 50.0,
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
    margin = float(pre_devend_cost_margin_ms)
    scale = max(1e-6, float(pre_devend_cost_scale_ms))
    rt_gap = (margin - logged_rt_ms)
    if str(pre_devend_cost_mode) == "none":
        pre_devend_cost_t = torch.zeros_like(decision_cost_t)
    elif str(pre_devend_cost_mode) == "flat":
        pre_devend_cost_t = (logged_rt_ms < margin).float()
    elif str(pre_devend_cost_mode) == "linear":
        pre_devend_cost_t = torch.clamp(rt_gap, min=0.0) / scale
    elif str(pre_devend_cost_mode) == "quadratic":
        pre_devend_cost_t = (torch.clamp(rt_gap, min=0.0) / scale) ** 2
    elif str(pre_devend_cost_mode) == "sigmoid_ramp":
        pre_devend_cost_t = torch.sigmoid(rt_gap / scale)
    else:
        raise ValueError(f"Unknown pre_devend_cost_mode: {pre_devend_cost_mode}")
    pre_devend_cost_t = pre_devend_cost_t * (logged_rt_ms < margin).float()
    decision_cost_t = decision_cost_t + float(pre_devend_cost_weight) * pre_devend_cost_t

    q = (1.0 - p_stop).clamp(1e-6, 1.0)
    prev_survive = torch.cumprod(q, dim=-1)
    prev_survive = torch.cat([torch.ones_like(prev_survive[..., :1]), prev_survive[..., :-1]], dim=-1)
    p_first_stop = (p_stop * prev_survive) * mask.float()
    p_no_response = torch.where(mask, q, torch.ones_like(q))
    p_no_response = torch.cumprod(p_no_response, dim=-1)[..., -1]

    expected = (p_first_stop * decision_cost_t).sum(dim=-1) + p_no_response * 1.0
    loss = expected.mean()
    pre_devend_cost_loss = (p_first_stop * pre_devend_cost_t).sum(dim=-1).mean()
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
        "pre_devend_cost_loss": float(pre_devend_cost_loss.item()),
        "weighted_pre_devend_cost_loss": float(float(pre_devend_cost_weight) * pre_devend_cost_loss.item()),
        "pre_devend_cost_mode": str(pre_devend_cost_mode),
        "pre_devend_cost_weight": float(pre_devend_cost_weight),
        "pre_devend_cost_margin_ms": float(pre_devend_cost_margin_ms),
        "pre_devend_cost_scale_ms": float(pre_devend_cost_scale_ms),
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
    token_loss_mode: str = "old",
    token_tau: float = 50.0,
    token_w_min: float = 0.05,
    token_supervision_reference: str = "deviant_offset",  # "deviant_onset" | "deviant_offset"
    include_anchor_token: bool = False,
    correct_ce_window: str = "deviant_onset_to_next_tone_offset",
    correct_ce_weighting: str = "equal",
    windowed_correct_ce_average: bool = False,
    anti_commit_window_ms: int = 0,
    anti_commit_start_offset_ms: int = 0,
    anti_commit_max_conf: float = 1.0,
    return_full_logits: bool = False,
    return_stop_logits: bool = False,
    return_hidden: bool = False,
    use_mask_loss: bool = False,      # 新增：是否使用掩码损失
    important_token_indices: Optional[List[int]] = None,  # 有意义token的位置
    supervision_mode: str = "post_deviant",
    block_context_training: bool = False,
    detach_hidden_between_trials: bool = False,
    detach_hidden_every_n_trials: int = 1,
    hidden_carryover_rho: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns:
        logits_end: (B,10,3)
        token_loss: scalar
        anti_commit_loss: scalar
        logits_all: (B,T,3) or None
        token_mask: (B,T) bool or None  # 新增：返回mask用于分析
    """
    normalized_token_loss_mode = _normalize_token_loss_mode(token_loss_mode)
    if str(correct_ce_weighting) != "equal":
        raise ValueError("correct_ce_weighting must be 'equal'")
    B, T, D = x.shape
    h = None
    collected_end_logits = []
    token_loss_sum = x.new_tensor(0.0)
    token_weight_sum = x.new_tensor(0.0)
    anti_commit_loss_sum = x.new_tensor(0.0)
    anti_commit_count = x.new_tensor(0.0)

    logits_chunks = [] if return_full_logits else None
    stop_logits_chunks = [] if (return_full_logits and return_stop_logits) else None
    hidden_chunks = [] if (return_full_logits and return_hidden) else None
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

    def _detach_hidden(hidden: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if hidden is None:
            return None
        return (hidden[0].detach(), hidden[1].detach()) if isinstance(hidden, tuple) else hidden.detach()

    def _scale_hidden(hidden: Optional[torch.Tensor], rho: float) -> Optional[torch.Tensor]:
        if hidden is None:
            return None
        if isinstance(hidden, tuple):
            return tuple(h * float(rho) for h in hidden)
        return hidden * float(rho)

    def _process_span(span_start: int, span_end: int, hidden_in: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        nonlocal token_loss_sum, token_weight_sum, anti_commit_loss_sum, anti_commit_count
        hidden = hidden_in
        for s in range(int(span_start), int(span_end), int(chunk_len)):
            e = min(s + int(chunk_len), int(span_end))
            x_in = x[:, s:e, :]
            h_seq, hidden = model.forward_chunk(x_in, h0=hidden)    # (B,L,H)
            logits = model.classify_tokens(h_seq)                   # (B,L,C)
            event_logits = model.classify_event(h_seq) if normalized_token_loss_mode == "event_deviance_ce" else None
            stop_logits = model.classify_stop(h_seq) if (return_full_logits and return_stop_logits) else None

            if return_full_logits:
                logits_chunks.append(logits.detach())
                if stop_logits_chunks is not None:
                    if stop_logits is None:
                        stop_logits_chunks.append(torch.full((B, e - s, 1), float("nan"), device=logits.device))
                    else:
                        stop_logits_chunks.append(stop_logits.detach())
                if hidden_chunks is not None:
                    hidden_chunks.append(h_seq.detach())
                if global_mask is not None:
                    mask_chunks.append(global_mask[:, s:e])

            abs_t = torch.arange(s, e, device=x.device)
            trial_id = (abs_t // int(trial_T_tokens)).long()

            event_target = None
            if normalized_token_loss_mode == "event_deviance_ce":
                if event_logits is None:
                    raise RuntimeError(
                        "token_loss_mode=event_deviance_ce requires ModelConfig.use_event_head=True "
                        "(pass --use_event_head)."
                    )
                mask, event_target = make_tone_event_targets_tokens(
                    abs_t=abs_t,
                    y_pos_456=y_pos_456,
                    trial_T_tokens=int(trial_T_tokens),
                    tone_T=int(tone_T),
                    isi_T=int(isi_T),
                )
                y_cls = labels_to_class_index(y_pos_456)
                target = y_cls[:, trial_id]
                target_probs = None
            elif str(supervision_mode) == "strict_online_p4":
                y_cls = labels_to_class_index(y_pos_456)
                target = y_cls[:, trial_id]
                target_probs = None
                if normalized_token_loss_mode == "strict_p4_causal_ce":
                    mask = make_strict_p4_post_offset_mask(
                        abs_t=abs_t,
                        y_pos_456=y_pos_456,
                        trial_T_tokens=int(trial_T_tokens),
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                    )
                    target_probs = make_strict_p4_causal_target_probs(
                        abs_t=abs_t,
                        y_pos_456=y_pos_456,
                        trial_T_tokens=int(trial_T_tokens),
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                    ).to(dtype=logits.dtype)
                else:
                    mask = make_strict_online_p4_mask(
                        abs_t=abs_t,
                        y_pos_456=y_pos_456,
                        trial_T_tokens=int(trial_T_tokens),
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                    )
            else:
                if use_mask_loss and global_mask is not None:
                    mask = global_mask[:, s:e]
                else:
                    if normalized_token_loss_mode == "old":
                        mask = make_deviant_to_next_standard_mask_tokens(
                            abs_t=abs_t,
                            y_pos_456=y_pos_456,
                            trial_T_tokens=int(trial_T_tokens),
                            tone_T=int(tone_T),
                            isi_T=int(isi_T),
                        )
                    else:
                        mask = make_correct_ce_window_mask_tokens(
                            abs_t=abs_t,
                            y_pos_456=y_pos_456,
                            trial_T_tokens=int(trial_T_tokens),
                            tone_T=int(tone_T),
                            isi_T=int(isi_T),
                            correct_ce_window=str(correct_ce_window),
                        )

                y_cls = labels_to_class_index(y_pos_456)
                target = y_cls[:, trial_id]
                target_probs = None

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
                    reference="deviant_offset",
                    include_anchor_token=False,
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
                if event_target is not None:
                    loss_per_token = F.binary_cross_entropy_with_logits(
                        event_logits.squeeze(-1),
                        event_target.to(dtype=event_logits.dtype),
                        reduction="none",
                    )
                elif target_probs is not None:
                    logp = F.log_softmax(logits, dim=-1)
                    loss_per_token = -(target_probs * logp[..., : target_probs.shape[-1]]).sum(dim=-1)
                else:
                    assert int(target.min().item()) >= 0
                    assert int(target.max().item()) < int(logits.shape[-1])
                    loss_per_token = F.cross_entropy(
                        logits.reshape(-1, int(logits.shape[-1])),
                        target.reshape(-1),
                        reduction="none",
                    ).reshape(target.shape)
                token_weights = mask.float()
                token_loss_sum = token_loss_sum + (loss_per_token * token_weights).sum()
                token_weight_sum = token_weight_sum + token_weights.sum()

            m_end = (end_idx >= s) & (end_idx < e)
            if m_end.any():
                rel = (end_idx[m_end] - s).long()
                h_end = h_seq.index_select(dim=1, index=rel)
                collected_end_logits.append(model.classify_from_states(h_end))

            hidden = _detach_hidden(hidden)
        return hidden

    detach_every = max(1, int(detach_hidden_every_n_trials))

    rho = float(hidden_carryover_rho)
    if bool(block_context_training) and not bool(detach_hidden_between_trials) and abs(rho - 1.0) < 1e-12:
        h = _process_span(0, T, h)
    else:
        n_trials = int(T // int(trial_T_tokens))
        if bool(block_context_training) and not bool(detach_hidden_between_trials):
            for trial in range(n_trials):
                trial_start = int(trial * int(trial_T_tokens))
                trial_end = int(min(T, trial_start + int(trial_T_tokens)))
                if trial > 0:
                    h = _scale_hidden(h, rho)
                h = _process_span(trial_start, trial_end, h)
        elif bool(block_context_training):
            for trial in range(0, n_trials, detach_every):
                trial_start = int(trial * int(trial_T_tokens))
                group_end_trial = min(n_trials, trial + detach_every)
                trial_end = int(min(T, group_end_trial * int(trial_T_tokens)))
                if trial > 0:
                    h = _scale_hidden(h, rho)
                h = _process_span(trial_start, trial_end, h)
                h = _detach_hidden(h)
        else:
            for trial in range(n_trials):
                trial_start = int(trial * int(trial_T_tokens))
                trial_end = int(min(T, trial_start + int(trial_T_tokens)))
                h = _process_span(trial_start, trial_end, None)
                h = None

    if len(collected_end_logits) == 0:
        raise RuntimeError("No trial-end states collected. Check end_idx and T.")

    logits_end = torch.cat(collected_end_logits, dim=1)
    online_valid_n = token_weight_sum.clamp_min(1.0)
    if normalized_token_loss_mode == "windowed_correct_ce":
        token_loss = (token_loss_sum / online_valid_n) if bool(windowed_correct_ce_average) else token_loss_sum
    elif normalized_token_loss_mode in ("strict_p4_causal_ce", "event_deviance_ce"):
        token_loss = token_loss_sum / online_valid_n
    else:
        token_loss = token_loss_sum
    anti_commit_loss = anti_commit_loss_sum / (anti_commit_count + 1e-8)
    setattr(
        model,
        "_last_token_loss_stats",
        {
            "token_loss_mode": str(token_loss_mode),
            "effective_token_loss_mode": str(normalized_token_loss_mode),
            "raw_token_ce_sum": float(token_loss_sum.detach().item()),
            "token_weight_sum": float(token_weight_sum.detach().item()),
            "token_loss_value": float(token_loss.detach().item()),
            "online_loss_sum": float(token_loss_sum.detach().item()),
            "online_loss_mean": float(token_loss.detach().item()),
            "online_valid_n": float(online_valid_n.detach().item()),
        },
    )

    logits_all = None
    full_mask = None
    if return_full_logits:
        logits_all = torch.cat(logits_chunks, dim=1)
        if mask_chunks:
            full_mask = torch.cat(mask_chunks, dim=1)

    stop_logits_all = None
    if stop_logits_chunks is not None:
        stop_logits_all = torch.cat(stop_logits_chunks, dim=1)

    h_seq_all = None
    if hidden_chunks is not None:
        h_seq_all = torch.cat(hidden_chunks, dim=1)

    return logits_end, token_loss, anti_commit_loss, logits_all, full_mask, stop_logits_all, h_seq_all


# -------------------------
# Metrics (acc, F1, AUC)
# -------------------------
def _collect_classification_metrics_from_logits(
    logits_end: torch.Tensor,   # (B,10,3)
    y_cls: torch.Tensor,        # (B,10)
) -> Dict[str, float]:
    with torch.no_grad():
        probs = torch.softmax(logits_end, dim=-1)  # (B,10,3)
        y_true = y_cls.reshape(-1).detach().cpu().numpy()
        y_prob = probs.reshape(-1, 3).detach().cpu().numpy()
        metrics = compute_multiclass_metrics_from_probs(y_true, y_prob, n_classes=3)
    return {"acc": float(metrics["acc"]), "f1_macro": float(metrics["f1_macro"]), "auc_ovr": float(metrics["auc_ovr"])}


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
    end_loss_weight: float,
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
    rt_k_consec: int = 1,
    rt_mode: str = "entropy",
    rt_entropy_thresh: float = 0.35,
    min_rt_tokens: int = 0,
    debug: bool = False,
    debug_steps: int = 0,
    log_every: int = 50,
    token_loss_mode: str = "old",
    token_tau: float = 50.0,
    token_w_min: float = 0.05,
    token_supervision_reference: str = "deviant_offset",
    include_anchor_token: bool = False,
    correct_ce_window: str = "deviant_onset_to_next_tone_offset",
    correct_ce_weighting: str = "equal",
    windowed_correct_ce_average: bool = False,
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
    pre_devend_cost_weight: float = 0.0,
    pre_devend_cost_mode: str = "none",
    pre_devend_cost_margin_ms: float = 0.0,
    pre_devend_cost_scale_ms: float = 50.0,
    stop_entropy_weight: float = 0.0,
    stop_prior_weight: float = 0.0,
    stop_prior_target: float = 0.05,
    pre_p4_uniformity_weight: float = 0.0,
    pre_evidence_uniform_kl_weight: float = 0.0,
    pre_evidence_uniform_kl_window: str = "trial_start_to_p4_onset",
    pre_p4_audit_enable: bool = False,
    lambda_online: Optional[float] = None,
    lambda_end: Optional[float] = None,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    grad_scaler: Optional[torch.amp.GradScaler] = None,

    debug_loss_check: bool = False,
    debug_overfit_tiny: bool = False,
    debug_first_batch_done: bool = False,
    optimizer_step_state: Optional[Dict[str, int]] = None,
    max_optimizer_steps: Optional[int] = None,
    step_checkpoint_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    supervision_mode: str = "post_deviant",
    compact_figure_dir: Optional[Path] = None,
    compact_figure_max_blocks: int = 2,
    block_context_training: bool = False,
    detach_hidden_between_trials: bool = False,
    detach_hidden_every_n_trials: int = 1,
    hidden_carryover_rho: float = 1.0,
    end_offset_from_trial_end: int = 0,
) -> dict:
    normalized_token_loss_mode = _normalize_token_loss_mode(token_loss_mode)
    model.train()
    ce_end = nn.CrossEntropyLoss(reduction="mean")
    ce_tok = nn.CrossEntropyLoss(reduction="mean")
    if lambda_online is None:
        lambda_online = float(lambda_token)
    if lambda_end is None:
        lambda_end = float(end_loss_weight)
    amp_dtype_t = _resolve_amp_dtype(str(amp_dtype))
    use_grad_scaler = bool(amp_enabled) and (grad_scaler is not None) and (str(device.type) == "cuda")

    total_end = 0.0
    total_tok = 0.0
    total_anti = 0.0
    n_examples = 0
    window_batch_diags: List[Dict[str, Any]] = []
    pre_p4_probability_audit_acc = _init_pre_p4_probability_audit_accumulator() if bool(pre_p4_audit_enable) else {}
    window_batch_diags: List[Dict[str, Any]] = []

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []

    rt_tokens_sum = 0.0
    rt_ms_sum = 0.0
    rt_n = 0
    rt_miss = 0
    rt_not_first = 0
    rt_negative = 0
    online_cost_sum = 0.0
    online_cost_n = 0
    online_loss_sum_diag = 0.0
    online_valid_n_diag = 0.0
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
    pre_p4_uniformity_sum = 0.0
    weighted_pre_p4_uniformity_sum = 0.0
    pre_evidence_uniform_kl_sum = 0.0
    weighted_pre_evidence_uniform_kl_sum = 0.0
    total_loss_sum = 0.0
    pre_devend_cost_sum = 0.0
    weighted_pre_devend_cost_sum = 0.0
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

        end_idx = infer_end_indices_from_T(T, trials_per_block=10, end_offset_from_trial_end=int(end_offset_from_trial_end)).to(device)

        if debug and (not first_batch_debug_printed):
            debug_print_train_window_trials(
                y_pos_456=y_cpu,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
                token_ms=int(token_ms),
                correct_ce_window=str(correct_ce_window),
                train_rt_diag_window=str(getattr(model, "_debug_train_rt_diag_window", "deviant_onset_to_next_tone_onset")),
                max_trials=int(getattr(model, "_debug_window_first_n_trials", 6)),
            )
            first_batch_debug_printed = True

        optimizer.zero_grad(set_to_none=True)

        w0 = None
        if debug and step <= int(debug_steps):
            w = next(model.parameters())
            w0 = w.detach().float().cpu().clone()

        with _make_amp_autocast(device=device, enabled=bool(amp_enabled), dtype=amp_dtype_t):
            logits_end, token_loss, anti_commit_loss, logits_all, token_mask, stop_logits_all, _h_discard = _run_block_through_tbptt(
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
                token_supervision_reference=str(token_supervision_reference),
                include_anchor_token=bool(include_anchor_token),
                correct_ce_window=str(correct_ce_window),
                correct_ce_weighting=str(correct_ce_weighting),
                windowed_correct_ce_average=bool(windowed_correct_ce_average),
                anti_commit_window_ms=int(anti_commit_window_ms),
                anti_commit_start_offset_ms=int(anti_commit_start_offset_ms),
                anti_commit_max_conf=float(anti_commit_max_conf),
                return_full_logits=True,
                return_stop_logits=bool(use_stop_head),
                use_mask_loss=use_mask_loss,
                important_token_indices=important_token_indices,
                supervision_mode=str(supervision_mode),
                block_context_training=bool(block_context_training),
                detach_hidden_between_trials=bool(detach_hidden_between_trials),
                detach_hidden_every_n_trials=int(detach_hidden_every_n_trials),
                hidden_carryover_rho=float(hidden_carryover_rho),
            )
            token_stats = getattr(model, "_last_token_loss_stats", {})
            online_loss_sum_diag += float(token_stats.get("online_loss_sum", 0.0))
            online_valid_n_diag += float(token_stats.get("online_valid_n", 0.0))

            y_cls = labels_to_class_index(y)
            assert int(y_cls.min().item()) >= 0
            assert int(y_cls.max().item()) <= 2
            end_loss_per = F.cross_entropy(
                logits_end.reshape(-1, int(logits_end.shape[-1])),
                y_cls.reshape(-1),
                reduction="none",
            ).reshape(y_cls.shape)
            if gap_weights_bt is None:
                end_loss = end_loss_per.mean()
            else:
                end_loss = (end_loss_per * gap_weights_bt).sum() / gap_weights_bt.sum().clamp_min(1e-8)
            if logits_all is not None:
                if compact_figure_dir is not None and step == 1:
                    fig_path = plot_compact_trial_probability_timeline(
                        logits_all=logits_all.detach(),
                        y_pos_456=y.detach(),
                        trial_T_tokens=int(trial_T_tokens),
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                        token_ms=int(token_ms),
                        out_dir=Path(compact_figure_dir),
                        epoch_global=int(epoch_global),
                        max_blocks=int(compact_figure_max_blocks),
                        time_cost_w=float(time_cost_w if time_cost_w is not None else 0.001),
                    )
                    if fig_path is not None:
                        setattr(model, "_last_compact_trial_timeline_path", str(fig_path))
                    agg_path = plot_aggregate_probability_by_position(
                        logits_all=logits_all.detach(),
                        y_pos_456=y.detach(),
                        trial_T_tokens=int(trial_T_tokens),
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                        token_ms=int(token_ms),
                        out_dir=Path(compact_figure_dir),
                    )
                    if agg_path is not None:
                        setattr(model, "_last_aggregate_probability_path", str(agg_path))
                window_batch_diags.append(
                    compute_window_level_diagnostics(
                        logits_all=logits_all.detach(),
                        y_pos_456=y.detach(),
                        trial_T_tokens=int(trial_T_tokens),
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                        token_ms=int(token_ms),
                        token_loss_mode=str(token_loss_mode),
                        correct_ce_window=str(correct_ce_window),
                    )
                )
                if bool(pre_p4_audit_enable):
                    _update_pre_p4_probability_audit_accumulator(
                        acc=pre_p4_probability_audit_acc,
                        logits_all=logits_all.detach(),
                        y_pos_456=y.detach(),
                        trial_T_tokens=int(trial_T_tokens),
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                        token_ms=int(token_ms),
                    )

            total_loss = (
                float(lambda_end) * end_loss
                + float(lambda_online) * token_loss
                + float(lambda_anti_commit) * anti_commit_loss
            )
        online_decision_loss = end_loss.new_tensor(0.0)
        online_ce_loss = end_loss.new_tensor(0.0)
        aux_token_ce_loss = end_loss.new_tensor(0.0)
        pre_p4_uniformity_loss = end_loss.new_tensor(0.0)
        pre_evidence_uniform_kl_loss = end_loss.new_tensor(0.0)
        anti_immediate_stop_loss = end_loss.new_tensor(0.0)
        pre_devend_stop_loss = end_loss.new_tensor(0.0)
        stop_entropy_mean = end_loss.new_tensor(0.0)
        stop_prior_loss = end_loss.new_tensor(0.0)
        if logits_all is not None:
            class_trial_raw = logits_all.view(B, 10, int(trial_T_tokens), 3)
            trial_local_t = torch.arange(int(trial_T_tokens), device=x.device)
            pre_p4_mask = make_pre_p4_uniform_mask_tokens(
                abs_t=trial_local_t,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
            ).view(1, 1, int(trial_T_tokens)).expand(B, 10, int(trial_T_tokens))
            pre_p4_uniformity_loss = compute_uniform_class_ce_loss(
                logits=class_trial_raw.reshape(-1, 3),
                mask=pre_p4_mask.reshape(-1),
            )
            if not torch.isfinite(pre_p4_uniformity_loss):
                pre_p4_uniformity_loss = end_loss.new_tensor(0.0)
            if float(pre_p4_uniformity_weight) != 0.0:
                total_loss = total_loss + float(pre_p4_uniformity_weight) * pre_p4_uniformity_loss
            pre_p4_uniformity_sum += float(pre_p4_uniformity_loss.item()) * B
            weighted_pre_p4_uniformity_sum += float(pre_p4_uniformity_weight) * float(pre_p4_uniformity_loss.item()) * B
            pre_evidence_mask = make_pre_evidence_uniform_mask_tokens(
                y_pos_456=y,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
                window=str(pre_evidence_uniform_kl_window),
            ).expand(B, 10, int(trial_T_tokens))
            pre_evidence_uniform_kl_loss = compute_uniform_class_kl_loss(
                logits=class_trial_raw.reshape(-1, 3),
                mask=pre_evidence_mask.reshape(-1),
            )
            if not torch.isfinite(pre_evidence_uniform_kl_loss):
                pre_evidence_uniform_kl_loss = end_loss.new_tensor(0.0)
            if float(pre_evidence_uniform_kl_weight) != 0.0:
                total_loss = total_loss + float(pre_evidence_uniform_kl_weight) * pre_evidence_uniform_kl_loss
            pre_evidence_uniform_kl_sum += float(pre_evidence_uniform_kl_loss.item()) * B
            weighted_pre_evidence_uniform_kl_sum += float(pre_evidence_uniform_kl_weight) * float(pre_evidence_uniform_kl_loss.item()) * B
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
                    pre_devend_cost_weight=float(pre_devend_cost_weight),
                    pre_devend_cost_mode=str(pre_devend_cost_mode),
                    pre_devend_cost_margin_ms=float(pre_devend_cost_margin_ms),
                    pre_devend_cost_scale_ms=float(pre_devend_cost_scale_ms),
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
                        pre_devend_stop_loss = p_stop[before_devend_mask].max()
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
            if "pre_devend_cost_loss" in od_stats and np.isfinite(od_stats["pre_devend_cost_loss"]):
                pre_devend_cost_sum += float(od_stats["pre_devend_cost_loss"]) * B
                weighted_pre_devend_cost_sum += float(od_stats.get("weighted_pre_devend_cost_loss", 0.0)) * B

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
        weighted_end_loss = float(lambda_end) * float(end_loss.detach().item())
        weighted_token = float(lambda_online) * float(token_loss.detach().item())
        token_loss_stats = getattr(model, "_last_token_loss_stats", {})
        raw_token_ce_sum = float(token_loss_stats.get("raw_token_ce_sum", float("nan")))
        token_weight_sum_dbg = float(token_loss_stats.get("token_weight_sum", float("nan")))
        token_loss_value_dbg = float(token_loss_stats.get("token_loss_value", float(token_loss.detach().item())))
        weighted_anti_commit = float(lambda_anti_commit) * float(anti_commit_loss.detach().item())
        weighted_online_decision = float(online_loss_weight) * float(online_warmup_factor) * float(online_decision_loss.detach().item())
        weighted_online_ce = float(online_ce_weight) * float(online_ce_loss.detach().item())
        weighted_aux_token_ce = float(aux_token_ce_weight) * float(aux_token_ce_loss.detach().item())
        weighted_pre_p4_uniformity = float(pre_p4_uniformity_weight) * float(pre_p4_uniformity_loss.detach().item())
        weighted_pre_evidence_uniform_kl = float(pre_evidence_uniform_kl_weight) * float(pre_evidence_uniform_kl_loss.detach().item())
        weighted_anti_immediate = float(anti_immediate_stop_weight) * float(anti_immediate_stop_loss.detach().item())
        weighted_pre_devend = float(pre_devend_stop_weight) * float(pre_devend_stop_loss.detach().item())
        weighted_pre_devend_cost = float(od_stats.get("weighted_pre_devend_cost_loss", 0.0)) if 'od_stats' in locals() else 0.0
        weighted_online_decision_base = weighted_online_decision - weighted_pre_devend_cost
        weighted_stop_prior = float(stop_prior_weight) * float(stop_prior_loss.detach().item())
        weighted_neg_stop_entropy = -float(stop_entropy_weight) * float(stop_entropy_mean.detach().item())
        reconstructed_loss = (
            weighted_end_loss
            + weighted_token
            + weighted_anti_commit
            + weighted_online_decision_base
            + weighted_online_ce
            + weighted_aux_token_ce
            + weighted_pre_p4_uniformity
            + weighted_pre_evidence_uniform_kl
            + weighted_anti_immediate
            + weighted_pre_devend
            + weighted_pre_devend_cost
            + weighted_stop_prior
            + weighted_neg_stop_entropy
        )
        full_loss_check_key = f"full_loss_check_done::{phase_name}"
        if optimizer_step_state is not None and not optimizer_step_state.get(full_loss_check_key, False):
            diff = abs(actual_backward_loss - reconstructed_loss)
            end_loss_scalar = float(end_loss.detach().item())
            token_to_end_ratio = weighted_token / max(end_loss_scalar, 1e-8)
            print("[FULL LOSS CHECK]")
            print("actual_backward_loss", actual_backward_loss)
            print("end_loss", end_loss_scalar)
            print("end_loss_weight", float(end_loss_weight))
            print("lambda_end * end_loss", weighted_end_loss)
            print("raw_token_ce_sum", raw_token_ce_sum)
            print("token_weight_sum", token_weight_sum_dbg)
            print("token_loss", token_loss_value_dbg)
            print("lambda_token", float(lambda_token))
            print("lambda_online * online_loss_mean", weighted_token)
            print("ratio = lambda_online * online_loss_mean / end_loss", token_to_end_ratio)
            print("lambda_anti_commit * anti_commit_loss", weighted_anti_commit)
            print("online_loss_weight * stochastic_expected_stop_loss", weighted_online_decision_base)
            print("online_ce_weight * online_ce_loss", weighted_online_ce)
            print("aux_token_ce_weight * aux_token_ce_loss", weighted_aux_token_ce)
            print("pre_p4_uniformity_weight * pre_p4_uniformity_loss", weighted_pre_p4_uniformity)
            print("pre_evidence_uniform_kl_weight * pre_evidence_uniform_kl_loss", weighted_pre_evidence_uniform_kl)
            print("anti_immediate_stop_weight * anti_immediate_stop_loss", weighted_anti_immediate)
            print("pre_devend_stop_weight * pre_devend_stop_loss", weighted_pre_devend)
            print("pre_devend_cost_weight * pre_devend_cost_loss", weighted_pre_devend_cost)
            print("stop_prior_weight * stop_prior_loss", weighted_stop_prior)
            print("- stop_entropy_weight * stop_entropy_mean", weighted_neg_stop_entropy)
            print("reconstructed_loss", reconstructed_loss)
            print("diff", diff)
            if (
                str(token_loss_stats.get("effective_token_loss_mode", "")) == "windowed_correct_ce"
                and token_loss_value_dbg > 10.0
            ):
                print(
                    "[WARN token_loss_scale] "
                    f"windowed_correct_ce token_loss={token_loss_value_dbg:.6f} > 10. "
                    "This is expected when token_loss is using the raw sum of supervised token CE."
                )
            optimizer_step_state[full_loss_check_key] = True
        if debug_loss_check:
            diff = abs(actual_backward_loss - reconstructed_loss)
            print("[DEBUG loss tensor]")
            print("actual_backward_loss", actual_backward_loss)
            print("reconstructed_loss", reconstructed_loss)
            print("diff", diff)
            print("end_loss", float(end_loss.detach().item()))
            print("end_loss_weight", float(end_loss_weight))
            print("lambda_end * end_loss", weighted_end_loss)
            print("token_loss", token_loss_value_dbg)
            print("lambda_token", float(lambda_token))
            print("lambda_online * online_loss_mean", weighted_token)
            print("lambda_anti_commit * anti_commit_loss", weighted_anti_commit)
            print("online_loss_weight * online_decision_loss", weighted_online_decision_base)
            print("online_ce_weight * online_ce_loss", weighted_online_ce)
            print("aux_token_ce_weight * aux_token_ce_loss", weighted_aux_token_ce)
            print("pre_p4_uniformity_weight * pre_p4_uniformity_loss", weighted_pre_p4_uniformity)
            print("pre_evidence_uniform_kl_weight * pre_evidence_uniform_kl_loss", weighted_pre_evidence_uniform_kl)
            print("anti_immediate_stop_weight * anti_immediate_stop_loss", weighted_anti_immediate)
            print("pre_devend_stop_weight * pre_devend_stop_loss", weighted_pre_devend)
            print("pre_devend_cost_weight * pre_devend_cost_loss", weighted_pre_devend_cost)
            print("stop_prior_weight * stop_prior_loss", weighted_stop_prior)
            print("- stop_entropy_weight * stop_entropy_mean", weighted_neg_stop_entropy)
            assert diff < 1e-5, f"Loss mismatch: actual={actual_backward_loss} reconstructed={reconstructed_loss} diff={diff}"
        if use_grad_scaler:
            grad_scaler.scale(total_loss).backward()
            grad_scaler.unscale_(optimizer)
        else:
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

        if use_grad_scaler:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()
        use_raw_windowed_ce = (
            str(normalized_token_loss_mode) == "windowed_correct_ce"
            and (not bool(windowed_correct_ce_average))
        )
        total_loss_sum += actual_backward_loss if use_raw_windowed_ce else (actual_backward_loss * B)
        if optimizer_step_state is not None:
            optimizer_step_state["count"] = int(optimizer_step_state.get("count", 0)) + 1
            global_opt_step = int(optimizer_step_state["count"])
        else:
            global_opt_step = step

        if step_checkpoint_callback is not None:
            step_checkpoint_callback(
                {
                    "epoch_global": int(epoch_global),
                    "global_opt_step": int(global_opt_step),
                    "step_in_epoch": int(step),
                    "phase_name": str(phase_name),
                    "isi_ms": int(isi_T) * int(token_ms),
                    "end_loss": float(end_loss.detach().item()),
                    "token_loss": float(token_loss.detach().item()),
                    "anti_commit_loss": float(anti_commit_loss.detach().item()),
                    "total_loss": float(actual_backward_loss),
                    "end_acc": float((torch.softmax(logits_end, dim=-1).argmax(dim=-1) == y_cls).float().mean().item()),
                    "model": model,
                    "optimizer": optimizer,
                }
            )

        if optimizer_step_state is not None and bool(optimizer_step_state.get("request_stop", False)):
            break

        if gru_sq > 0.0:
            grad_norm_gru = gru_sq ** 0.5
        if class_sq > 0.0:
            grad_norm_class_head = class_sq ** 0.5

        total_end += float(end_loss.item()) * B
        total_tok += float(token_stats.get("online_loss_sum", token_loss.item())) if use_raw_windowed_ce else (float(token_loss.item()) * B)
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
                if str(supervision_mode) == "strict_online_p4":
                    decision_tokens_cpu, found_cpu, pred_at_rt_cpu = compute_strict_p4_argmin_cost_decision_tokens_from_logits(
                        logits=logits_trial,
                        y_pos_456=y_cpu_rt,
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                        token_ms=int(token_ms),
                        time_cost_w=float(time_cost_w if time_cost_w is not None else 0.001),
                    )
                    dev_on_cpu = deviant_onset_token_in_trial(y_cpu_rt, tone_T=int(tone_T), isi_T=int(isi_T))
                    rt_tokens_cpu = decision_tokens_cpu - dev_on_cpu
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
                    rt_negative += int((rt_tokens_cpu[found_cpu] < 0).sum().item())

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
                f"end_acc={batch_acc:.3f} logits_mean={logits_end.detach().float().mean().item():.4f}"
            )

        if debug_overfit_tiny and (global_opt_step == 1 or global_opt_step % 25 == 0):
            train_acc = float((torch.softmax(logits_end, dim=-1).argmax(dim=-1) == y_cls).float().mean().item())
            print(
                f"[debug_overfit step {global_opt_step}] "
                f"loss={actual_backward_loss:.6f} end_loss={float(end_loss.detach().item()):.6f} "
                f"train_end_acc={train_acc:.4f} grad_norm_gru={grad_norm_gru:.6f} "
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
                    if str(supervision_mode) == "strict_online_p4":
                        decision_tokens_cpu, found_cpu, pred_at_rt_cpu = compute_strict_p4_argmin_cost_decision_tokens_from_logits(
                            logits=logits_trial,
                            y_pos_456=y_cpu_rt,
                            tone_T=int(tone_T),
                            isi_T=int(isi_T),
                            token_ms=int(token_ms),
                            time_cost_w=float(time_cost_w if time_cost_w is not None else 0.001),
                        )
                        dev_on_cpu = deviant_onset_token_in_trial(y_cpu_rt, tone_T=int(tone_T), isi_T=int(isi_T))
                        rt_tokens_cpu = decision_tokens_cpu - dev_on_cpu
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
                        batch_mean_rt_tokens = float(rt_tokens_cpu[found_cpu].float().mean().item())
                        batch_mean_rt_ms = batch_mean_rt_tokens * float(token_ms)
                        batch_rt_msg = f"batch_meanRT={batch_mean_rt_tokens:.2f}tok/{batch_mean_rt_ms:.1f}ms"

            mask_info = ""
            if use_mask_loss and token_mask is not None:
                mask_ratio = token_mask.float().mean().item()
                mask_info = f" mask_ratio={mask_ratio:.3f}"

            if float(lambda_end) == 0.0:
                print(
                    f"[train step {step:>4d}/{n_steps}] "
                    f"dt={dt:.2f}s ema={ema_step:.2f}s "
                    f"ETA_epoch={_fmt_hms(eta_epoch)} elapsed={_fmt_hms(elapsed)} "
                    f"online={float(token_loss.item()):.4f} "
                    f"anti={float(anti_commit_loss.item()):.4f} "
                    f"aux_end={end_loss.item():.4f} "
                    f"{batch_rt_msg}{mask_info}"
                )
            else:
                print(
                    f"[train step {step:>4d}/{n_steps}] "
                    f"dt={dt:.2f}s ema={ema_step:.2f}s "
                    f"ETA_epoch={_fmt_hms(eta_epoch)} elapsed={_fmt_hms(elapsed)} "
                    f"online={float(token_loss.item()):.4f} aux_end={end_loss.item():.4f} "
                    f"anti={float(anti_commit_loss.item()):.4f} "
                    f"end_acc={batch_acc:.3f} {batch_rt_msg}{mask_info}"
                )

        if max_optimizer_steps is not None and global_opt_step >= int(max_optimizer_steps):
            break

    denom = max(1, n_examples)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.zeros((0, 3), dtype=np.float32)

    mean_rt_tokens = (rt_tokens_sum / rt_n) if rt_n > 0 else float("nan")
    mean_rt_ms = (rt_ms_sum / rt_n) if rt_n > 0 else float("nan")
    rt_not_first_rate = (rt_not_first / rt_n) if rt_n > 0 else float("nan")
    window_diag, window_diag_by_pos = finalize_window_epoch_diagnostics(window_batch_diags, token_ms=int(token_ms))
    pre_p4_probability_audit_rows = finalize_pre_p4_probability_audit(pre_p4_probability_audit_acc)
    if debug and pre_p4_probability_audit_rows:
        labels_dbg = sorted({str(r.get("timepoint_label")) for r in pre_p4_probability_audit_rows})
        print(
            f"[pre_p4_prob debug] n_rows={len(pre_p4_probability_audit_rows)} "
            f"labels={labels_dbg} first_row={json.dumps(_serialize_jsonable(pre_p4_probability_audit_rows[0]), ensure_ascii=False)}"
        )
    elif debug:
        print("[pre_p4_prob debug] n_rows=0")
    end_acc = float((y_true == y_pred).mean()) if y_true.size > 0 else float("nan")
    end_f1 = safe_f1_macro(y_true, y_pred) if y_true.size > 0 else float("nan")
    end_auc = safe_auc_ovr(y_true, y_prob, n_classes=3) if y_true.size > 0 else float("nan")
    window_acc = float(window_diag.get("window_acc", float("nan")))
    window_f1 = float(window_diag.get("window_f1", float("nan")))
    window_auc = float(window_diag.get("window_auc", float("nan")))

    reported_token_loss = total_tok if (
        str(normalized_token_loss_mode) == "windowed_correct_ce" and (not bool(windowed_correct_ce_average))
    ) else (total_tok / denom)
    reported_total_loss = total_loss_sum if (
        str(normalized_token_loss_mode) == "windowed_correct_ce" and (not bool(windowed_correct_ce_average))
    ) else (total_loss_sum / denom)

    return {
        "end_loss": total_end / denom,
        "weighted_end_loss": float(lambda_end) * (total_end / denom),
        "token_loss": reported_token_loss,
        "online_loss_sum": float(online_loss_sum_diag),
        "online_valid_n": float(online_valid_n_diag),
        "online_loss_mean": float(online_loss_sum_diag / max(1.0, online_valid_n_diag)),
        "old_token_loss": (total_tok / denom) if normalized_token_loss_mode == "old" else float("nan"),
        "windowed_correct_ce_loss": reported_token_loss if normalized_token_loss_mode == "windowed_correct_ce" else float("nan"),
        "strict_p4_causal_ce_loss": (total_tok / denom) if normalized_token_loss_mode == "strict_p4_causal_ce" else float("nan"),
        "event_deviance_ce_loss": (total_tok / denom) if normalized_token_loss_mode == "event_deviance_ce" else float("nan"),
        "anti_commit_loss": total_anti / denom,
        "total_loss": reported_total_loss,
        "acc": window_acc,
        "f1_macro": window_f1,
        "auc_ovr": window_auc,
        "window_acc": window_acc,
        "window_f1": window_f1,
        "window_auc": window_auc,
        "window_acc_first_token": float(window_diag.get("window_acc_first_token", float("nan"))),
        "window_acc_last_token": float(window_diag.get("window_acc_last_token", float("nan"))),
        "window_acc_mean_token": float(window_diag.get("window_acc_mean_token", float("nan"))),
        "window_auc_mean_token": float(window_diag.get("window_auc_mean_token", float("nan"))),
        "window_p_correct_mean": float(window_diag.get("window_p_correct_mean", float("nan"))),
        "window_p_correct_std": float(window_diag.get("window_p_correct_std", float("nan"))),
        "window_p_correct_p10": float(window_diag.get("window_p_correct_p10", float("nan"))),
        "window_p_correct_p50": float(window_diag.get("window_p_correct_p50", float("nan"))),
        "window_p_correct_p90": float(window_diag.get("window_p_correct_p90", float("nan"))),
        "early_p_correct_mean": float(window_diag.get("early_p_correct_mean", float("nan"))),
        "window_p_correct_end": float(window_diag.get("window_p_correct_end", float("nan"))),
        "window_p_correct_max": float(window_diag.get("window_p_correct_max", float("nan"))),
        "window_p_correct_auc": float(window_diag.get("window_p_correct_auc", float("nan"))),
        "window_prediction_mean": float(window_diag.get("window_prediction_mean", float("nan"))),
        "window_prediction_std": float(window_diag.get("window_prediction_std", float("nan"))),
        "window_prediction_min": float(window_diag.get("window_prediction_min", float("nan"))),
        "window_prediction_max": float(window_diag.get("window_prediction_max", float("nan"))),
        "window_y_true_frac_class0": float(window_diag.get("window_y_true_frac_class0", float("nan"))),
        "window_y_true_frac_class1": float(window_diag.get("window_y_true_frac_class1", float("nan"))),
        "window_y_true_frac_class2": float(window_diag.get("window_y_true_frac_class2", float("nan"))),
        "end_acc": end_acc,
        "end_f1": float(end_f1),
        "end_auc": float(end_auc),
        "mean_rt_tokens": float(mean_rt_tokens),
        "mean_rt_ms": float(mean_rt_ms),
        "rt_found": int(rt_n),
        "rt_miss": int(rt_miss),
        "rt_not_first": int(rt_not_first),
        "rt_negative": int(rt_negative),
        "rt_not_first_rate": float(rt_not_first_rate),
        "window_diagnostics": window_diag,
        "window_diagnostics_by_position": window_diag_by_pos,
        "pre_p4_probability_audit_rows": pre_p4_probability_audit_rows,
        "online_decision_cost": float(online_cost_sum / max(1, online_cost_n)) if online_cost_n > 0 else float("nan"),
        "online_decision_loss": float(online_loss_sum / denom) if online_decision_training else float("nan"),
        "online_ce_loss": float(online_ce_sum / denom) if online_decision_training else float("nan"),
        "aux_token_ce_loss": float(aux_token_ce_sum / denom) if aux_token_ce_sum > 0 else 0.0,
        "pre_p4_uniformity_loss": float(pre_p4_uniformity_sum / denom) if pre_p4_uniformity_sum > 0 else 0.0,
        "pre_evidence_uniform_kl_loss": float(pre_evidence_uniform_kl_sum / denom) if pre_evidence_uniform_kl_sum > 0 else 0.0,
        "phase_name": str(phase_name),
        "anti_immediate_stop_loss": float(anti_immediate_stop_sum / denom) if anti_immediate_stop_sum > 0 else 0.0,
        "stop_entropy_bonus": float(stop_entropy_sum / denom) if stop_entropy_sum > 0 else 0.0,
        "stop_prior_loss": float(stop_prior_sum / denom) if stop_prior_sum > 0 else 0.0,
        "lambda_online": float(lambda_online),
        "lambda_end_current": float(lambda_end),
        "effective_online_loss_weight": float(online_loss_weight) * float(online_warmup_factor) if include_online_decision_loss else 0.0,
        "weighted_token_loss": float(lambda_online) * float(reported_token_loss),
        "weighted_online_decision_loss": (float(online_loss_weight) * float(online_warmup_factor) * (online_loss_sum / denom)) if include_online_decision_loss and denom > 0 else 0.0,
        "weighted_online_ce_loss": (float(online_ce_weight) * (online_ce_sum / denom)) if include_online_ce_loss and denom > 0 else 0.0,
        "weighted_aux_token_ce_loss": float(weighted_aux_token_ce_sum / denom) if weighted_aux_token_ce_sum > 0 else 0.0,
        "weighted_pre_p4_uniformity_loss": float(weighted_pre_p4_uniformity_sum / denom) if weighted_pre_p4_uniformity_sum > 0 else 0.0,
        "weighted_pre_evidence_uniform_kl_loss": float(weighted_pre_evidence_uniform_kl_sum / denom) if weighted_pre_evidence_uniform_kl_sum > 0 else 0.0,
        "preEvidenceUniformKlWindow": str(pre_evidence_uniform_kl_window),
        "preDevCostRaw": float(pre_devend_cost_sum / denom) if pre_devend_cost_sum > 0 else 0.0,
        "wPreDevCost": float(weighted_pre_devend_cost_sum / denom) if weighted_pre_devend_cost_sum > 0 else 0.0,
        "preDevCostMode": str(pre_devend_cost_mode),
        "preDevCostWeight": float(pre_devend_cost_weight),
        "preDevCostMargin": float(pre_devend_cost_margin_ms),
        "preDevCostScale": float(pre_devend_cost_scale_ms),
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
    end_loss_weight: float,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_loss_mode: str,
    token_tau: float,
    token_w_min: float,
    token_ms: int,
    tok_window_ms: int,
    tok_start_offset_ms: int,
    token_supervision_reference: str,
    include_anchor_token: bool,
    correct_ce_window: str,
    correct_ce_weighting: str,
    windowed_correct_ce_average: bool = False,
    lambda_anti_commit: float = 0.0,
    anti_commit_window_ms: int = 0,
    anti_commit_start_offset_ms: int = 0,
    anti_commit_max_conf: float = 1.0,
    rt_p_thresh: float = 0.7,
    rt_k_consec: int = 1,
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
    pre_devend_cost_weight: float = 0.0,
    pre_devend_cost_mode: str = "none",
    pre_devend_cost_margin_ms: float = 0.0,
    pre_devend_cost_scale_ms: float = 50.0,
    stop_entropy_weight: float = 0.0,
    stop_prior_weight: float = 0.0,
    stop_prior_target: float = 0.05,
    pre_p4_uniformity_weight: float = 0.0,
    pre_evidence_uniform_kl_weight: float = 0.0,
    pre_evidence_uniform_kl_window: str = "trial_start_to_p4_onset",
    pre_p4_audit_enable: bool = False,
    lambda_online: Optional[float] = None,
    lambda_end: Optional[float] = None,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    supervision_mode: str = "post_deviant",
    block_context_training: bool = False,
    detach_hidden_between_trials: bool = False,
    detach_hidden_every_n_trials: int = 1,
    hidden_carryover_rho: float = 1.0,
    end_offset_from_trial_end: int = 0,
) -> dict:
    normalized_token_loss_mode = _normalize_token_loss_mode(token_loss_mode)
    model.eval()
    ce_end = nn.CrossEntropyLoss(reduction="mean")
    ce_tok = nn.CrossEntropyLoss(reduction="mean")
    if lambda_online is None:
        lambda_online = float(lambda_token)
    if lambda_end is None:
        lambda_end = float(end_loss_weight)
    amp_dtype_t = _resolve_amp_dtype(str(amp_dtype))

    total_end = 0.0
    total_tok = 0.0
    total_anti = 0.0
    n_examples = 0
    window_batch_diags: List[Dict[str, Any]] = []

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []

    rt_tokens_sum = 0.0
    rt_ms_sum = 0.0
    rt_n = 0
    rt_miss = 0
    rt_not_first = 0
    rt_negative = 0
    online_cost_sum = 0.0
    online_cost_n = 0
    online_loss_sum_diag = 0.0
    online_valid_n_diag = 0.0
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
    pre_p4_uniformity_sum = 0.0
    weighted_pre_p4_uniformity_sum = 0.0
    pre_evidence_uniform_kl_sum = 0.0
    weighted_pre_evidence_uniform_kl_sum = 0.0
    total_loss_sum = 0.0
    pre_devend_cost_sum = 0.0
    weighted_pre_devend_cost_sum = 0.0
    last_token_acc_sum = 0.0
    last_token_acc_n = 0
    devend_acc_sum = 0.0
    devend_acc_n = 0
    pre_p4_probability_audit_acc = _init_pre_p4_probability_audit_accumulator()

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

        end_idx = infer_end_indices_from_T(T, trials_per_block=10, end_offset_from_trial_end=int(end_offset_from_trial_end)).to(device)

        with _make_amp_autocast(device=device, enabled=bool(amp_enabled), dtype=amp_dtype_t):
            logits_end, token_loss, anti_commit_loss, logits_all, _, stop_logits_all, _h_d2 = _run_block_through_tbptt(
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
                token_supervision_reference=str(token_supervision_reference),
                include_anchor_token=bool(include_anchor_token),
                correct_ce_window=str(correct_ce_window),
                correct_ce_weighting=str(correct_ce_weighting),
                windowed_correct_ce_average=bool(windowed_correct_ce_average),
                anti_commit_window_ms=int(anti_commit_window_ms),
                anti_commit_start_offset_ms=int(anti_commit_start_offset_ms),
                anti_commit_max_conf=float(anti_commit_max_conf),
                return_full_logits=True,
                return_stop_logits=bool(use_stop_head),
                use_mask_loss=use_mask_loss,
                important_token_indices=important_token_indices,
                supervision_mode=str(supervision_mode),
                block_context_training=bool(block_context_training),
                detach_hidden_between_trials=bool(detach_hidden_between_trials),
                detach_hidden_every_n_trials=int(detach_hidden_every_n_trials),
                hidden_carryover_rho=float(hidden_carryover_rho),
            )
            token_stats = getattr(model, "_last_token_loss_stats", {})
            online_loss_sum_diag += float(token_stats.get("online_loss_sum", 0.0))
            online_valid_n_diag += float(token_stats.get("online_valid_n", 0.0))

            y_cls = labels_to_class_index(y)
            assert int(y_cls.min().item()) >= 0
            assert int(y_cls.max().item()) <= 2
            end_loss_per = F.cross_entropy(
                logits_end.reshape(-1, int(logits_end.shape[-1])),
                y_cls.reshape(-1),
                reduction="none",
            ).reshape(y_cls.shape)
            if gap_weights_bt is None:
                end_loss = end_loss_per.mean()
            else:
                end_loss = (end_loss_per * gap_weights_bt).sum() / gap_weights_bt.sum().clamp_min(1e-8)
            if logits_all is not None:
                window_batch_diags.append(
                    compute_window_level_diagnostics(
                        logits_all=logits_all.detach(),
                        y_pos_456=y.detach(),
                        trial_T_tokens=int(trial_T_tokens),
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                        token_ms=int(token_ms),
                        token_loss_mode=str(token_loss_mode),
                        correct_ce_window=str(correct_ce_window),
                    )
                )

            use_raw_windowed_ce = (
                str(normalized_token_loss_mode) == "windowed_correct_ce"
                and (not bool(windowed_correct_ce_average))
            )
            total_end += float(end_loss.item()) * B
            total_tok += float(token_stats.get("online_loss_sum", token_loss.item())) if use_raw_windowed_ce else (float(token_loss.item()) * B)
            total_anti += float(anti_commit_loss.item()) * B
            total_loss = (
                float(lambda_end) * end_loss
                + float(lambda_online) * token_loss
                + float(lambda_anti_commit) * anti_commit_loss
            )
        pre_p4_uniformity_loss = end_loss.new_tensor(0.0)
        pre_evidence_uniform_kl_loss = end_loss.new_tensor(0.0)
        if logits_all is not None:
            class_trial_raw = logits_all.view(B, 10, int(trial_T_tokens), 3)
            trial_local_t = torch.arange(int(trial_T_tokens), device=x.device)
            pre_p4_mask = make_pre_p4_uniform_mask_tokens(
                abs_t=trial_local_t,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
            ).view(1, 1, int(trial_T_tokens)).expand(B, 10, int(trial_T_tokens))
            pre_p4_uniformity_loss = compute_uniform_class_ce_loss(
                logits=class_trial_raw.reshape(-1, 3),
                mask=pre_p4_mask.reshape(-1),
            )
            if not torch.isfinite(pre_p4_uniformity_loss):
                pre_p4_uniformity_loss = end_loss.new_tensor(0.0)
            if float(pre_p4_uniformity_weight) != 0.0:
                total_loss = total_loss + float(pre_p4_uniformity_weight) * pre_p4_uniformity_loss
            pre_p4_uniformity_sum += float(pre_p4_uniformity_loss.item()) * B
            weighted_pre_p4_uniformity_sum += float(pre_p4_uniformity_weight) * float(pre_p4_uniformity_loss.item()) * B
            pre_evidence_mask = make_pre_evidence_uniform_mask_tokens(
                y_pos_456=y,
                trial_T_tokens=int(trial_T_tokens),
                tone_T=int(tone_T),
                isi_T=int(isi_T),
                window=str(pre_evidence_uniform_kl_window),
            ).expand(B, 10, int(trial_T_tokens))
            pre_evidence_uniform_kl_loss = compute_uniform_class_kl_loss(
                logits=class_trial_raw.reshape(-1, 3),
                mask=pre_evidence_mask.reshape(-1),
            )
            if not torch.isfinite(pre_evidence_uniform_kl_loss):
                pre_evidence_uniform_kl_loss = end_loss.new_tensor(0.0)
            if float(pre_evidence_uniform_kl_weight) != 0.0:
                total_loss = total_loss + float(pre_evidence_uniform_kl_weight) * pre_evidence_uniform_kl_loss
            pre_evidence_uniform_kl_sum += float(pre_evidence_uniform_kl_loss.item()) * B
            weighted_pre_evidence_uniform_kl_sum += float(pre_evidence_uniform_kl_weight) * float(pre_evidence_uniform_kl_loss.item()) * B
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
                    pre_devend_cost_weight=float(pre_devend_cost_weight),
                    pre_devend_cost_mode=str(pre_devend_cost_mode),
                    pre_devend_cost_margin_ms=float(pre_devend_cost_margin_ms),
                    pre_devend_cost_scale_ms=float(pre_devend_cost_scale_ms),
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
            eff_w = float(online_loss_weight) * float(online_warmup_factor)
            if bool(include_online_decision_loss):
                total_loss = total_loss + eff_w * online_decision_loss
            if bool(include_online_ce_loss):
                total_loss = total_loss + float(online_ce_weight) * online_ce_loss
            if float(aux_token_ce_weight) != 0.0:
                total_loss = total_loss + float(aux_token_ce_weight) * aux_token_ce_loss
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
                    ent_bonus = stop_ent[elig].mean() if elig.any() else stop_ent.mean()
                    total_loss = total_loss - float(stop_entropy_weight) * ent_bonus
                    stop_entropy_sum += float(ent_bonus.item()) * B
                if float(stop_prior_weight) > 0.0:
                    mean_p = p_stop[elig].mean() if elig.any() else p_stop.mean()
                    sp_loss = (mean_p - float(stop_prior_target)) ** 2
                    total_loss = total_loss + float(stop_prior_weight) * sp_loss
                    stop_prior_sum += float(sp_loss.item()) * B
                if float(pre_devend_stop_weight) > 0.0:
                    dev_end_tok = deviant_end_token_in_trial(y, tone_T=int(tone_T), isi_T=int(isi_T))
                    tgrid = torch.arange(int(trial_T_tokens), device=x.device).view(1, 1, -1)
                    before_devend_mask = (tgrid < dev_end_tok.unsqueeze(-1)) & elig
                    if before_devend_mask.any():
                        pre_devend_stop_loss = p_stop[before_devend_mask].mean()
                        total_loss = total_loss + float(pre_devend_stop_weight) * pre_devend_stop_loss
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
            if "pre_devend_cost_loss" in od_stats and np.isfinite(od_stats["pre_devend_cost_loss"]):
                pre_devend_cost_sum += float(od_stats["pre_devend_cost_loss"]) * B
                weighted_pre_devend_cost_sum += float(od_stats.get("weighted_pre_devend_cost_loss", 0.0)) * B

            last_pred = class_trial[:, :, -1, :].argmax(dim=-1)
            last_token_acc_sum += float((last_pred == y_cls).float().mean().item()) * B
            last_token_acc_n += B
            dev_end_tok = deviant_end_token_in_trial(y, tone_T=int(tone_T), isi_T=int(isi_T))
            gather_idx = dev_end_tok.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
            devend_logits = class_trial.gather(dim=2, index=gather_idx).squeeze(2)
            devend_pred = devend_logits.argmax(dim=-1)
            devend_acc_sum += float((devend_pred == y_cls).float().mean().item()) * B
            devend_acc_n += B
        total_loss_sum += float(total_loss.item()) if use_raw_windowed_ce else (float(total_loss.item()) * B)

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
                if str(supervision_mode) == "strict_online_p4":
                    decision_tokens_cpu, found_cpu, pred_at_rt_cpu = compute_strict_p4_argmin_cost_decision_tokens_from_logits(
                        logits=logits_trial_cpu,
                        y_pos_456=y_cpu,
                        tone_T=int(tone_T),
                        isi_T=int(isi_T),
                        token_ms=int(token_ms),
                        time_cost_w=float(time_cost_w if time_cost_w is not None else 0.001),
                    )
                    dev_on_cpu = deviant_onset_token_in_trial(y_cpu, tone_T=int(tone_T), isi_T=int(isi_T))
                    rt_tokens_cpu = decision_tokens_cpu - dev_on_cpu
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
                    rt_negative += int((rt_tokens_cpu[found_cpu] < 0).sum().item())

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

    mean_rt_tokens = (rt_tokens_sum / rt_n) if rt_n > 0 else float("nan")
    mean_rt_ms = (rt_ms_sum / rt_n) if rt_n > 0 else float("nan")
    rt_not_first_rate = (rt_not_first / rt_n) if rt_n > 0 else float("nan")
    window_diag, window_diag_by_pos = finalize_window_epoch_diagnostics(window_batch_diags, token_ms=int(token_ms))
    end_acc = float((y_true == y_pred).mean()) if y_true.size > 0 else float("nan")
    end_f1 = safe_f1_macro(y_true, y_pred) if y_true.size > 0 else float("nan")
    end_auc = safe_auc_ovr(y_true, y_prob, n_classes=3) if y_true.size > 0 else float("nan")
    window_acc = float(window_diag.get("window_acc", float("nan")))
    window_f1 = float(window_diag.get("window_f1", float("nan")))
    window_auc = float(window_diag.get("window_auc", float("nan")))

    reported_token_loss = total_tok if (
        str(normalized_token_loss_mode) == "windowed_correct_ce" and (not bool(windowed_correct_ce_average))
    ) else (total_tok / denom)
    reported_total_loss = total_loss_sum if (
        str(normalized_token_loss_mode) == "windowed_correct_ce" and (not bool(windowed_correct_ce_average))
    ) else (total_loss_sum / denom)

    return {
        "end_loss": total_end / denom,
        "weighted_end_loss": float(lambda_end) * (total_end / denom),
        "token_loss": reported_token_loss,
        "online_loss_sum": float(online_loss_sum_diag),
        "online_valid_n": float(online_valid_n_diag),
        "online_loss_mean": float(online_loss_sum_diag / max(1.0, online_valid_n_diag)),
        "old_token_loss": (total_tok / denom) if normalized_token_loss_mode == "old" else float("nan"),
        "windowed_correct_ce_loss": reported_token_loss if normalized_token_loss_mode == "windowed_correct_ce" else float("nan"),
        "strict_p4_causal_ce_loss": (total_tok / denom) if normalized_token_loss_mode == "strict_p4_causal_ce" else float("nan"),
        "event_deviance_ce_loss": (total_tok / denom) if normalized_token_loss_mode == "event_deviance_ce" else float("nan"),
        "anti_commit_loss": total_anti / denom,
        "total_loss": reported_total_loss,
        "acc": window_acc,
        "f1_macro": window_f1,
        "auc_ovr": window_auc,
        "window_acc": window_acc,
        "window_f1": window_f1,
        "window_auc": window_auc,
        "window_acc_first_token": float(window_diag.get("window_acc_first_token", float("nan"))),
        "window_acc_last_token": float(window_diag.get("window_acc_last_token", float("nan"))),
        "window_acc_mean_token": float(window_diag.get("window_acc_mean_token", float("nan"))),
        "window_auc_mean_token": float(window_diag.get("window_auc_mean_token", float("nan"))),
        "window_p_correct_mean": float(window_diag.get("window_p_correct_mean", float("nan"))),
        "window_p_correct_std": float(window_diag.get("window_p_correct_std", float("nan"))),
        "window_p_correct_p10": float(window_diag.get("window_p_correct_p10", float("nan"))),
        "window_p_correct_p50": float(window_diag.get("window_p_correct_p50", float("nan"))),
        "window_p_correct_p90": float(window_diag.get("window_p_correct_p90", float("nan"))),
        "early_p_correct_mean": float(window_diag.get("early_p_correct_mean", float("nan"))),
        "window_p_correct_end": float(window_diag.get("window_p_correct_end", float("nan"))),
        "window_p_correct_max": float(window_diag.get("window_p_correct_max", float("nan"))),
        "window_p_correct_auc": float(window_diag.get("window_p_correct_auc", float("nan"))),
        "window_prediction_mean": float(window_diag.get("window_prediction_mean", float("nan"))),
        "window_prediction_std": float(window_diag.get("window_prediction_std", float("nan"))),
        "window_prediction_min": float(window_diag.get("window_prediction_min", float("nan"))),
        "window_prediction_max": float(window_diag.get("window_prediction_max", float("nan"))),
        "window_y_true_frac_class0": float(window_diag.get("window_y_true_frac_class0", float("nan"))),
        "window_y_true_frac_class1": float(window_diag.get("window_y_true_frac_class1", float("nan"))),
        "window_y_true_frac_class2": float(window_diag.get("window_y_true_frac_class2", float("nan"))),
        "end_acc": end_acc,
        "end_f1": float(end_f1),
        "end_auc": float(end_auc),
        "mean_rt_tokens": float(mean_rt_tokens),
        "mean_rt_ms": float(mean_rt_ms),
        "rt_found": int(rt_n),
        "rt_miss": int(rt_miss),
        "rt_not_first": int(rt_not_first),
        "rt_negative": int(rt_negative),
        "rt_not_first_rate": float(rt_not_first_rate),
        "window_diagnostics": window_diag,
        "window_diagnostics_by_position": window_diag_by_pos,
        "online_decision_cost": float(online_cost_sum / max(1, online_cost_n)) if online_cost_n > 0 else float("nan"),
        "online_decision_loss": float(online_loss_sum / denom) if online_decision_training else float("nan"),
        "online_ce_loss": float(online_ce_sum / denom) if online_decision_training else float("nan"),
        "aux_token_ce_loss": float(aux_token_ce_sum / denom) if aux_token_ce_sum > 0 else 0.0,
        "pre_p4_uniformity_loss": float(pre_p4_uniformity_sum / denom) if pre_p4_uniformity_sum > 0 else 0.0,
        "pre_evidence_uniform_kl_loss": float(pre_evidence_uniform_kl_sum / denom) if pre_evidence_uniform_kl_sum > 0 else 0.0,
        "phase_name": str(phase_name),
        "anti_immediate_stop_loss": float(anti_immediate_stop_sum / denom) if anti_immediate_stop_sum > 0 else 0.0,
        "stop_entropy_bonus": float(stop_entropy_sum / denom) if stop_entropy_sum > 0 else 0.0,
        "stop_prior_loss": float(stop_prior_sum / denom) if stop_prior_sum > 0 else 0.0,
        "lambda_online": float(lambda_online),
        "lambda_end_current": float(lambda_end),
        "effective_online_loss_weight": float(online_loss_weight) * float(online_warmup_factor) if include_online_decision_loss else 0.0,
        "weighted_token_loss": float(lambda_online) * float(reported_token_loss),
        "weighted_online_decision_loss": (float(online_loss_weight) * float(online_warmup_factor) * (online_loss_sum / denom)) if include_online_decision_loss and denom > 0 else 0.0,
        "weighted_online_ce_loss": (float(online_ce_weight) * (online_ce_sum / denom)) if include_online_ce_loss and denom > 0 else 0.0,
        "weighted_aux_token_ce_loss": float(weighted_aux_token_ce_sum / denom) if weighted_aux_token_ce_sum > 0 else 0.0,
        "weighted_pre_p4_uniformity_loss": float(weighted_pre_p4_uniformity_sum / denom) if weighted_pre_p4_uniformity_sum > 0 else 0.0,
        "weighted_pre_evidence_uniform_kl_loss": float(weighted_pre_evidence_uniform_kl_sum / denom) if weighted_pre_evidence_uniform_kl_sum > 0 else 0.0,
        "preEvidenceUniformKlWindow": str(pre_evidence_uniform_kl_window),
        "preDevCostRaw": float(pre_devend_cost_sum / denom) if pre_devend_cost_sum > 0 else 0.0,
        "wPreDevCost": float(weighted_pre_devend_cost_sum / denom) if weighted_pre_devend_cost_sum > 0 else 0.0,
        "preDevCostMode": str(pre_devend_cost_mode),
        "preDevCostWeight": float(pre_devend_cost_weight),
        "preDevCostMargin": float(pre_devend_cost_margin_ms),
        "preDevCostScale": float(pre_devend_cost_scale_ms),
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
        add_bos=bool(getattr(args, "add_bos", False)),
        eos_mode=str(getattr(args, "eos_mode", "separate")),
        sigma_other_noise=float(args.sigma_other_noise),
        p_other_noise=float(args.p_other_noise),
        sigma_silence_noise=float(args.sigma_silence_noise),
        resample_noise_per_epoch=False,
        quiet=True,
        encoding_mode=str(getattr(args, "encoding_mode", "onehot")),
        sigma_rf=float(getattr(args, "sigma_rf", 1.0)),
        rf_normalization=str(getattr(args, "rf_normalization", "peak")),
        sigma_rf_noise=float(getattr(args, "sigma_rf_noise", 0.0)),
        rf_noise_per_token=bool(getattr(args, "rf_noise_per_token", True)),
        noise_mode=str(getattr(args, "noise_mode", "per_token")),
        noise_rho=float(getattr(args, "noise_rho", 0.0)),
        use_prerendered_tokens=bool(getattr(args, "use_prerendered_tokens", False)),
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
        add_bos=bool(getattr(args, "add_bos", False)),
        eos_mode=str(getattr(args, "eos_mode", "separate")),
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
        encoding_mode=str(getattr(args, "encoding_mode", "onehot")),
        sigma_rf=float(getattr(args, "sigma_rf", 1.0)),
        rf_normalization=str(getattr(args, "rf_normalization", "peak")),
        sigma_rf_noise=float(getattr(args, "sigma_rf_noise", 0.0)),
        rf_noise_per_token=bool(getattr(args, "rf_noise_per_token", True)),
        noise_mode=str(getattr(args, "noise_mode", "per_token")),
        noise_rho=float(getattr(args, "noise_rho", 0.0)),
    )

    # rebuild model
    cfg = ModelConfig(
        input_dim=int(cfg_dict["input_dim"]),
        hidden_dim=int(cfg_dict["hidden_dim"]),
        num_classes=int(cfg_dict.get("num_classes", 3)),
        num_layers=int(cfg_dict["num_layers"]),
        dropout=float(cfg_dict["dropout"]),
        layer_norm=bool(cfg_dict.get("layer_norm", False)),
        hidden_noise_std=float(cfg_dict.get("hidden_noise_std", 0.0)),
        use_stop_head=bool(cfg_dict.get("use_stop_head", False)),
        use_event_head=bool(cfg_dict.get("use_event_head", False)),
    )
    model = PredictiveGRU(cfg).to(device)
    print(
        "[model] "
        f"supervision_mode={str(getattr(args, 'supervision_mode', 'post_deviant'))} "
        f"output_dim={int(cfg.num_classes)}"
    )
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
            end_idx = infer_end_indices_from_T(
                T,
                trials_per_block=10,
                end_offset_from_trial_end=(1 if bool(getattr(args, "add_bos", False)) else 0),
            ).to(device)
            logits_end, token_loss, _, logits_all, _, _, _h_d3 = _run_block_through_tbptt(
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
                block_context_training=bool(getattr(args, "block_context_training", False)),
                detach_hidden_between_trials=bool(getattr(args, "detach_hidden_between_trials", False)),
                token_tau=float(args.token_tau),
                token_w_min=float(args.token_w_min),
                token_supervision_reference=str(getattr(args, "token_supervision_reference", "deviant_offset")),
                include_anchor_token=bool(getattr(args, "include_anchor_token", False)),
                correct_ce_window=str(getattr(args, "correct_ce_window", "deviant_onset_to_next_tone_offset")),
                correct_ce_weighting=str(getattr(args, "correct_ce_weighting", "equal")),
                return_full_logits=True,
                use_mask_loss=bool(getattr(args, "use_mask_loss", False)),
                important_token_indices=getattr(args, "important_token_indices", None),
                supervision_mode=str(getattr(args, "supervision_mode", cfg_dict.get("supervision_mode", "post_deviant"))),
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
            logits_trial = logits_all.view(B, 10, int(ds.trial_T_tokens), int(logits_all.shape[-1])).detach().cpu()
            y_cpu = y_pos_456.detach().cpu().long()
            y_cls_cpu = (y_cpu - 4).long()
            if str(getattr(args, "supervision_mode", "post_deviant")) == "strict_online_p4":
                decision_tokens_cpu, found_cpu, _ = compute_strict_p4_argmin_cost_decision_tokens_from_logits(
                    logits=logits_trial,
                    y_pos_456=y_cpu,
                    tone_T=int(ds.tone_T),
                    isi_T=int(ds.isi_T),
                    token_ms=int(ds.token_ms),
                    time_cost_w=float(getattr(args, "time_cost_w", 0.001)),
                )
                dev_on_cpu = deviant_onset_token_in_trial(y_cpu, tone_T=int(ds.tone_T), isi_T=int(ds.isi_T))
                rt_tokens_cpu = decision_tokens_cpu - dev_on_cpu
            else:
                dev_end_cpu = deviant_end_token_in_trial(y_pos_456=y_cpu, tone_T=int(ds.tone_T), isi_T=int(ds.isi_T))

                rt_tokens_cpu, found_cpu, _ = compute_rt_from_logits(
                    logits=logits_trial,
                    y_cls=y_cls_cpu,
                    dev_end=dev_end_cpu,
                    p_thresh=float(args.rt_p_thresh),
                    k_consec=int(args.rt_k_consec),
                )
            rt_ms_cpu = rt_tokens_cpu.float() * float(ds.token_ms)
            probs_trial = torch.softmax(logits_trial, dim=-1)[..., :3]
            pred_trial = probs_trial.argmax(dim=-1)
            pmax_trial = probs_trial.max(dim=-1).values
            p4_tokens_cpu = first_possible_deviant_onset_token_in_trial(y_cpu, tone_T=int(ds.tone_T), isi_T=int(ds.isi_T))
            dev_on_cpu = deviant_onset_token_in_trial(y_cpu, tone_T=int(ds.tone_T), isi_T=int(ds.isi_T))

            for b in range(B):
                for tr in range(10):
                    decision_token = int(decision_tokens_cpu[b, tr].item())
                    raw_rt_ms = float(rt_ms_cpu[b, tr].item())
                    if decision_token >= 0:
                        pred_pos = int(pred_trial[b, tr, decision_token].item() + 4)
                        decision_conf = float(pmax_trial[b, tr, decision_token].item())
                    else:
                        pred_pos = -1
                        decision_conf = float("nan")
                    rows.append({
                        "model_id": model_id,
                        "isi_ms": int(isi_ms_for_export),
                        "supervision_mode": str(getattr(args, "supervision_mode", "post_deviant")),
                        "position": int(y_cpu[b, tr].item()),
                        "rt_ms": raw_rt_ms if np.isfinite(raw_rt_ms) else raw_rt_ms,
                        "model_rt_raw": raw_rt_ms if np.isfinite(raw_rt_ms) else raw_rt_ms,
                        "model_rt_clipped": float(max(raw_rt_ms, 0.0)) if np.isfinite(raw_rt_ms) else float("nan"),
                        "found": int(bool(found_cpu[b, tr].item())),
                        "predicted_deviant_position": pred_pos,
                        "decision_token": decision_token,
                        "decision_confidence": decision_conf,
                        "p4_token": int(p4_tokens_cpu[b, tr].item()),
                        "true_deviant_token": int(dev_on_cpu[b, tr].item()),
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
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot_history] skip plots because matplotlib is unavailable: {e}")
        return

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


def plot_window_diagnostics(
    window_rows: List[Dict[str, Any]],
    by_pos_rows: List[Dict[str, Any]],
    rf_rows: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    def _parse_numeric_list(value: Any) -> np.ndarray:
        if value is None:
            return np.array([], dtype=float)
        if isinstance(value, np.ndarray):
            try:
                return value.astype(float)
            except Exception:
                return np.array([], dtype=float)
        if isinstance(value, (list, tuple)):
            try:
                return np.asarray(value, dtype=float)
            except Exception:
                return np.array([], dtype=float)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return np.array([], dtype=float)
            try:
                return np.asarray(json.loads(s), dtype=float)
            except Exception:
                pass
            try:
                import ast
                return np.asarray(ast.literal_eval(s), dtype=float)
            except Exception:
                return np.array([], dtype=float)
        try:
            return np.asarray(value, dtype=float)
        except Exception:
            return np.array([], dtype=float)

    def _parse_mapping_of_numeric_lists(value: Any) -> Dict[str, np.ndarray]:
        if value is None:
            return {}
        if isinstance(value, dict):
            parsed: Dict[str, np.ndarray] = {}
            for k, v in value.items():
                arr = _parse_numeric_list(v)
                if arr.size > 0:
                    parsed[str(k)] = arr
            return parsed
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return {}
            for parser in (json.loads,):
                try:
                    obj = parser(s)
                    if isinstance(obj, dict):
                        return _parse_mapping_of_numeric_lists(obj)
                except Exception:
                    pass
            try:
                import ast
                obj = ast.literal_eval(s)
                if isinstance(obj, dict):
                    return _parse_mapping_of_numeric_lists(obj)
            except Exception:
                pass
        return {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot_window_diagnostics] skip plots because matplotlib is unavailable: {e}")
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        val_rows = [r for r in window_rows if str(r.get("split")) == "val"]
        if val_rows:
            latest = val_rows[-1]
            rel_ms = _parse_numeric_list(latest.get("trajectory_rel_ms", []))
            traj = _parse_mapping_of_numeric_lists(latest.get("trajectory_by_position", {}))
            if rel_ms.size > 0 and traj:
                plt.figure()
                plotted = False
                for pos, color in [(4, "#b22222"), (5, "#1f77b4"), (6, "#228b22")]:
                    vals = traj.get(str(pos), np.array([], dtype=float))
                    if vals.size == rel_ms.size:
                        plt.plot(rel_ms, vals, label=f"P{pos}", color=color)
                        plotted = True
                if plotted:
                    plt.xlabel("Window-Relative Time (ms)")
                    plt.ylabel("p_correct")
                    plt.title("p_correct Trajectory by Position")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(out_dir / "p_correct_trajectory_by_position.png", dpi=160)
                else:
                    print("[plot_window_diagnostics] warning: no valid trajectory_by_position series matched trajectory_rel_ms; skip p_correct trajectory plot.")
                plt.close()

        if by_pos_rows:
            latest_epoch = max(r.get("epoch_global", 0) for r in by_pos_rows)
            latest_val = [r for r in by_pos_rows if r.get("epoch_global", 0) == latest_epoch and str(r.get("split")) == "val"]
            if latest_val:
                xs = [f"P{int(r['position'])}" for r in latest_val]
                ys_end = [float(r.get("p_correct_at_window_end", float("nan"))) for r in latest_val]
                ys_rt = [float(r.get("first_crossing_060_ms", float("nan"))) for r in latest_val]
                plt.figure()
                plt.bar(xs, ys_end, color=["#b22222", "#1f77b4", "#228b22"])
                plt.ylabel("p_correct at Window End")
                plt.title("p_correct at Window End by Position")
                plt.tight_layout()
                plt.savefig(out_dir / "p_correct_at_window_end_by_position.png", dpi=160)
                plt.close()

                plt.figure()
                plt.bar(xs, ys_rt, color=["#b22222", "#1f77b4", "#228b22"])
                plt.ylabel("Median First-Crossing RT 0.60 (ms)")
                plt.title("First-Crossing RT by Position")
                plt.tight_layout()
                plt.savefig(out_dir / "first_crossing_rt_by_position.png", dpi=160)
                plt.close()

        if rf_rows:
            latest_rf = rf_rows[-1]
            if "rf_curve_bin" in latest_rf:
                bins = _parse_numeric_list(latest_rf.get("rf_curve_bin", []))
                std_curve = _parse_numeric_list(latest_rf.get("rf_curve_standard", []))
                dev_curve = _parse_numeric_list(latest_rf.get("rf_curve_deviant", []))
                if bins.size > 0 and std_curve.size == bins.size and dev_curve.size == bins.size:
                    plt.figure()
                    plt.plot(bins, std_curve, label="standard_rf")
                    plt.plot(bins, dev_curve, label="deviant_rf")
                    plt.xlabel("Frequency Bin")
                    plt.ylabel("Activation")
                    plt.title(
                        f"RF Curves sigma_rf={latest_rf.get('sigma_rf', float('nan'))} "
                        f"sigma_rf_noise={latest_rf.get('sigma_rf_noise', float('nan'))}"
                    )
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(out_dir / "rf_curves_current_sigma.png", dpi=160)
                    plt.close()
                else:
                    print("[plot_window_diagnostics] warning: invalid RF curve arrays; skip rf_curves_current_sigma plot.")
    except Exception as e:
        print(f"[plot_window_diagnostics] warning: plotting failed but training will continue: {e}")


def plot_compact_trial_probability_timeline(
    logits_all: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_ms: int,
    out_dir: Path,
    epoch_global: int,
    max_blocks: int = 3,
    time_cost_w: float = 0.001,
) -> Optional[Path]:
    """Save one wrapped compact heatmap: one row per 10-trial block."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patches as patches
    except Exception as e:
        print(f"[compact_trial_timeline] skip because matplotlib is unavailable: {e}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        B, T, C = logits_all.shape
        Tt = int(trial_T_tokens)
        if T % Tt != 0:
            return None
        n_trials_block = T // Tt
        class_trial = logits_all.view(B, n_trials_block, Tt, C).detach().cpu()
        probs = torch.softmax(class_trial, dim=-1)[..., :3].numpy()
        y_np = y_pos_456.detach().cpu().long().numpy()

    n_blocks = min(int(B), max(1, int(max_blocks)))
    if n_blocks <= 0:
        return None
    trials_per_row = int(n_trials_block)
    n_rows = int(n_blocks)
    tone_step = int(tone_T + isi_T)
    cmap = LinearSegmentedColormap.from_list("std_to_dev", ["#2166AC", "#F7F7F7", "#B2182B"])

    fig_h = max(2.4, 1.25 * n_rows + 0.6)
    fig_w = max(8.0, 1.25 * trials_per_row)
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[:, 0]
    for row_i, ax in enumerate(axes):
        ax.set_xlim(0, trials_per_row)
        ax.set_ylim(0, 1.0)
        ax.axis("off")
        for j in range(trials_per_row):
            true_pos = int(y_np[row_i, j])
            pr = probs[row_i, j]
            x0 = float(j)
            top_y = 0.70
            top_h = 0.18
            for tone_pos in range(1, 9):
                color = "#B2182B" if int(tone_pos) == int(true_pos) else "#2166AC"
                tx = x0 + (tone_pos - 1) / 8.0
                ax.add_patch(patches.Rectangle((tx, top_y), 1.0 / 8.0 - 0.006, top_h, color=color, linewidth=0))
                if tone_pos in (4, 5, 6):
                    ax.text(tx + 1.0 / 16.0, top_y + top_h + 0.03, f"P{tone_pos}", ha="center", va="bottom", fontsize=5)

            # Three probability rows over the true trial time axis.
            heat = np.asarray(pr[:, :3].T, dtype=float)
            ax.imshow(
                heat,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                extent=(x0, x0 + 1.0, 0.18, 0.56),
                origin="lower",
            )

            p4_token = int(3 * tone_step)
            cls = int(true_pos - 4)
            if 0 <= cls <= 2 and p4_token < Tt:
                p_correct = pr[:, cls]
                elapsed_ms = np.arange(Tt - p4_token, dtype=float) * float(token_ms)
                expected_cost = (1.0 - p_correct[p4_token:]) + p_correct[p4_token:] * float(time_cost_w) * elapsed_ms
                decision_tok = int(p4_token + np.nanargmin(expected_cost))
                dx = x0 + min(1.0, max(0.0, decision_tok / float(max(1, Tt))))
                ax.plot([dx, dx], [0.14, 0.60], color="#111111", lw=0.6, linestyle="--", alpha=0.85)
                ax.text(dx, 0.12, "argmin", ha="center", va="top", fontsize=5)
            ax.text(x0 + 0.50, 0.96, f"trial {j + 1} dev=P{true_pos}", ha="center", va="top", fontsize=6)
        if row_i == 0:
            ax.text(
                0.0,
                1.08,
                f"epoch {int(epoch_global)} | top: stimulus; bottom: model P(P4/P5/P6), dashed=argmin expected cost",
                ha="left",
                va="bottom",
                fontsize=8,
            )
        ax.text(-0.08, 0.48, f"block {row_i + 1}", ha="right", va="center", fontsize=8, rotation=90)
        ax.text(-0.02, 0.24, "P4", ha="right", va="center", fontsize=6)
        ax.text(-0.02, 0.37, "P5", ha="right", va="center", fontsize=6)
        ax.text(-0.02, 0.50, "P6", ha="right", va="center", fontsize=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.018, pad=0.01)
    cbar.set_label("P(candidate is deviant)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    png = out_dir / "compact_trial_timeline.png"
    pdf = out_dir / "compact_trial_timeline.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_aggregate_probability_by_position(
    logits_all: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    token_ms: int,
    out_dir: Path,
) -> Optional[Path]:
    """Save mean P(correct) and mean P(max P4/P5/P6) grouped by true deviant position."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[aggregate_probability_by_position] skip because matplotlib is unavailable: {e}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        B, T, C = logits_all.shape
        Tt = int(trial_T_tokens)
        if T % Tt != 0:
            return None
        n_trials_block = T // Tt
        probs = torch.softmax(logits_all.view(B, n_trials_block, Tt, C), dim=-1)[..., :3].detach().cpu().numpy()
        y_np = y_pos_456.detach().cpu().long().numpy()

    time_ms = np.arange(Tt, dtype=float) * float(token_ms)
    p4_ms = float(3 * (int(tone_T) + int(isi_T)) * int(token_ms))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8), sharex=True, sharey=True)
    palette = {4: "#32037D", 5: "#C94E65", 6: "#D9995B"}
    for ax, pos in zip(axes, [4, 5, 6]):
        mask = y_np == int(pos)
        if not np.any(mask):
            ax.set_title(f"P{pos}: no trials")
            continue
        cls = int(pos - 4)
        p_correct = probs[..., cls][mask]
        p_max = probs.max(axis=-1)[mask]
        ax.plot(time_ms, np.nanmean(p_correct, axis=0), color=palette[pos], lw=1.5, label="P(correct)")
        ax.plot(time_ms, np.nanmean(p_max, axis=0), color="#333333", lw=1.0, alpha=0.75, label="P(max)")
        dev_ms = float((int(pos) - 1) * (int(tone_T) + int(isi_T)) * int(token_ms))
        ax.axvline(p4_ms, color="#999999", lw=0.8, linestyle="--")
        ax.axvline(dev_ms, color=palette[pos], lw=0.8, linestyle=":")
        ax.set_title(f"True deviant P{pos}", fontsize=9)
        ax.set_xlabel("Time in trial (ms)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean probability")
    axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    png = out_dir / "aggregate_probability_by_position.png"
    pdf = out_dir / "aggregate_probability_by_position.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


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

# ============================================================
# Human RT correlation monitoring (every N epochs during training)
# ============================================================
def monitor_human_rt_correlation(
    model: PredictiveGRU,
    human_csv: Path,
    device: torch.device,
    chunk_len: int,
    token_ms: int,
    tone_ms: int,
    isi_ms: int,
    f_min_hz: float,
    f_max_hz: float,
    n_bins: int,
    add_eos: bool,
    add_bos: bool,
    sigma_other_noise: float,
    p_other_noise: float,
    sigma_silence_noise: float,
    seed: int,
    encoding_mode: str = "onehot",
    sigma_rf: float = 1.0,
    rf_normalization: str = "peak",
    sigma_rf_noise: float = 0.0,
    rf_noise_per_token: bool = True,
    rt_p_thresh: float = 0.7,
    rt_k_consec: int = 1,
    rt_mode: str = "entropy",
    rt_entropy_thresh: float = 0.35,
    min_rt_tokens: int = 0,
    readout_mode: str = "simple_threshold",
    readout_start: str = "deviant_onset",
    readout_end: str = "trial_end",
    rt_reference: str = "deviant_onset",
    advisor_time_cost: float = 0.0005,
    expected_cost_threshold: float = 0.5,
    advisor_force_deadline: bool = False,
    decision_not_before: str = "window_start",
    cost_elapsed_reference: str = "window_start",
    decision_min_elapsed_ms: float = 0.0,
) -> Dict[str, float]:
    """
    Compute three trial-by-trial Pearson correlations:
      a) model_predicted_rt vs human_observed_rt
      b) model_predicted_rt vs optimal_prior (1/3, 1/2, 1)
      c) human_observed_rt vs optimal_prior

    Returns dict with keys:
      human_rt_corr_r, human_rt_corr_p
      predicted_dependency_r, predicted_dependency_p
      observed_dependency_r, observed_dependency_p
      n_trials
    """
    try:
        import pandas as pd
    except Exception:
        return {"human_rt_corr_r": float("nan"), "n_trials": 0}

    try:
        from scipy.stats import pearsonr
        _HAVE_SCIPY = True
    except Exception:
        _HAVE_SCIPY = False

    if not human_csv.exists():
        print(f"[human_rt_monitor] human_csv not found: {human_csv}. Skip.")
        return {"human_rt_corr_r": float("nan"), "n_trials": 0}

    # Load human trials
    df = pd.read_csv(human_csv)

    # Filter to target ISI
    if "isi_ms" in df.columns:
        df = df[df["isi_ms"] == int(isi_ms)].copy()
    if df.shape[0] == 0:
        print(f"[human_rt_monitor] No trials for ISI={isi_ms}. Skip.")
        return {"human_rt_corr_r": float("nan"), "n_trials": 0}

    # Detect columns
    col_map = {}
    pos_candidates = ["position", "pos", "deviant_pos", "dev_pos", "deviant_position"]
    rt_candidates = ["rt_ms", "rt", "reaction_time_ms", "reaction_time", "RT_ms", "RT"]
    std_candidates = ["f_std", "std_hz", "standard_hz", "f_standard", "standard_freq_hz", "std_freq"]
    dev_candidates = ["f_dev", "dev_hz", "deviant_hz", "f_deviant", "deviant_freq_hz", "dev_freq"]

    for name, candidates in [("position", pos_candidates), ("rt_ms", rt_candidates),
                              ("f_std", std_candidates), ("f_dev", dev_candidates)]:
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            print(f"[human_rt_monitor] Missing column: {name} (tried {candidates}). Skip.")
            return {"human_rt_corr_r": float("nan"), "n_trials": 0}
        col_map[name] = found

    # Render human trials as model input
    try:
        from eval_models_on_human import render_trials_onehot, forward_full_logits
        from rt_readout import prepare_rt_readout, run_rt_readout_sweeps, labels_to_class_index
    except Exception as e:
        print(f"[human_rt_monitor] Cannot import render/readout helpers: {e}. Skip.")
        return {"human_rt_corr_r": float("nan"), "n_trials": 0}

    prev_training = bool(model.training)
    try:
        X, y_pos, trial_T_tokens, tone_T, isi_T = render_trials_onehot(
            df=df, col=col_map,
            tone_ms=int(tone_ms), token_ms=int(token_ms), isi_ms=int(isi_ms),
            f_min_hz=float(f_min_hz), f_max_hz=float(f_max_hz),
            n_bins=int(n_bins), add_eos=bool(add_eos), add_bos=bool(add_bos),
            sigma_other_noise=float(sigma_other_noise),
            p_other_noise=float(p_other_noise),
            sigma_silence_noise=float(sigma_silence_noise),
            encoding_mode=str(encoding_mode),
            sigma_rf=float(sigma_rf),
            rf_normalization=str(rf_normalization),
            sigma_rf_noise=float(sigma_rf_noise),
            rf_noise_per_token=bool(rf_noise_per_token),
            seed=int(seed),
        )

        # Forward model
        X = X.to(device)
        logits_all = forward_full_logits(model, X, chunk_len=int(chunk_len))  # (N, T, 3)

        prepared = prepare_rt_readout(
            logits_trial=logits_all.detach().cpu(),
            y_cls=labels_to_class_index(y_pos),
            y_pos_456=y_pos,
            tone_T=int(tone_T),
            isi_T=int(isi_T),
            token_ms=int(token_ms),
            readout_start=str(readout_start),
            readout_end=str(readout_end),
            rt_reference=str(rt_reference),
        )
        if str(readout_mode) == "advisor_expected_cost_dp":
            trial_rows, _ = run_rt_readout_sweeps(
                prepared,
                rt_readout_mode="advisor_expected_cost_dp",
                bayes_time_cost_list=[float(advisor_time_cost)],
                bayes_k_consec_list=[int(rt_k_consec)],
                bayes_force_deadline=bool(advisor_force_deadline),
                decision_not_before=str(decision_not_before),
                cost_elapsed_reference=str(cost_elapsed_reference),
                decision_min_elapsed_ms=float(decision_min_elapsed_ms),
            )
        elif str(readout_mode) == "expected_cost_threshold":
            trial_rows, _ = run_rt_readout_sweeps(
                prepared,
                rt_readout_mode="expected_cost_threshold",
                cost_w_list=[float(advisor_time_cost)],
                cost_threshold_list=[float(expected_cost_threshold)],
                k_consec_list=[int(rt_k_consec)],
                decision_not_before=str(decision_not_before),
                cost_elapsed_reference=str(cost_elapsed_reference),
                decision_min_elapsed_ms=float(decision_min_elapsed_ms),
            )
        elif str(readout_mode) == "simple_threshold":
            trial_rows, _ = run_rt_readout_sweeps(
                prepared,
                rt_readout_mode="simple_threshold",
                p_threshold_list=[float(rt_p_thresh)],
                k_consec_list=[int(rt_k_consec)],
                decision_not_before=str(decision_not_before),
                cost_elapsed_reference=str(cost_elapsed_reference),
                decision_min_elapsed_ms=float(decision_min_elapsed_ms),
            )
        else:
            raise ValueError(f"Unsupported human RT monitor readout_mode: {readout_mode}")
    finally:
        if prev_training:
            model.train()
        else:
            model.eval()

    rt_df = pd.DataFrame(trial_rows).sort_values("trial_index", kind="stable").reset_index(drop=True)
    model_rt_ms = pd.to_numeric(rt_df["model_rt_ms"], errors="coerce").to_numpy(dtype=float)
    found_np = rt_df["found_flag"].astype(bool).to_numpy()

    # For advisor DP readout, treat "not found" as a timeout RT rather than dropping
    # those trials from the RT correlation.
    if str(readout_mode) == "advisor_expected_cost_dp":
        model_rt_ms = model_rt_ms.copy()
        model_rt_ms[~found_np] = 1500.0

    # Human RT
    human_rt_ms = df[col_map["rt_ms"]].to_numpy(dtype=float)

    # Optimal prior for each position: P4=1/3, P5=1/2, P6=1
    pos = df[col_map["position"]].to_numpy(dtype=int)
    optimal_prior = np.where(pos == 4, 1.0/3.0, np.where(pos == 5, 0.5, 1.0))

    # For simple threshold, keep the old "found only" behavior.
    # For advisor DP, include timeout-imputed trials as long as RTs are finite.
    if str(readout_mode) == "advisor_expected_cost_dp":
        valid = np.isfinite(human_rt_ms) & np.isfinite(model_rt_ms)
    else:
        valid = found_np & np.isfinite(human_rt_ms) & np.isfinite(model_rt_ms)
    n_valid = int(valid.sum())

    result: Dict[str, float] = {
        "human_rt_corr_r": float("nan"),
        "human_rt_corr_r2": float("nan"),
        "human_rt_corr_p": float("nan"),
        "predicted_dependency_r": float("nan"),
        "predicted_dependency_p": float("nan"),
        "observed_dependency_r": float("nan"),
        "observed_dependency_r2": float("nan"),
        "observed_dependency_p": float("nan"),
        "delta_r2": float("nan"),
        "n_trials": n_valid,
        "n_total": int(df.shape[0]),
        "n_found": int(found_np.sum()),
        "found_rate": float(found_np.mean()) if found_np.size > 0 else float("nan"),
        "readout_mode": str(readout_mode),
    }

    if n_valid < 10:
        print(f"[human_rt_monitor] Only {n_valid} valid trials (need >=10). Skip correlation.")
        return result

    # a) model predicted RT vs human observed RT
    mr = model_rt_ms[valid]
    hr = human_rt_ms[valid]
    if _HAVE_SCIPY:
        r_a, p_a = pearsonr(mr, hr)
        result["human_rt_corr_r"] = float(r_a)
        result["human_rt_corr_r2"] = float(r_a * r_a) if np.isfinite(r_a) else float("nan")
        result["human_rt_corr_p"] = float(p_a)
    else:
        result["human_rt_corr_r"] = float(np.corrcoef(mr, hr)[0, 1])
        result["human_rt_corr_r2"] = float(result["human_rt_corr_r"] * result["human_rt_corr_r"]) if np.isfinite(result["human_rt_corr_r"]) else float("nan")

    # b) model predicted RT vs optimal prior
    op_valid = optimal_prior[valid]
    if _HAVE_SCIPY:
        r_b, p_b = pearsonr(mr, op_valid)
        result["predicted_dependency_r"] = float(r_b)
        result["predicted_dependency_p"] = float(p_b)
    else:
        result["predicted_dependency_r"] = float(np.corrcoef(mr, op_valid)[0, 1])

    # c) human observed RT vs optimal prior
    if _HAVE_SCIPY:
        r_c, p_c = pearsonr(hr, op_valid)
        result["observed_dependency_r"] = float(r_c)
        result["observed_dependency_r2"] = float(r_c * r_c) if np.isfinite(r_c) else float("nan")
        result["observed_dependency_p"] = float(p_c)
    else:
        result["observed_dependency_r"] = float(np.corrcoef(hr, op_valid)[0, 1])
        result["observed_dependency_r2"] = float(result["observed_dependency_r"] * result["observed_dependency_r"]) if np.isfinite(result["observed_dependency_r"]) else float("nan")

    if np.isfinite(result["human_rt_corr_r2"]) and np.isfinite(result["observed_dependency_r2"]):
        result["delta_r2"] = float(result["human_rt_corr_r2"] - result["observed_dependency_r2"])

    return result


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


def normalize_selection_metric_name(name: str) -> str:
    s = str(name).strip().lower()
    aliases = {
        "loss": "val_loss",
        "val_total_loss": "val_loss",
        "human_rt_corr_r2": "human_rt_r2",
        "rt_r2": "human_rt_r2",
        "r2": "human_rt_r2",
        "delta": "delta_r2",
        "delta_r2": "delta_r2",
    }
    return aliases.get(s, s)


def metric_direction(name: str) -> str:
    metric = normalize_selection_metric_name(name)
    if metric == "val_loss":
        return "min"
    if metric == "human_rt_r2":
        return "max"
    if metric == "delta_r2":
        return "max"
    raise ValueError(f"Unknown metric direction for: {name}")


def metric_initial_best(name: str) -> float:
    return float("inf") if metric_direction(name) == "min" else float("-inf")


def metric_value_from_sources(
    metric_name: str,
    val_metrics: Dict[str, Any],
    human_rt_metrics: Optional[Dict[str, Any]] = None,
) -> float:
    metric = normalize_selection_metric_name(metric_name)
    if metric == "val_loss":
        return _safe_float(val_metrics.get("total_loss"), float("nan"))
    if metric == "human_rt_r2":
        hr = human_rt_metrics or {}
        r = _safe_float(hr.get("human_rt_corr_r"), float("nan"))
        return float(r * r) if np.isfinite(r) else float("nan")
    if metric == "delta_r2":
        hr = human_rt_metrics or {}
        a_r = _safe_float(hr.get("human_rt_corr_r"), float("nan"))
        c_r = _safe_float(hr.get("observed_dependency_r"), float("nan"))
        if np.isfinite(a_r) and np.isfinite(c_r):
            return float((a_r * a_r) - (c_r * c_r))
        return float("nan")
    raise ValueError(f"Unsupported metric: {metric_name}")


def metric_improved(metric_name: str, current: float, best: float, min_delta: float = 0.0) -> bool:
    if not np.isfinite(current):
        return False
    direction = metric_direction(metric_name)
    if not np.isfinite(best):
        return True
    if direction == "min":
        return current < (best - float(min_delta))
    return current > (best + float(min_delta))


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
    add_bos: bool,
    eos_mode: str,
    sigma_other_noise: float,
    p_other_noise: float,
    sigma_silence_noise: float,
    resample_noise_per_epoch: bool,
    train_idx: List[int],
    val_idx: List[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    assert_labels: bool = False,
    encoding_mode: str = "onehot",
    sigma_rf: float = 1.0,
    rf_normalization: str = "peak",
    sigma_rf_noise: float = 0.0,
    rf_noise_per_token: bool = True,
    noise_mode: str = "per_token",
    noise_rho: float = 0.0,
    use_prerendered_tokens: bool = False,
    pin_memory: Optional[bool] = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> Tuple[OnlineRenderDataset, DataLoader, DataLoader]:
    ds_cls = PreRenderedBlockDataset if bool(use_prerendered_tokens) else OnlineRenderDataset
    ds_train = ds_cls(
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
        add_bos=bool(add_bos),
        eos_mode=str(eos_mode),
        sigma_other_noise=sigma_other_noise,
        p_other_noise=p_other_noise,
        sigma_silence_noise=sigma_silence_noise,
        **({"resample_noise_per_epoch": bool(resample_noise_per_epoch)} if ds_cls is OnlineRenderDataset else {}),
        quiet=True,
        assert_labels=assert_labels,
        encoding_mode=encoding_mode,
        sigma_rf=sigma_rf,
        rf_normalization=rf_normalization,
        sigma_rf_noise=sigma_rf_noise,
        rf_noise_per_token=rf_noise_per_token,
        noise_mode=noise_mode,
        noise_rho=noise_rho,
    )
    ds_val = ds_cls(
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
        add_bos=bool(add_bos),
        eos_mode=str(eos_mode),
        sigma_other_noise=sigma_other_noise,
        p_other_noise=p_other_noise,
        sigma_silence_noise=sigma_silence_noise,
        **({"resample_noise_per_epoch": False} if ds_cls is OnlineRenderDataset else {}),
        quiet=True,
        assert_labels=assert_labels,
        encoding_mode=encoding_mode,
        sigma_rf=sigma_rf,
        rf_normalization=rf_normalization,
        sigma_rf_noise=sigma_rf_noise,
        rf_noise_per_token=rf_noise_per_token,
        noise_mode=noise_mode,
        noise_rho=noise_rho,
    )
    train_ds = Subset(ds_train, train_idx)
    val_ds = Subset(ds_val, val_idx)

    pin = (device.type == "cuda") if pin_memory is None else bool(pin_memory)
    loader_extra: Dict[str, Any] = {}
    if int(num_workers) > 0:
        loader_extra["persistent_workers"] = bool(persistent_workers)
        loader_extra["prefetch_factor"] = max(1, int(prefetch_factor))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        **loader_extra,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        **loader_extra,
    )
    return ds_train, train_loader, val_loader


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
    add_bos: bool,
    eos_mode: str,
    sigma_other_noise: float,
    p_other_noise: float,
    sigma_silence_noise: float,
    resample_noise_per_epoch: bool,
    n_blocks: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    assert_labels: bool = False,
    encoding_mode: str = "onehot",
    sigma_rf: float = 1.0,
    rf_normalization: str = "peak",
    sigma_rf_noise: float = 0.0,
    rf_noise_per_token: bool = True,
    noise_mode: str = "per_token",
    noise_rho: float = 0.0,
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
        add_bos=bool(add_bos),
        eos_mode=str(eos_mode),
        sigma_other_noise=sigma_other_noise,
        p_other_noise=p_other_noise,
        sigma_silence_noise=sigma_silence_noise,
        resample_noise_per_epoch=bool(resample_noise_per_epoch),
        quiet=False,
        assert_labels=assert_labels,
        encoding_mode=encoding_mode,
        sigma_rf=sigma_rf,
        rf_normalization=rf_normalization,
        sigma_rf_noise=sigma_rf_noise,
        rf_noise_per_token=rf_noise_per_token,
        noise_mode=noise_mode,
        noise_rho=noise_rho,
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
        add_bos=bool(getattr(args, "add_bos", False)),
        eos_mode=str(getattr(args, "eos_mode", "separate")),
        sigma_other_noise=float(args.sigma_other_noise),
        p_other_noise=float(args.p_other_noise),
        sigma_silence_noise=float(args.sigma_silence_noise),
        resample_noise_per_epoch=False,
        n_blocks=int(getattr(args, "debug_n_blocks", 16)),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        assert_labels=True,
        encoding_mode=str(getattr(args, "encoding_mode", "onehot")),
        sigma_rf=float(getattr(args, "sigma_rf", 1.0)),
        rf_normalization=str(getattr(args, "rf_normalization", "peak")),
        sigma_rf_noise=float(getattr(args, "sigma_rf_noise", 0.0)),
        rf_noise_per_token=bool(getattr(args, "rf_noise_per_token", True)),
        noise_mode=str(getattr(args, "noise_mode", "per_token")),
        noise_rho=float(getattr(args, "noise_rho", 0.0)),
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
    print("add_eos", bool(args.add_eos))
    print("eos_mode", str(getattr(args, "eos_mode", "separate")))
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
        use_event_head=bool(getattr(args, "use_event_head", False)),
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
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    set_all_seeds(int(args.seed))

    device = resolve_device(args.device)
    print(f"[device] using: {device}")

    selection_metric_name = normalize_selection_metric_name(
        str(getattr(args, "model_selection_metric", "val_loss"))
    )
    early_stop_metric_raw = str(getattr(args, "early_stop_metric", "auto"))
    early_stop_metric_name = (
        selection_metric_name
        if early_stop_metric_raw.strip().lower() in {"", "auto"}
        else normalize_selection_metric_name(early_stop_metric_raw)
    )
    stage_restore_mode = str(getattr(args, "stage_restore_mode", "auto")).strip().lower()
    if stage_restore_mode == "auto":
        stage_restore_mode = "selection_metric" if selection_metric_name != "val_loss" else "behavior_valid_then_loss"
    valid_restore_modes = {"selection_metric", "behavior_valid_then_loss", "loss_only"}
    if stage_restore_mode not in valid_restore_modes:
        raise ValueError(f"Unknown stage_restore_mode: {stage_restore_mode}")
    if (
        selection_metric_name in {"human_rt_r2", "delta_r2"}
        or early_stop_metric_name in {"human_rt_r2", "delta_r2"}
    ) and not str(getattr(args, "human_csv_for_rt_corr", "")).strip():
        raise ValueError(
            "behavioral metric requires --human_csv_for_rt_corr pointing to the participant/full validation human trial CSV."
        )
    print(
        "[selection] "
        f"model_selection_metric={selection_metric_name} "
        f"early_stop_metric={early_stop_metric_name} "
        f"stage_restore_mode={stage_restore_mode}"
    )

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
        add_bos=bool(getattr(args, "add_bos", False)),
        eos_mode=str(getattr(args, "eos_mode", "separate")),
        sigma_other_noise=float(args.sigma_other_noise),
        p_other_noise=float(args.p_other_noise),
        sigma_silence_noise=float(args.sigma_silence_noise),
        resample_noise_per_epoch=bool(getattr(args, "resample_noise_per_epoch", False)),
        quiet=False,
        encoding_mode=str(getattr(args, "encoding_mode", "onehot")),
        sigma_rf=float(getattr(args, "sigma_rf", 1.0)),
        rf_normalization=str(getattr(args, "rf_normalization", "peak")),
        sigma_rf_noise=float(getattr(args, "sigma_rf_noise", 0.0)),
        rf_noise_per_token=bool(getattr(args, "rf_noise_per_token", True)),
        noise_mode=str(getattr(args, "noise_mode", "per_token")),
        noise_rho=float(getattr(args, "noise_rho", 0.0)),
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
        num_classes=supervision_num_classes(str(getattr(args, "supervision_mode", "post_deviant"))),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        layer_norm=bool(args.layer_norm),
        hidden_noise_std=float(getattr(args, "hidden_noise_std", 0.0)),
        use_stop_head=bool(getattr(args, "use_stop_head", False)),
        use_event_head=bool(getattr(args, "use_event_head", False)),
    )
    model = PredictiveGRU(cfg).to(device)
    amp_enabled = bool(getattr(args, "amp", False)) and (str(device.type) == "cuda")
    amp_dtype = str(getattr(args, "amp_dtype", "float16"))
    grad_scaler = None
    if str(device.type) == "cuda":
        try:
            grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch_global = 1
    best_val = metric_initial_best(selection_metric_name)
    best_epoch = 0
    resumed_optimizer_global_step = 0
    epoch_save_gate_open = False
    first_epoch_save_gate_epoch: Optional[int] = None
    stop_after_valid_trigger_epoch: Optional[int] = None
    
    # 1) init_from: weights only, for cross-condition finetuning / sweep
    if getattr(args, "init_from", ""):
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[init_from] loaded model weights only (strict=False): {args.init_from}")
        print("[init_from] optimizer / epoch / best_val are RESET for a new run.")
    
    # 2) resume: full resume, only for continuing the same run
    elif getattr(args, "resume", ""):
        freeze_variant = str(getattr(args, "freeze_variant", "full_finetune")).strip().lower()
        if freeze_variant not in {"", "none", "full_finetune"}:
            raise ValueError(
                "--resume cannot be combined with freeze variants. "
                "Use --init_from for controlled finetuning with --freeze_variant."
            )
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        start_epoch_global = int(ckpt.get("epoch_global", ckpt.get("epoch", 0))) + 1
        best_val = float(ckpt.get("best_val", best_val))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        resumed_optimizer_global_step = int(ckpt.get("optimizer_global_step", ckpt.get("global_opt_step", 0)) or 0)
        print(
            f"[resume] loaded full state: {args.resume} "
            f"start_epoch_global={start_epoch_global} "
            f"optimizer_global_step={resumed_optimizer_global_step}"
        )

    freeze_summary = configure_trainable_parameters(
        model,
        str(getattr(args, "freeze_variant", "full_finetune")),
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after applying --freeze_variant.")
    optim = torch.optim.AdamW(
        trainable_params,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    if getattr(args, "resume", ""):
        optim_state = ckpt.get("optim_state", ckpt.get("optimizer_state"))
        if optim_state is not None:
            optim.load_state_dict(optim_state)
        if bool(getattr(args, "resume_override_lr", False)):
            for pg in optim.param_groups:
                pg["lr"] = float(args.lr)
            print(f"[resume] overriding restored optimizer lr with args.lr={float(args.lr):.6g}")
    freeze_summary_path = run_dir / "logs" / "freeze_summary.json"
    freeze_summary_path.write_text(json.dumps(freeze_summary, indent=2), encoding="utf-8")
    print(
        "[freeze] "
        f"variant={freeze_summary['freeze_variant']} "
        f"trainable={freeze_summary['trainable_param_count']} "
        f"frozen={freeze_summary['frozen_param_count']} "
        f"core_trainable={freeze_summary['core_trainable_param_count']} "
        f"head_trainable={freeze_summary['head_trainable_param_count']} "
        f"other_trainable={freeze_summary['other_trainable_param_count']}"
    )

    jsonl_path = run_dir / "logs" / "metrics.jsonl"
    csv_path = run_dir / "logs" / "metrics.csv"
    step_metrics_csv = run_dir / "logs" / "checkpoint_step_metrics.csv"
    behavior_step_eval_csv = run_dir / "logs" / "behavior_step_eval.csv"
    window_diag_csv = run_dir / "logs" / "window_diagnostics.csv"
    window_diag_by_pos_csv = run_dir / "logs" / "window_diagnostics_by_position.csv"
    pre_p4_probability_audit_csv = run_dir / "pre_p4_probability_audit.csv"
    rf_diag_csv = run_dir / "logs" / "rf_ambiguity_diagnostics.csv"
    train_rt_diag_csv = run_dir / "logs" / "train_rt_readout_diagnostics.csv"
    train_rt_diag_jsonl = run_dir / "logs" / "train_rt_readout_diagnostics.jsonl"
    epoch_rt_diag_csv = run_dir / "epoch_rt_diagnostics.csv"
    epoch_window_by_pos_csv = run_dir / "epoch_window_by_pos.csv"
    collapse_diag_csv = run_dir / "collapse_diagnostics.csv"
    stage_best_summary_csv = run_dir / "stage_best_summary.csv"
    epoch_traj_dir = run_dir / "epoch_trajectory_diagnostics"
    checkpoints_dir = run_dir / "checkpoints"
    save_epoch_after_valid = bool(getattr(args, "save_epoch_after_valid", False))
    save_steps_after_valid = bool(getattr(args, "save_steps_after_valid", False))
    if bool(getattr(args, "save_epoch_checkpoints", False)):
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    epoch_ckpt_dir = run_dir / str(getattr(args, "epoch_ckpt_dir", "checkpoints_by_epoch"))
    if bool(getattr(args, "save_every_epoch", True)) or save_epoch_after_valid:
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
    step_ckpt_dir = run_dir / str(getattr(args, "step_ckpt_dir", "checkpoints_by_step"))
    exact_save_steps = set(parse_list_of_ints(str(getattr(args, "save_steps", ""))))
    max_exact_save_step = max(exact_save_steps) if exact_save_steps else None
    if (
        (bool(getattr(args, "save_step_checkpoints", False)) and int(getattr(args, "save_step_checkpoints_every", 0)) > 0)
        or len(exact_save_steps) > 0
    ):
        step_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if bool(getattr(args, "save_every_epoch", True)) or save_epoch_after_valid or bool(getattr(args, "rt_candidate_save_json", True)):
        epoch_traj_dir.mkdir(parents=True, exist_ok=True)

    def _format_ckpt_float(x: float, decimals: int = 2) -> str:
        return f"{float(x):.{int(decimals)}f}"

    def _format_ckpt_rf(x: float) -> str:
        xv = float(x)
        if abs(xv - round(xv)) < 1e-8:
            return str(int(round(xv)))
        return f"{xv:g}"

    def _make_explicit_ckpt_tag() -> str:
        encoding_mode = str(getattr(args, "encoding_mode", "onehot"))
        sigma_input_noise = float(getattr(args, "sigma_input_noise", getattr(args, "sigma_other_noise", 0.0)))
        if encoding_mode == "gaussian_rf":
            sigma_rf_width = float(getattr(args, "sigma_rf_width", getattr(args, "sigma_rf", 1.0)))
            return f"rf{_format_ckpt_rf(sigma_rf_width)}_noise{_format_ckpt_float(sigma_input_noise)}"
        return f"{encoding_mode}_noise{_format_ckpt_float(sigma_input_noise)}"

    explicit_ckpt_tag = _make_explicit_ckpt_tag()
    explicit_best_path = run_dir / f"best_{explicit_ckpt_tag}.pt"
    explicit_last_path = run_dir / f"last_{explicit_ckpt_tag}.pt"

    header = [
        "epoch_global", "stage", "isi_ms", "token_ms",
        "train_total_loss", "train_end_loss", "train_weighted_end_loss", "train_token_loss", "train_online_loss_sum", "train_online_loss_mean", "train_online_valid_n", "train_old_token_loss", "train_windowed_correct_ce_loss", "train_event_deviance_ce_loss", "train_anti_commit_loss", "train_acc", "train_f1_macro", "train_auc_ovr",
        "train_window_acc", "train_window_f1", "train_window_auc", "train_window_acc_first_token", "train_window_acc_last_token", "train_window_acc_mean_token", "train_window_auc_mean_token",
        "train_window_p_correct_mean", "train_window_p_correct_std", "train_window_p_correct_p10", "train_window_p_correct_p50", "train_window_p_correct_p90", "train_early_p_correct_mean", "train_window_p_correct_end", "train_window_p_correct_max", "train_window_p_correct_auc",
        "train_window_prediction_mean", "train_window_prediction_std", "train_window_prediction_min", "train_window_prediction_max",
        "train_window_y_true_frac_class0", "train_window_y_true_frac_class1", "train_window_y_true_frac_class2",
        "train_end_acc", "train_end_f1", "train_end_auc",
        "train_window_acc_P4", "train_window_acc_P5", "train_window_acc_P6",
        "train_window_f1_P4", "train_window_f1_P5", "train_window_f1_P6",
        "train_window_auc_P4", "train_window_auc_P5", "train_window_auc_P6",
        "train_window_p_correct_mean_P4", "train_window_p_correct_mean_P5", "train_window_p_correct_mean_P6",
        "train_window_p_correct_end_P4", "train_window_p_correct_end_P5", "train_window_p_correct_end_P6",
        "train_window_token_loss_P4", "train_window_token_loss_P5", "train_window_token_loss_P6",
        "train_mean_rt_tokens", "train_mean_rt_ms", "train_rt_found", "train_rt_miss",
        "train_rt_not_first", "train_rt_negative", "train_rt_not_first_rate",
        "train_online_decision_cost",
        "train_online_decision_loss", "train_online_ce_loss",
        "train_mean_p_stop", "train_mean_no_response_prob", "train_mean_expected_decision_time_ms",
        "train_expected_found_prob", "train_expected_rt_logged_ms",
        "train_expected_rt_from_deviant_onset_ms", "train_expected_rt_from_deviant_end_ms",
        "train_proportion_negative_rt", "train_proportion_decisions_before_deviant_end",
        "train_phase_name", "train_effective_online_loss_weight",
        "train_lambda_online", "train_lambda_end_current", "train_weighted_token_loss", "train_weighted_online_decision_loss", "train_weighted_online_ce_loss",
                "train_aux_token_ce_loss", "train_weighted_aux_token_ce_loss",
                "train_pre_p4_uniformity_loss", "train_weighted_pre_p4_uniformity_loss",
                "train_pre_evidence_uniform_kl_loss", "train_weighted_pre_evidence_uniform_kl_loss",
                "train_preDevCostRaw", "train_wPreDevCost", "train_preDevCostMode", "train_preDevCostWeight", "train_preDevCostMargin", "train_preDevCostScale",
        "train_anti_immediate_stop_loss", "train_stop_entropy_bonus", "train_stop_prior_loss",
        "train_sampled_mean_rt_logged_ms", "train_last_token_acc", "train_deviant_end_acc",
        "val_total_loss", "val_end_loss", "val_weighted_end_loss", "val_token_loss", "val_online_loss_sum", "val_online_loss_mean", "val_online_valid_n", "val_old_token_loss", "val_windowed_correct_ce_loss", "val_event_deviance_ce_loss", "val_anti_commit_loss", "val_acc", "val_f1_macro", "val_auc_ovr",
        "val_window_acc", "val_window_f1", "val_window_auc", "val_window_acc_first_token", "val_window_acc_last_token", "val_window_acc_mean_token", "val_window_auc_mean_token",
        "val_window_p_correct_mean", "val_window_p_correct_std", "val_window_p_correct_p10", "val_window_p_correct_p50", "val_window_p_correct_p90", "val_early_p_correct_mean", "val_window_p_correct_end", "val_window_p_correct_max", "val_window_p_correct_auc",
        "val_window_prediction_mean", "val_window_prediction_std", "val_window_prediction_min", "val_window_prediction_max",
        "val_window_y_true_frac_class0", "val_window_y_true_frac_class1", "val_window_y_true_frac_class2",
        "val_end_acc", "val_end_f1", "val_end_auc",
        "val_window_acc_P4", "val_window_acc_P5", "val_window_acc_P6",
        "val_window_f1_P4", "val_window_f1_P5", "val_window_f1_P6",
        "val_window_auc_P4", "val_window_auc_P5", "val_window_auc_P6",
        "val_window_p_correct_mean_P4", "val_window_p_correct_mean_P5", "val_window_p_correct_mean_P6",
        "val_window_p_correct_end_P4", "val_window_p_correct_end_P5", "val_window_p_correct_end_P6",
        "val_window_token_loss_P4", "val_window_token_loss_P5", "val_window_token_loss_P6",
        "val_mean_rt_tokens", "val_mean_rt_ms", "val_rt_found", "val_rt_miss",
        "val_rt_not_first", "val_rt_negative", "val_rt_not_first_rate",
        "val_online_decision_cost",
        "val_online_decision_loss", "val_online_ce_loss",
        "val_mean_p_stop", "val_mean_no_response_prob", "val_mean_expected_decision_time_ms",
        "val_expected_found_prob", "val_expected_rt_logged_ms",
        "val_expected_rt_from_deviant_onset_ms", "val_expected_rt_from_deviant_end_ms",
        "val_proportion_negative_rt", "val_proportion_decisions_before_deviant_end",
        "val_phase_name", "val_effective_online_loss_weight",
        "val_lambda_online", "val_lambda_end_current", "val_weighted_token_loss", "val_weighted_online_decision_loss", "val_weighted_online_ce_loss",
                "val_aux_token_ce_loss", "val_weighted_aux_token_ce_loss",
                "val_pre_p4_uniformity_loss", "val_weighted_pre_p4_uniformity_loss",
                "val_pre_evidence_uniform_kl_loss", "val_weighted_pre_evidence_uniform_kl_loss",
                "val_preDevCostRaw", "val_wPreDevCost", "val_preDevCostMode", "val_preDevCostWeight", "val_preDevCostMargin", "val_preDevCostScale",
        "val_anti_immediate_stop_loss", "val_stop_entropy_bonus", "val_stop_prior_loss",
        "val_sampled_mean_rt_logged_ms", "val_last_token_acc", "val_deviant_end_acc",
        "collapse_invalid", "collapse_reason",
        "behavior_meanRT_ms", "behavior_batch_meanRT_tokens", "behavior_batch_meanRT_ms",
        "behavior_floor0", "behavior_floor5", "behavior_P4", "behavior_P5", "behavior_P6", "behavior_ordering",
        "behavior_online_loss_mean", "behavior_p_correct_p50", "behavior_p_correct_p90", "behavior_early_p_correct_mean",
        "human_rt_corr_r", "human_rt_corr_r2", "human_rt_corr_p", "predicted_dependency_r", "observed_dependency_r",
        "model_selection_metric", "model_selection_metric_value", "early_stop_metric", "early_stop_metric_value",
        "loss_best_epoch", "behavior_valid_best_epoch",
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
    best_target_val = metric_initial_best(selection_metric_name)
    best_target_epoch = 0
    best_target_path = run_dir / f"best_isi{best_target_isi}.pt"

    history: List[Dict[str, Any]] = []
    window_diag_rows: List[Dict[str, Any]] = []
    window_diag_by_pos_rows: List[Dict[str, Any]] = []
    rf_diag_rows: List[Dict[str, Any]] = []
    train_rt_diag_rows: List[Dict[str, Any]] = []
    epoch_rt_diag_rows_all: List[Dict[str, Any]] = []
    epoch_window_by_pos_rows_all: List[Dict[str, Any]] = []
    pre_p4_probability_audit_rows_all: List[Dict[str, Any]] = []
    collapse_diag_rows_all: List[Dict[str, Any]] = []
    stage_best_summary_rows_all: List[Dict[str, Any]] = []
    t_run0 = time.time()
    epoch_global = start_epoch_global - 1
    optimizer_step_state = {"count": int(resumed_optimizer_global_step)}
    step_save_gate_open = False
    first_step_save_gate_epoch: Optional[int] = None
    first_step_save_gate_step: Optional[int] = None
    pending_resume_step_estimate = bool(
        getattr(args, "resume", "") and int(resumed_optimizer_global_step) <= 0 and int(start_epoch_global) > 1
    )
    first_batch_debug_done = False
    best_rt_candidate_score = float("-inf")
    best_rt_candidate_summary: Dict[str, Any] = {
        "selected_epoch": None,
        "isi_ms": None,
        "rt_candidate_score": None,
        "eligible": False,
        "notes": "no eligible epoch yet",
    }

    def _build_step_checkpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        global_opt_step = int(payload["global_opt_step"])
        step_ckpt = {
            "model_state": payload["model"].state_dict(),
            "optim_state": payload["optimizer"].state_dict() if payload.get("optimizer") is not None else None,
            "args": vars(args),
            "args_dict": vars(args),
            "epoch_global": int(payload["epoch_global"]),
            "optimizer_global_step": global_opt_step,
            "step_in_epoch": int(payload["step_in_epoch"]),
            "phase_name": str(payload.get("phase_name", "")),
            "step_metrics": {
                "end_loss": float(payload.get("end_loss", float("nan"))),
                "token_loss": float(payload.get("token_loss", float("nan"))),
                "anti_commit_loss": float(payload.get("anti_commit_loss", float("nan"))),
                "total_loss": float(payload.get("total_loss", float("nan"))),
                "end_acc": float(payload.get("end_acc", float("nan"))),
            },
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return step_ckpt

    def _append_step_checkpoint_metrics(payload: Dict[str, Any], checkpoint_path: Path, save_kind: str) -> None:
        """Record batch-level loss at the exact optimizer step used for a saved checkpoint."""
        row = {
            "global_step": int(payload["global_opt_step"]),
            "checkpoint_step": int(payload["global_opt_step"]),
            "epoch_global": int(payload["epoch_global"]),
            "step_in_epoch": int(payload["step_in_epoch"]),
            "phase_name": str(payload.get("phase_name", "")),
            "save_kind": str(save_kind),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "train_total_loss_batch": float(payload.get("total_loss", float("nan"))),
            "train_token_loss_batch": float(payload.get("token_loss", float("nan"))),
            "train_end_loss_batch": float(payload.get("end_loss", float("nan"))),
            "train_anti_commit_loss_batch": float(payload.get("anti_commit_loss", float("nan"))),
            "train_end_acc_batch": float(payload.get("end_acc", float("nan"))),
            "lr": float(optim.param_groups[0].get("lr", float("nan"))) if optim.param_groups else float("nan"),
            "isi_ms": int(payload.get("isi_ms", 0) or 0),
            "token_ms": int(getattr(args, "token_ms", 0)),
            "sigma_rf": float(getattr(args, "sigma_rf", float("nan"))),
            "sigma_rf_noise": float(getattr(args, "sigma_rf_noise", float("nan"))),
            "sigma_other_noise": float(getattr(args, "sigma_other_noise", float("nan"))),
        }
        fieldnames = list(row.keys())
        step_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not step_metrics_csv.exists()
        with step_metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_exact_step_checkpoint(payload: Dict[str, Any]) -> None:
        global_opt_step = int(payload["global_opt_step"])
        if save_steps_after_valid and first_step_save_gate_step is not None:
            relative_step = global_opt_step - int(first_step_save_gate_step)
            out_path = step_ckpt_dir / f"ckpt_after_valid_step{relative_step:06d}.pt"
        else:
            out_path = step_ckpt_dir / f"ckpt_step{global_opt_step:06d}.pt"
        step_ckpt = _build_step_checkpoint(payload)
        if first_step_save_gate_step is not None:
            step_ckpt["step_save_gate_open"] = bool(step_save_gate_open)
            step_ckpt["first_step_save_gate_epoch"] = (
                int(first_step_save_gate_epoch) if first_step_save_gate_epoch is not None else None
            )
            step_ckpt["first_step_save_gate_step"] = int(first_step_save_gate_step)
            step_ckpt["relative_to_valid_step"] = int(global_opt_step - int(first_step_save_gate_step))
        torch.save(step_ckpt, out_path)
        _append_step_checkpoint_metrics(payload, out_path, "exact")
        print(f"[step_ckpt exact] saved {out_path.relative_to(run_dir)}")

    def _save_step_checkpoint(payload: Dict[str, Any]) -> None:
        global_opt_step = int(payload["global_opt_step"])
        if save_steps_after_valid:
            if not step_save_gate_open or first_step_save_gate_step is None:
                return
            relative_step = global_opt_step - int(first_step_save_gate_step)
            if relative_step in exact_save_steps:
                _save_exact_step_checkpoint(payload)
                if max_exact_save_step is not None and relative_step >= int(max_exact_save_step):
                    optimizer_step_state["request_stop"] = True
                    optimizer_step_state["request_stop_reason"] = (
                        f"reached_final_relative_save_step:{int(relative_step)}"
                    )
                    print(
                        f"[step_ckpt_stop] reached final relative save step "
                        f"{int(relative_step)}; stopping after current epoch."
                    )
        else:
            if global_opt_step in exact_save_steps:
                _save_exact_step_checkpoint(payload)
        if not bool(getattr(args, "save_step_checkpoints", False)):
            return
        save_every = int(getattr(args, "save_step_checkpoints_every", 0))
        if save_every <= 0:
            return
        if save_steps_after_valid:
            if not step_save_gate_open or first_step_save_gate_step is None:
                return
            relative_step = global_opt_step - int(first_step_save_gate_step)
            if relative_step <= 0 or relative_step % save_every != 0:
                return
        else:
            if global_opt_step % save_every != 0:
                return
        step_ckpt = _build_step_checkpoint(payload)
        if save_steps_after_valid and first_step_save_gate_step is not None:
            out_path = step_ckpt_dir / f"step_after_valid_{relative_step:06d}.pt"
            step_ckpt["step_save_gate_open"] = bool(step_save_gate_open)
            step_ckpt["first_step_save_gate_epoch"] = (
                int(first_step_save_gate_epoch) if first_step_save_gate_epoch is not None else None
            )
            step_ckpt["first_step_save_gate_step"] = int(first_step_save_gate_step)
            step_ckpt["relative_to_valid_step"] = int(relative_step)
        else:
            out_path = step_ckpt_dir / f"step_{global_opt_step:06d}.pt"
        torch.save(step_ckpt, out_path)
        _append_step_checkpoint_metrics(payload, out_path, "periodic")
        print(f"[step_ckpt] saved {out_path.relative_to(run_dir)}")

    def _append_behavior_step_eval_row(row: Dict[str, Any]) -> None:
        behavior_step_eval_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        write_header = not behavior_step_eval_csv.exists()
        with behavior_step_eval_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    if 0 in exact_save_steps:
        _save_exact_step_checkpoint(
            {
                "epoch_global": int(start_epoch_global - 1),
                "global_opt_step": 0,
                "step_in_epoch": 0,
                "phase_name": "pretrain_init",
                "isi_ms": int(args.isi_schedule[0]) if len(args.isi_schedule) > 0 else 0,
                "end_loss": float("nan"),
                "token_loss": float("nan"),
                "anti_commit_loss": float("nan"),
                "total_loss": float("nan"),
                "end_acc": float("nan"),
                "model": model,
                "optimizer": optim,
            }
        )

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

        stage_best_val = metric_initial_best(early_stop_metric_name)
        stage_bad_count = 0
        stage_best_state: Optional[Dict[str, torch.Tensor]] = None
        stage_best_optim: Optional[Dict[str, Any]] = None
        stage_best_epoch_global: Optional[int] = None
        stage_best_path = run_dir / f"best_isi{int(isi_ms)}.pt"
        stage_best_metric_value = metric_initial_best(selection_metric_name)
        stage_loss_best_val = float("inf")
        stage_loss_best_state: Optional[Dict[str, torch.Tensor]] = None
        stage_loss_best_optim: Optional[Dict[str, Any]] = None
        stage_loss_best_epoch_global: Optional[int] = None
        stage_loss_best_metrics: Optional[Dict[str, Any]] = None
        stage_loss_best_behavior: Optional[Dict[str, Any]] = None
        stage_loss_best_path = run_dir / f"best_isi{int(isi_ms)}_loss_best.pt"
        stage_behavior_best_val = float("inf")
        stage_behavior_best_state: Optional[Dict[str, torch.Tensor]] = None
        stage_behavior_best_optim: Optional[Dict[str, Any]] = None
        stage_behavior_best_epoch_global: Optional[int] = None
        stage_behavior_best_metrics: Optional[Dict[str, Any]] = None
        stage_behavior_best_behavior: Optional[Dict[str, Any]] = None
        stage_behavior_best_fallback_score = (-1, -1, -1, float("-inf"), float("-inf"), float("-inf"))
        stage_behavior_best_found = False
        stage_behavior_best_path = run_dir / f"best_isi{int(isi_ms)}_behavior_valid.pt"
        stage_collapse_epoch_first: Optional[int] = None
        stage_collapse_epoch_count = 0
        stage_reached_threshold = False  # 标记是否达到阈值
        optimizer_step_state.pop("request_stop", None)
        optimizer_step_state.pop("request_stop_reason", None)
        behavior_eval_every_steps = int(getattr(args, "behavior_eval_every_steps", 0))
        behavior_patience_evals = int(getattr(args, "behavior_patience_evals", 0))
        stage_behavior_step_tracker: Dict[str, Any] = {
            "best_metric": metric_initial_best(selection_metric_name),
            "best_step": None,
            "best_epoch": None,
            "best_found_rate": float("nan"),
            "bad_evals": 0,
            "n_evals": 0,
            "activated": False,
        }
        stage_behavior_step_best_path = run_dir / f"best_isi{int(isi_ms)}_{selection_metric_name}_step.pt"

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
            add_bos=bool(getattr(args, "add_bos", False)),
            eos_mode=str(getattr(args, "eos_mode", "separate")),
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
            encoding_mode=str(getattr(args, "encoding_mode", "onehot")),
            sigma_rf=float(getattr(args, "sigma_rf", 1.0)),
            rf_normalization=str(getattr(args, "rf_normalization", "peak")),
            sigma_rf_noise=float(getattr(args, "sigma_rf_noise", 0.0)),
            rf_noise_per_token=bool(getattr(args, "rf_noise_per_token", True)),
            noise_mode=str(getattr(args, "noise_mode", "per_token")),
            noise_rho=float(getattr(args, "noise_rho", 0.0)),
            use_prerendered_tokens=bool(getattr(args, "use_prerendered_tokens", False)),
            pin_memory=bool(getattr(args, "pin_memory", True)),
            persistent_workers=bool(getattr(args, "persistent_workers", True)),
            prefetch_factor=int(getattr(args, "prefetch_factor", 4)),
        )
        if pending_resume_step_estimate:
            estimated_steps = int(max(0, start_epoch_global - 1) * len(train_loader))
            optimizer_step_state["count"] = estimated_steps
            print(
                "[resume] checkpoint did not store optimizer_global_step; "
                f"estimating from completed epochs: {start_epoch_global - 1} * {len(train_loader)} = {estimated_steps}"
            )
            pending_resume_step_estimate = False

        end_preview = infer_end_indices_from_T(
            int(ds.T),
            trials_per_block=10,
            end_offset_from_trial_end=(1 if bool(getattr(args, "add_bos", False)) else 0),
        )
        print(
            f"[data@isi={isi_ms}] token_ms={ds.token_ms} trial_T_ms={ds.trial_T_ms} "
            f"trial_T_tokens={ds.trial_T_tokens} T={ds.T} end_idx preview: "
            f"{end_preview[:3].tolist()} ... {end_preview[-3:].tolist()}"
        )
        if str(getattr(args, "supervision_mode", "post_deviant")) == "strict_online_p4":
            p4_tok = int(3 * (int(ds.tone_T) + int(ds.isi_T)))
            print(
                "[strict_online_p4 audit] "
                f"output_dim={int(cfg.num_classes)} "
                f"loss_onset_token={p4_tok} loss_onset_ms={p4_tok * int(ds.token_ms)} "
                f"response_start_token={p4_tok} response_start_ms={p4_tok * int(ds.token_ms)} "
                "rt_reference_token=true_deviant_onset"
            )
        rf_diag_stage = compute_rf_ambiguity_diagnostics(
            freqs_blocks=ds.X,
            y_pos_456=ds.Y.long(),
            encoding_cfg=ds.encoding_cfg,
        )

        normalized_token_loss_mode = _normalize_token_loss_mode(str(args.token_loss_mode))
        if normalized_token_loss_mode == "windowed_correct_ce":
            _ws, _we, _clipped = compute_correct_ce_window_bounds_in_trial(
                y_pos_456=ds.Y.long(),
                trial_T_tokens=int(ds.trial_T_tokens),
                tone_T=int(ds.tone_T),
                isi_T=int(ds.isi_T),
                correct_ce_window=str(getattr(args, "correct_ce_window", "deviant_onset_to_next_tone_offset")),
            )
            dbg = get_correct_ce_window_debug_summary(
                y_pos_456=ds.Y.long(),
                trial_T_tokens=int(ds.trial_T_tokens),
                tone_T=int(ds.tone_T),
                isi_T=int(ds.isi_T),
                correct_ce_window=str(getattr(args, "correct_ce_window", "deviant_onset_to_next_tone_offset")),
            )
            print(
                "[windowed_correct_ce] "
                f"token_loss_mode={str(args.token_loss_mode)} "
                f"effective_mode={normalized_token_loss_mode} "
                f"correct_ce_window={str(getattr(args, 'correct_ce_window', 'deviant_onset_to_next_tone_offset'))} "
                f"correct_ce_weighting={str(getattr(args, 'correct_ce_weighting', 'equal'))} "
                f"lambda_end={float(getattr(args, 'lambda_end', float(getattr(args, 'end_loss_weight', 1.0)))):.6f} "
                f"lambda_online={float(getattr(args, 'lambda_online', float(args.lambda_token))):.6f} "
                f"token_ms={int(ds.token_ms)} tone_ms={int(args.tone_ms)} isi_ms={int(isi_ms)} "
                f"example_pos={dbg['example_deviant_position']} "
                f"dev_on={dbg['example_deviant_onset_token']} dev_off={dbg['example_deviant_offset_token']} "
                f"next_on={dbg['example_next_tone_onset_token']} next_off={dbg['example_next_tone_offset_token']} "
                f"second_next_on={dbg['example_second_next_tone_onset_token']} second_next_off={dbg['example_second_next_tone_offset_token']} "
                f"window_start={dbg['example_window_start']} window_end_exclusive={dbg['example_window_end_exclusive']} "
                f"n_window_tokens={dbg['example_n_window_tokens']} clipped_trials={int(_clipped.sum().item())}"
            )
        else:
            print(
                "[token_loss_mode] "
                f"token_loss_mode={str(args.token_loss_mode)} effective_mode={normalized_token_loss_mode} "
                f"lambda_end={float(getattr(args, 'lambda_end', float(getattr(args, 'end_loss_weight', 1.0)))):.6f} "
                f"lambda_online={float(getattr(args, 'lambda_online', float(args.lambda_token))):.6f} token_ms={int(ds.token_ms)} "
                f"tone_ms={int(args.tone_ms)} isi_ms={int(isi_ms)}"
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
                phase_online_ce = bool(getattr(args, "online_decision_training", False)) and (not bool(getattr(args, "disable_aux_online_ce", False)))
                phase_online_ce_weight = float(getattr(args, "online_ce_weight", 0.1))
                phase_use_stop_head = bool(getattr(args, "use_stop_head", False))
                phase_decision_cost_mode = str(getattr(args, "decision_cost_mode", "expected_cost_softmin"))
                phase_anti_immediate = bool(getattr(args, "anti_immediate_stop", False))
                phase_stop_entropy_w = float(getattr(args, "stop_entropy_weight", 0.0))
                phase_stop_prior_w = float(getattr(args, "stop_prior_weight", 0.0))
                phase_lr = float(getattr(args, "lr", 3e-4))

            if bool(getattr(args, "two_phase_training", False)):
                warm_ep_total = int(getattr(args, "classifier_warmup_epochs", 0))
                stop_ft_total = getattr(args, "stop_finetune_epochs", None)
                if stop_ft_total is None:
                    stage_epochs_total = max(1, int(args.epochs_per_isi))
                else:
                    stage_epochs_total = max(1, int(warm_ep_total) + int(stop_ft_total))
            else:
                stage_epochs_total = max(1, int(args.epochs_per_isi))
            stage_progress_0based = float(max(0, int(_e) - 1))
            lambda_online_curr = float(getattr(args, "lambda_online", getattr(args, "lambda_token", 1.0)))
            lambda_end_curr = float(getattr(args, "lambda_end", getattr(args, "end_loss_weight", 0.0)))
            if bool(getattr(args, "lambda_end_anneal", False)):
                half_stage = max(1e-12, 0.5 * float(max(1, stage_epochs_total - 1)))
                anneal_scale = max(0.0, 1.0 - (stage_progress_0based / half_stage))
                lambda_end_curr *= anneal_scale

            if int(epoch_global) == 1 and int(_e) == 1:
                _eff_token_loss_mode = _normalize_token_loss_mode(str(args.token_loss_mode))
                _token_window_label = (
                    str(getattr(args, "correct_ce_window", "deviant_onset_to_next_tone_offset"))
                    if _eff_token_loss_mode == "windowed_correct_ce"
                    else (
                        "p4_onset_to_trial_end_causal_targets"
                        if _eff_token_loss_mode == "strict_p4_causal_ce"
                        else (
                            "all_tone_tokens"
                            if _eff_token_loss_mode == "event_deviance_ce"
                            else "deviant_onset_to_before_next_standard_onset"
                        )
                    )
                )
                _token_weighting_label = (
                    str(getattr(args, "correct_ce_weighting", "equal"))
                    if _eff_token_loss_mode == "windowed_correct_ce"
                    else (
                        "causal_soft_targets"
                        if _eff_token_loss_mode == "strict_p4_causal_ce"
                        else (
                            "binary_event_bce_no_position_posterior"
                            if _eff_token_loss_mode == "event_deviance_ce"
                            else "equal_sum_of_true_class_ce"
                        )
                    )
                )
                print(
                    "[audit train] "
                    f"online_decision_training={bool(phase_online_decision)} "
                    f"use_stop_head={bool(phase_use_stop_head)} "
                    f"encoding_mode={str(getattr(args, 'encoding_mode', 'onehot'))} "
                    f"sigma_rf_width={float(getattr(args, 'sigma_rf_width', float(getattr(args, 'sigma_rf', 1.0)))):.4f} "
                    f"rf_normalization={str(getattr(args, 'rf_normalization', 'peak'))} "
                    f"sigma_input_noise={float(getattr(args, 'sigma_input_noise', float(getattr(args, 'sigma_rf_noise', 0.0)))):.4f} "
                    f"rf_noise_per_token={bool(getattr(args, 'rf_noise_per_token', True))} "
                    f"noise_mode={str(getattr(args, 'noise_mode', 'per_token'))} "
                    f"noise_rho={float(getattr(args, 'noise_rho', 0.0)):.4f} "
                    f"lambda_end={float(lambda_end_curr):.6f} "
                    f"token_loss_mode={str(args.token_loss_mode)} "
                    f"effective_token_loss_mode={_eff_token_loss_mode} "
                    f"token_window={_token_window_label} "
                    f"token_weighting={_token_weighting_label} "
                    f"lambda_online={float(lambda_online_curr):.6f} "
                    f"response_start={str(getattr(args, 'response_start', 'deviant_offset'))} "
                    f"total_loss_terms=lambda_end*end_loss + lambda_online*online_loss_mean"
                    f"{' + online_ce_weight*online_ce_loss' if bool(phase_online_ce) else ''}"
                    f"{' + online_loss_weight*online_decision_loss' if bool(phase_online_decision) else ''}"
                )
                print(
                    "[rt_diag_def] RT diagnostics use zero-based RT from deviant onset token; "
                    "crossing at deviant onset => 0ms. Legacy batch_meanRT remains a one-indexed diagnostic only."
                )

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

            setattr(model, "_debug_train_rt_diag_window", str(getattr(args, "train_rt_diag_window", "deviant_onset_to_next_tone_onset")))
            setattr(model, "_debug_window_first_n_trials", int(getattr(args, "debug_window_first_n_trials", 6)))

            def _step_behavior_eval_callback(payload: Dict[str, Any]) -> None:
                if behavior_eval_every_steps <= 0:
                    _save_step_checkpoint(payload)
                    return
                global_opt_step = int(payload["global_opt_step"])
                if global_opt_step % behavior_eval_every_steps != 0:
                    _save_step_checkpoint(payload)
                    return

                _save_step_checkpoint(payload)

                human_rt_csv = str(getattr(args, "human_csv_for_rt_corr", ""))
                if not human_rt_csv:
                    return
                human_rt_path = Path(human_rt_csv).expanduser()
                if not human_rt_path.exists():
                    return

                hr_metrics: Dict[str, Any]
                try:
                    hr_metrics = monitor_human_rt_correlation(
                        model=payload["model"],
                        human_csv=human_rt_path,
                        device=device,
                        chunk_len=int(args.chunk_len),
                        token_ms=int(ds.token_ms),
                        tone_ms=int(args.tone_ms),
                        isi_ms=int(isi_ms),
                        f_min_hz=float(args.f_min_hz),
                        f_max_hz=float(args.f_max_hz),
                        n_bins=int(args.n_bins),
                        add_eos=bool(args.add_eos),
                        add_bos=bool(getattr(args, "add_bos", False)),
                        sigma_other_noise=float(args.sigma_other_noise),
                        p_other_noise=float(args.p_other_noise),
                        sigma_silence_noise=float(args.sigma_silence_noise),
                        encoding_mode=str(getattr(args, "encoding_mode", "onehot")),
                        sigma_rf=float(getattr(args, "sigma_rf", 1.0)),
                        rf_normalization=str(getattr(args, "rf_normalization", "peak")),
                        sigma_rf_noise=float(getattr(args, "sigma_rf_noise", 0.0)),
                        rf_noise_per_token=bool(getattr(args, "rf_noise_per_token", True)),
                        seed=int(args.seed),
                        rt_p_thresh=float(args.rt_p_thresh),
                        rt_k_consec=int(args.rt_k_consec),
                        rt_mode=str(getattr(args, "rt_mode", "entropy")),
                        rt_entropy_thresh=float(getattr(args, "rt_entropy_thresh", 0.35)),
                        min_rt_tokens=int(getattr(args, "min_rt_tokens", 0)),
                        readout_mode=str(getattr(args, "human_rt_monitor_readout_mode", "simple_threshold")),
                        readout_start=str(getattr(args, "human_rt_monitor_readout_start", "deviant_onset")),
                        readout_end=str(getattr(args, "human_rt_monitor_readout_end", "trial_end")),
                        rt_reference=str(getattr(args, "human_rt_monitor_rt_reference", "deviant_onset")),
                        advisor_time_cost=float(getattr(args, "human_rt_monitor_advisor_time_cost", 0.0005)),
                        expected_cost_threshold=float(getattr(args, "human_rt_monitor_expected_cost_threshold", 0.5)),
                        advisor_force_deadline=bool(getattr(args, "human_rt_monitor_advisor_force_deadline", False)),
                        decision_not_before=str(getattr(args, "human_rt_monitor_decision_not_before", "window_start")),
                        cost_elapsed_reference=str(getattr(args, "human_rt_monitor_cost_elapsed_reference", "window_start")),
                        decision_min_elapsed_ms=float(getattr(args, "human_rt_monitor_decision_min_elapsed_ms", 0.0)),
                    )
                except Exception as e:
                    print(f"[behavior_step_eval] step={global_opt_step} failed: {e}")
                    return

                metric_value = metric_value_from_sources(selection_metric_name, {}, hr_metrics)
                found_gate = float(getattr(args, "early_stop_min_found_rate", 0.5))
                found_rate = _safe_float(hr_metrics.get("found_rate", float("nan")), float("nan"))
                metric_ready = np.isfinite(found_rate) and found_rate >= found_gate and np.isfinite(metric_value)

                row = {
                    "global_step": global_opt_step,
                    "epoch_global": int(payload["epoch_global"]),
                    "step_in_epoch": int(payload["step_in_epoch"]),
                    "isi_ms": int(isi_ms),
                    "selection_metric": str(selection_metric_name),
                    "selection_metric_value": float(metric_value) if np.isfinite(metric_value) else float("nan"),
                    "found_rate": float(found_rate) if np.isfinite(found_rate) else float("nan"),
                    "metric_ready": bool(metric_ready),
                    "a_r": _safe_float(hr_metrics.get("human_rt_corr_r", float("nan")), float("nan")),
                    "a_r2": _safe_float(hr_metrics.get("human_rt_corr_r2", float("nan")), float("nan")),
                    "c_r": _safe_float(hr_metrics.get("observed_dependency_r", float("nan")), float("nan")),
                    "c_r2": _safe_float(hr_metrics.get("observed_dependency_r2", float("nan")), float("nan")),
                    "delta_r2": _safe_float(hr_metrics.get("delta_r2", float("nan")), float("nan")),
                    "n_trials": int(hr_metrics.get("n_trials", 0) or 0),
                    "n_found": int(hr_metrics.get("n_found", 0) or 0),
                    "readout_mode": str(hr_metrics.get("readout_mode", "")),
                }
                _append_behavior_step_eval_row(row)

                if not metric_ready:
                    print(
                        f"[behavior_step_eval] step={global_opt_step} "
                        f"{selection_metric_name}=nan found_rate={found_rate:.3f} "
                        f"ready=False gate={found_gate:.2f}"
                    )
                    return

                stage_behavior_step_tracker["activated"] = True
                stage_behavior_step_tracker["n_evals"] = int(stage_behavior_step_tracker.get("n_evals", 0)) + 1
                improved = metric_improved(
                    selection_metric_name,
                    float(metric_value),
                    float(stage_behavior_step_tracker.get("best_metric", metric_initial_best(selection_metric_name))),
                    min_delta,
                )
                if improved:
                    stage_behavior_step_tracker["best_metric"] = float(metric_value)
                    stage_behavior_step_tracker["best_step"] = global_opt_step
                    stage_behavior_step_tracker["best_epoch"] = int(payload["epoch_global"])
                    stage_behavior_step_tracker["best_found_rate"] = float(found_rate)
                    stage_behavior_step_tracker["bad_evals"] = 0
                    step_ckpt = _build_step_checkpoint(payload)
                    step_ckpt["behavior_step_eval"] = row
                    torch.save(step_ckpt, stage_behavior_step_best_path)
                else:
                    stage_behavior_step_tracker["bad_evals"] = int(stage_behavior_step_tracker.get("bad_evals", 0)) + 1

                print(
                    f"[behavior_step_eval] step={global_opt_step} "
                    f"{selection_metric_name}={float(metric_value):.4f} "
                    f"found_rate={found_rate:.3f} "
                    f"best={float(stage_behavior_step_tracker['best_metric']):.4f}@{stage_behavior_step_tracker['best_step']} "
                    f"bad={int(stage_behavior_step_tracker['bad_evals'])}/{behavior_patience_evals}"
                )

                if behavior_patience_evals > 0 and int(stage_behavior_step_tracker["bad_evals"]) >= behavior_patience_evals:
                    optimizer_step_state["request_stop"] = True
                    optimizer_step_state["request_stop_reason"] = (
                        f"behavior_step_patience_exhausted:{selection_metric_name}:"
                        f"{int(stage_behavior_step_tracker['bad_evals'])}/{behavior_patience_evals}"
                    )
                    print(
                        f"[behavior_step_early_stop] step={global_opt_step} "
                        f"reason={optimizer_step_state['request_stop_reason']}"
                    )

            tr = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optim,
                device=device,
                chunk_len=int(args.chunk_len),
                lambda_token=float(args.lambda_token),
                end_loss_weight=float(getattr(args, "end_loss_weight", 1.0)),
                lambda_online=float(lambda_online_curr),
                lambda_end=float(lambda_end_curr),
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
                token_supervision_reference=str(getattr(args, "token_supervision_reference", "deviant_offset")),
                include_anchor_token=bool(getattr(args, "include_anchor_token", False)),
                correct_ce_window=str(getattr(args, "correct_ce_window", "deviant_onset_to_next_tone_offset")),
                correct_ce_weighting=str(getattr(args, "correct_ce_weighting", "equal")),
                windowed_correct_ce_average=bool(getattr(args, "windowed_correct_ce_average", False)),
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
                pre_devend_cost_weight=float(getattr(args, "pre_devend_cost_weight", 0.0)),
                pre_devend_cost_mode=str(getattr(args, "pre_devend_cost_mode", "none")),
                pre_devend_cost_margin_ms=float(getattr(args, "pre_devend_cost_margin_ms", 0.0)),
                pre_devend_cost_scale_ms=float(getattr(args, "pre_devend_cost_scale_ms", 50.0)),
                stop_entropy_weight=float(phase_stop_entropy_w),
                stop_prior_weight=float(phase_stop_prior_w),
                stop_prior_target=float(getattr(args, "stop_prior_target", 0.05)),
                pre_p4_uniformity_weight=float(getattr(args, "pre_p4_uniformity_weight", 0.0)),
                pre_evidence_uniform_kl_weight=float(getattr(args, "pre_evidence_uniform_kl_weight", 0.0)),
                pre_evidence_uniform_kl_window=str(getattr(args, "pre_evidence_uniform_kl_window", "trial_start_to_p4_onset")),
                pre_p4_audit_enable=bool(getattr(args, "pre_p4_audit_enable", False)),
                amp_enabled=bool(amp_enabled),
                amp_dtype=str(amp_dtype),
                grad_scaler=grad_scaler,
                debug_loss_check=bool(getattr(args, "debug_loss_check", False)),
                debug_overfit_tiny=bool(getattr(args, "debug_overfit_tiny", False)),
                debug_first_batch_done=bool(first_batch_debug_done),
                optimizer_step_state=optimizer_step_state,
                max_optimizer_steps=(
                    int(getattr(args, "stop_after_optimizer_steps", 0))
                    if int(getattr(args, "stop_after_optimizer_steps", 0)) > 0
                    else (int(getattr(args, "debug_max_steps", 0)) if bool(getattr(args, "debug_overfit_tiny", False)) else None)
                ),
                step_checkpoint_callback=_step_behavior_eval_callback,
                supervision_mode=str(getattr(args, "supervision_mode", "post_deviant")),
                compact_figure_dir=run_dir / "figures",
                compact_figure_max_blocks=3,
                block_context_training=bool(getattr(args, "block_context_training", False)),
                detach_hidden_between_trials=bool(getattr(args, "detach_hidden_between_trials", False)),
                detach_hidden_every_n_trials=int(getattr(args, "detach_hidden_every_n_trials", 1)),
                hidden_carryover_rho=float(getattr(args, "hidden_carryover_rho", 1.0)),
                end_offset_from_trial_end=(1 if bool(getattr(args, "add_bos", False)) else 0),
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
                    end_loss_weight=float(getattr(args, "end_loss_weight", 1.0)),
                    lambda_online=float(lambda_online_curr),
                    lambda_end=float(lambda_end_curr),
                    trial_T_tokens=int(ds.trial_T_tokens),
                    tone_T=int(ds.tone_T),
                    isi_T=int(ds.isi_T),
                    token_loss_mode=str(args.token_loss_mode),
                    token_tau=float(args.token_tau),
                    token_w_min=float(args.token_w_min),
                    token_supervision_reference=str(getattr(args, "token_supervision_reference", "deviant_offset")),
                    include_anchor_token=bool(getattr(args, "include_anchor_token", False)),
                    correct_ce_window=str(getattr(args, "correct_ce_window", "deviant_onset_to_next_tone_offset")),
                    correct_ce_weighting=str(getattr(args, "correct_ce_weighting", "equal")),
                    windowed_correct_ce_average=bool(getattr(args, "windowed_correct_ce_average", False)),
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
                    pre_devend_cost_weight=float(getattr(args, "pre_devend_cost_weight", 0.0)),
                    pre_devend_cost_mode=str(getattr(args, "pre_devend_cost_mode", "none")),
                    pre_devend_cost_margin_ms=float(getattr(args, "pre_devend_cost_margin_ms", 0.0)),
                    pre_devend_cost_scale_ms=float(getattr(args, "pre_devend_cost_scale_ms", 50.0)),
                    stop_entropy_weight=float(phase_stop_entropy_w),
                    stop_prior_weight=float(phase_stop_prior_w),
                    stop_prior_target=float(getattr(args, "stop_prior_target", 0.05)),
                    pre_p4_uniformity_weight=float(getattr(args, "pre_p4_uniformity_weight", 0.0)),
                    pre_evidence_uniform_kl_weight=float(getattr(args, "pre_evidence_uniform_kl_weight", 0.0)),
                    pre_evidence_uniform_kl_window=str(getattr(args, "pre_evidence_uniform_kl_window", "trial_start_to_p4_onset")),
                    supervision_mode=str(getattr(args, "supervision_mode", "post_deviant")),
                    pre_p4_audit_enable=bool(getattr(args, "pre_p4_audit_enable", False)),
                    amp_enabled=bool(amp_enabled),
                    amp_dtype=str(amp_dtype),
                    block_context_training=bool(getattr(args, "block_context_training", False)),
                    detach_hidden_between_trials=bool(getattr(args, "detach_hidden_between_trials", False)),
                    detach_hidden_every_n_trials=int(getattr(args, "detach_hidden_every_n_trials", 1)),
                    hidden_carryover_rho=float(getattr(args, "hidden_carryover_rho", 1.0)),
                    end_offset_from_trial_end=(1 if bool(getattr(args, "add_bos", False)) else 0),
                )
                if bool(getattr(args, "debug", False)) and bool(getattr(args, "pre_p4_audit_enable", False)):
                    _pre_rows_dbg = va.get("pre_p4_probability_audit_rows", []) or []
                    print(
                        f"[pre_p4_prob post_eval] n_rows={len(_pre_rows_dbg)} "
                        f"type={type(_pre_rows_dbg).__name__}"
                    )


            elapsed = time.time() - t_run0
            rt_diag_rows_epoch: List[Dict[str, Any]] = []
            rt_diag_every_n_epochs = int(getattr(args, "train_rt_diag_every_n_epochs", 1))
            if (
                bool(getattr(args, "train_rt_diag_enable", True))
                and bool(getattr(args, "train_rt_diag_every_epoch", True))
                and (not bool(getattr(args, "debug_disable_val", False)))
                and len(val_idx) > 0
                and rt_diag_every_n_epochs > 0
                and (int(epoch_global) % max(1, rt_diag_every_n_epochs) == 0)
            ):
                default_diag_modes = "bayes_cost_argmin" if str(getattr(args, "supervision_mode", "post_deviant")) == "strict_online_p4" else "simple_threshold,bayesian_cost"
                modes_tokens = [m.strip() for m in str(getattr(args, "train_rt_diag_modes", default_diag_modes)).replace(",", " ").split() if m.strip()]
                if "bayes_cost_argmin" in modes_tokens:
                    rt_diag_mode = "bayes_cost_argmin"
                elif ("simple_threshold" in modes_tokens) and ("bayesian_cost" in modes_tokens):
                    rt_diag_mode = "both"
                elif "bayesian_cost" in modes_tokens:
                    rt_diag_mode = "bayesian_cost"
                else:
                    rt_diag_mode = "simple_threshold"
                diag_cost_timeouts_ms = [] if rt_diag_mode == "bayes_cost_argmin" else parse_list_of_floats(str(getattr(args, "train_rt_diag_cost_timeouts_ms", "3000 5000 10000")))
                rt_diag_rows_epoch = run_validation_rt_readout_diagnostics(
                    model=model,
                    loader=val_loader,
                    device=device,
                    chunk_len=int(args.chunk_len),
                    tone_T=int(ds.tone_T),
                    isi_T=int(ds.isi_T),
                    trial_T_tokens=int(ds.trial_T_tokens),
                    token_ms=int(ds.token_ms),
                    readout_window=str(getattr(args, "train_rt_diag_window", "deviant_onset_to_next_tone_onset")),
                    rt_readout_mode=str(rt_diag_mode),
                    p_thresholds=parse_list_of_floats(str(getattr(args, "train_rt_diag_p_thresholds", "0.50 0.60 0.70 0.80 0.90 0.95 0.99"))),
                    cost_weights=parse_list_of_floats(str(getattr(args, "cost_w_list", "0.001"))),
                    cost_timeouts_ms=diag_cost_timeouts_ms,
                    cost_thresholds=parse_list_of_floats(str(getattr(args, "train_rt_diag_cost_thresholds", "0.50 0.60 0.70"))),
                    k_consec_list=parse_list_of_ints(str(getattr(args, "train_rt_diag_k_consec", "1 3"))),
                    max_trials=int(getattr(args, "train_rt_diag_max_trials", 5000)),
                    epoch_global=int(epoch_global),
                    isi_ms=int(isi_ms),
                    checkpoint_label=f"epoch_{int(epoch_global):04d}",
                    supervision_mode=str(getattr(args, "supervision_mode", "post_deviant")),
                )

            behavior_diag = _epoch_behavior_diagnostics(
                val_metrics=va,
                rt_rows=rt_diag_rows_epoch,
                token_ms=int(ds.token_ms),
            )
            if bool(behavior_diag.get("collapse_invalid", False)):
                stage_collapse_epoch_count += 1
                if stage_collapse_epoch_first is None:
                    stage_collapse_epoch_first = int(epoch_global)

            hr_metrics: Dict[str, Any] = {}
            hr_monitor_row: Dict[str, Any] = {}
            human_rt_csv = str(getattr(args, "human_csv_for_rt_corr", ""))
            monitor_every = int(getattr(args, "monitor_human_rt_every", 3))
            if selection_metric_name in {"human_rt_r2", "delta_r2"} or early_stop_metric_name in {"human_rt_r2", "delta_r2"}:
                monitor_every = 1
            if human_rt_csv and monitor_every > 0 and (epoch_global % monitor_every == 0):
                human_rt_path = Path(human_rt_csv).expanduser()
                if human_rt_path.exists():
                    try:
                        hr_metrics = monitor_human_rt_correlation(
                            model=model,
                            human_csv=human_rt_path,
                            device=device,
                            chunk_len=int(args.chunk_len),
                            token_ms=int(ds.token_ms),
                            tone_ms=int(args.tone_ms),
                            isi_ms=int(isi_ms),
                            f_min_hz=float(args.f_min_hz),
                            f_max_hz=float(args.f_max_hz),
                            n_bins=int(args.n_bins),
                            add_eos=bool(args.add_eos),
                            add_bos=bool(getattr(args, "add_bos", False)),
                            sigma_other_noise=float(args.sigma_other_noise),
                            p_other_noise=float(args.p_other_noise),
                            sigma_silence_noise=float(args.sigma_silence_noise),
                            encoding_mode=str(getattr(args, "encoding_mode", "onehot")),
                            sigma_rf=float(getattr(args, "sigma_rf", 1.0)),
                            rf_normalization=str(getattr(args, "rf_normalization", "peak")),
                            sigma_rf_noise=float(getattr(args, "sigma_rf_noise", 0.0)),
                            rf_noise_per_token=bool(getattr(args, "rf_noise_per_token", True)),
                            seed=int(args.seed),
                            rt_p_thresh=float(args.rt_p_thresh),
                            rt_k_consec=int(args.rt_k_consec),
                            rt_mode=str(getattr(args, "rt_mode", "entropy")),
                            rt_entropy_thresh=float(getattr(args, "rt_entropy_thresh", 0.35)),
                            min_rt_tokens=int(getattr(args, "min_rt_tokens", 0)),
                            readout_mode=str(getattr(args, "human_rt_monitor_readout_mode", "simple_threshold")),
                            readout_start=str(getattr(args, "human_rt_monitor_readout_start", "deviant_onset")),
                            readout_end=str(getattr(args, "human_rt_monitor_readout_end", "trial_end")),
                            rt_reference=str(getattr(args, "human_rt_monitor_rt_reference", "deviant_onset")),
                            advisor_time_cost=float(getattr(args, "human_rt_monitor_advisor_time_cost", 0.0005)),
                            expected_cost_threshold=float(getattr(args, "human_rt_monitor_expected_cost_threshold", 0.5)),
                            advisor_force_deadline=bool(getattr(args, "human_rt_monitor_advisor_force_deadline", False)),
                            decision_not_before=str(getattr(args, "human_rt_monitor_decision_not_before", "window_start")),
                            cost_elapsed_reference=str(getattr(args, "human_rt_monitor_cost_elapsed_reference", "window_start")),
                            decision_min_elapsed_ms=float(getattr(args, "human_rt_monitor_decision_min_elapsed_ms", 0.0)),
                        )
                        hr_r = hr_metrics.get("human_rt_corr_r", float("nan"))
                        hr_r2 = float(hr_r * hr_r) if np.isfinite(hr_r) else float("nan")
                        hr_monitor_row = {
                            "human_rt_corr_r": hr_r,
                            "human_rt_corr_r2": hr_r2,
                            "human_rt_corr_p": hr_metrics.get("human_rt_corr_p", float("nan")),
                            "predicted_dependency_r": hr_metrics.get("predicted_dependency_r", float("nan")),
                            "observed_dependency_r": hr_metrics.get("observed_dependency_r", float("nan")),
                            "observed_dependency_r2": hr_metrics.get("observed_dependency_r2", float("nan")),
                            "delta_r2": hr_metrics.get("delta_r2", float("nan")),
                            "participant_found_rate": hr_metrics.get("found_rate", float("nan")),
                        }
                    except Exception as e:
                        print(f"[human_rt_monitor] failed: {e}")
                else:
                    print(f"[human_rt_monitor] human_csv not found: {human_rt_path}")

            selection_metric_value = metric_value_from_sources(selection_metric_name, va, hr_metrics)
            early_stop_metric_value = metric_value_from_sources(early_stop_metric_name, va, hr_metrics)

            behavior_metric_names = {"human_rt_r2", "delta_r2"}
            found_gate = float(getattr(args, "early_stop_min_found_rate", 0.5))
            participant_found_rate = _safe_float(hr_metrics.get("found_rate", float("nan")), float("nan"))
            behavior_metric_ready = (
                np.isfinite(participant_found_rate) and participant_found_rate >= found_gate
            )
            if selection_metric_name in behavior_metric_names and not behavior_metric_ready:
                selection_metric_value = float("nan")
            if early_stop_metric_name in behavior_metric_names and not behavior_metric_ready:
                early_stop_metric_value = float("nan")

            improved_global = metric_improved(selection_metric_name, selection_metric_value, best_val, min_delta)
            if improved_global:
                best_val = float(selection_metric_value)
                best_epoch = int(epoch_global)

            improved_target = False
            if int(isi_ms) == int(best_target_isi):
                improved_target = metric_improved(selection_metric_name, selection_metric_value, best_target_val, min_delta)
                if improved_target:
                    best_target_val = float(selection_metric_value)
                    best_target_epoch = int(epoch_global)

            improved_stage = metric_improved(early_stop_metric_name, early_stop_metric_value, stage_best_val, min_delta)
            if improved_stage:
                stage_best_val = float(early_stop_metric_value)
                stage_bad_count = 0
                stage_best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
                stage_best_optim = copy.deepcopy(optim.state_dict())
                stage_best_epoch_global = int(epoch_global)
            else:
                if (
                    early_stop_metric_name in behavior_metric_names
                    and not behavior_metric_ready
                ):
                    pass
                else:
                    stage_bad_count += 1

            improved_stage_selection = metric_improved(
                selection_metric_name, selection_metric_value, stage_best_metric_value, min_delta
            )
            if improved_stage_selection:
                stage_best_metric_value = float(selection_metric_value)

            improved_stage_loss = (va["total_loss"] < (stage_loss_best_val - min_delta))
            if improved_stage_loss:
                stage_loss_best_val = float(va["total_loss"])
                stage_loss_best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
                stage_loss_best_optim = copy.deepcopy(optim.state_dict())
                stage_loss_best_epoch_global = int(epoch_global)
                stage_loss_best_metrics = dict(va)
                stage_loss_best_behavior = dict(behavior_diag)

            improved_behavior_best = False
            if bool(behavior_diag.get("strong_valid", False)):
                if (not stage_behavior_best_found) or (float(va["total_loss"]) < (stage_behavior_best_val - min_delta)):
                    stage_behavior_best_found = True
                    stage_behavior_best_val = float(va["total_loss"])
                    improved_behavior_best = True
            elif not stage_behavior_best_found:
                if tuple(behavior_diag.get("fallback_score", ())) > tuple(stage_behavior_best_fallback_score):
                    improved_behavior_best = True
                    stage_behavior_best_fallback_score = tuple(behavior_diag.get("fallback_score", ()))

            if improved_behavior_best:
                stage_behavior_best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
                stage_behavior_best_optim = copy.deepcopy(optim.state_dict())
                stage_behavior_best_epoch_global = int(epoch_global)
                stage_behavior_best_metrics = dict(va)
                stage_behavior_best_behavior = dict(behavior_diag)

            target_msg = ""
            if int(isi_ms) == int(best_target_isi):
                target_msg = f" target_best={best_target_val:.4f}@{best_target_epoch}"

            mask_info = ""
            if bool(getattr(args, "use_mask_loss", False)):
                mask_info = " [MASK LOSS ENABLED]"

            if float(getattr(args, "end_loss_weight", 1.0)) == 0.0:
                print(
                    f"[epoch {epoch_global:04d}] (stage={stage} isi={isi_ms} token_ms={ds.token_ms} phase={phase_name} lr={phase_lr:.2e}){mask_info}"
                    f" train: loss={tr['total_loss']:.4f} tok={tr['token_loss']:.4f} "
                    f"win_acc={tr['window_acc']:.4f} | "
                    f"val: loss={va['total_loss']:.4f} tok={va['token_loss']:.4f} "
                    f"win_acc={va['window_acc']:.4f} | "
                    f"best_val={best_val:.4f}@{best_epoch} stage_bad={stage_bad_count}/{patience} "
                    f"loss_best_epoch={stage_loss_best_epoch_global} "
                    f"behavior_valid_best_epoch={stage_behavior_best_epoch_global} "
                    f"elapsed={_fmt_hms(elapsed)}{target_msg}"
                )
            else:
                print(
                    f"[epoch {epoch_global:04d}] (stage={stage} isi={isi_ms} token_ms={ds.token_ms} phase={phase_name} lr={phase_lr:.2e}){mask_info}"
                    f" train: loss={tr['total_loss']:.4f} end={tr['end_loss']:.4f} tok={tr['token_loss']:.4f} "
                    f"win_acc={tr['window_acc']:.4f} | "
                    f"val: loss={va['total_loss']:.4f} end={va['end_loss']:.4f} tok={va['token_loss']:.4f} "
                    f"win_acc={va['window_acc']:.4f} | "
                    f"best_val={best_val:.4f}@{best_epoch} stage_bad={stage_bad_count}/{patience} "
                    f"loss_best_epoch={stage_loss_best_epoch_global} "
                    f"behavior_valid_best_epoch={stage_behavior_best_epoch_global} "
                    f"elapsed={_fmt_hms(elapsed)}{target_msg}"
                )
            print(build_window_by_pos_summary_line("train", tr))
            print(build_window_by_pos_summary_line("val", va))
            gate_p = float(getattr(args, "epoch_save_found_p_threshold", 0.60))
            gate_k = int(getattr(args, "rt_k_consec", 1))
            gate_rt_row = next(
                (
                    r for r in (rt_diag_rows_epoch or [])
                    if str(r.get("readout_mode")) == "simple_threshold"
                    and abs(_safe_float(r.get("p_threshold"), float("nan")) - gate_p) < 1e-8
                    and int(r.get("k_consec", 1)) == gate_k
                ),
                None,
            )
            print(
                f"[gate_acc val] "
                f"acc@{gate_p:.2f}={_safe_float((gate_rt_row or {}).get('acc_at_rt')):.3f} "
                f"P4={_safe_float((gate_rt_row or {}).get('acc_at_rt_P4')):.3f} "
                f"P5={_safe_float((gate_rt_row or {}).get('acc_at_rt_P5')):.3f} "
                f"P6={_safe_float((gate_rt_row or {}).get('acc_at_rt_P6')):.3f}"
            )
            print(
                f"[gate_found val@{gate_p:.2f}] "
                f"P4={_safe_float((gate_rt_row or {}).get('found_rate_P4')):.3f} "
                f"P5={_safe_float((gate_rt_row or {}).get('found_rate_P5')):.3f} "
                f"P6={_safe_float((gate_rt_row or {}).get('found_rate_P6')):.3f}"
            )

            ckpt = {
                "epoch_global": epoch_global,
                "stage": stage,
                "isi_ms": isi_ms,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "optimizer_global_step": int(optimizer_step_state.get("count", 0)),
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
            torch.save(ckpt, run_dir / "latest.pt")
            torch.save(ckpt, run_dir / f"last_isi{int(isi_ms)}.pt")
            torch.save(ckpt, explicit_last_path)
            if improved_global:
                torch.save(ckpt, run_dir / "best.pt")
                torch.save(ckpt, explicit_best_path)
                if selection_metric_name == "val_loss":
                    torch.save(ckpt, run_dir / "best_by_loss.pt")
                else:
                    torch.save(ckpt, run_dir / f"best_by_{selection_metric_name}.pt")
            if improved_stage_selection:
                torch.save(ckpt, stage_best_path)
            if improved_stage_loss:
                torch.save(ckpt, stage_loss_best_path)
            if improved_behavior_best:
                torch.save(ckpt, stage_behavior_best_path)
            if improved_target:
                torch.save(ckpt, best_target_path)
            if bool(getattr(args, "save_epoch_checkpoints", False)):
                save_every = max(1, int(getattr(args, "save_epoch_checkpoints_every", 1)))
                if (int(epoch_global) % save_every) == 0:
                    torch.save(ckpt, checkpoints_dir / f"epoch_{int(epoch_global):04d}.pt")

            row = {
                "epoch_global": epoch_global,
                "stage": stage,
                "isi_ms": isi_ms,
                "token_ms": int(ds.token_ms),

                "train_total_loss": tr["total_loss"],
                "train_end_loss": tr["end_loss"],
                "train_weighted_end_loss": tr.get("weighted_end_loss", float("nan")),
                "train_token_loss": tr["token_loss"],
                "train_online_loss_sum": tr.get("online_loss_sum", float("nan")),
                "train_online_loss_mean": tr.get("online_loss_mean", float("nan")),
                "train_online_valid_n": tr.get("online_valid_n", float("nan")),
                "train_old_token_loss": tr.get("old_token_loss", float("nan")),
                "train_windowed_correct_ce_loss": tr.get("windowed_correct_ce_loss", float("nan")),
                "train_event_deviance_ce_loss": tr.get("event_deviance_ce_loss", float("nan")),
                "train_anti_commit_loss": tr["anti_commit_loss"],
                "train_acc": tr["acc"],
                "train_f1_macro": tr["f1_macro"],
                "train_auc_ovr": tr["auc_ovr"],
                "train_window_acc": tr.get("window_acc", float("nan")),
                "train_window_f1": tr.get("window_f1", float("nan")),
                "train_window_auc": tr.get("window_auc", float("nan")),
                "train_window_acc_first_token": tr.get("window_acc_first_token", float("nan")),
                "train_window_acc_last_token": tr.get("window_acc_last_token", float("nan")),
                "train_window_acc_mean_token": tr.get("window_acc_mean_token", float("nan")),
                "train_window_auc_mean_token": tr.get("window_auc_mean_token", float("nan")),
                "train_window_p_correct_mean": tr.get("window_p_correct_mean", float("nan")),
                "train_window_p_correct_std": tr.get("window_p_correct_std", float("nan")),
                "train_window_p_correct_p10": tr.get("window_p_correct_p10", float("nan")),
                "train_window_p_correct_p50": tr.get("window_p_correct_p50", float("nan")),
                "train_window_p_correct_p90": tr.get("window_p_correct_p90", float("nan")),
                "train_early_p_correct_mean": tr.get("early_p_correct_mean", float("nan")),
                "train_window_p_correct_end": tr.get("window_p_correct_end", float("nan")),
                "train_window_p_correct_max": tr.get("window_p_correct_max", float("nan")),
                "train_window_p_correct_auc": tr.get("window_p_correct_auc", float("nan")),
                "train_window_prediction_mean": tr.get("window_prediction_mean", float("nan")),
                "train_window_prediction_std": tr.get("window_prediction_std", float("nan")),
                "train_window_prediction_min": tr.get("window_prediction_min", float("nan")),
                "train_window_prediction_max": tr.get("window_prediction_max", float("nan")),
                "train_window_y_true_frac_class0": tr.get("window_y_true_frac_class0", float("nan")),
                "train_window_y_true_frac_class1": tr.get("window_y_true_frac_class1", float("nan")),
                "train_window_y_true_frac_class2": tr.get("window_y_true_frac_class2", float("nan")),
                "train_end_acc": tr.get("end_acc", float("nan")),
                "train_end_f1": tr.get("end_f1", float("nan")),
                "train_end_auc": tr.get("end_auc", float("nan")),
                "train_window_acc_P4": tr.get("window_diagnostics", {}).get("window_acc_P4", float("nan")),
                "train_window_acc_P5": tr.get("window_diagnostics", {}).get("window_acc_P5", float("nan")),
                "train_window_acc_P6": tr.get("window_diagnostics", {}).get("window_acc_P6", float("nan")),
                "train_window_f1_P4": tr.get("window_diagnostics", {}).get("window_f1_P4", float("nan")),
                "train_window_f1_P5": tr.get("window_diagnostics", {}).get("window_f1_P5", float("nan")),
                "train_window_f1_P6": tr.get("window_diagnostics", {}).get("window_f1_P6", float("nan")),
                "train_window_auc_P4": tr.get("window_diagnostics", {}).get("window_auc_P4", float("nan")),
                "train_window_auc_P5": tr.get("window_diagnostics", {}).get("window_auc_P5", float("nan")),
                "train_window_auc_P6": tr.get("window_diagnostics", {}).get("window_auc_P6", float("nan")),
                "train_window_p_correct_mean_P4": tr.get("window_diagnostics", {}).get("window_p_correct_mean_P4", float("nan")),
                "train_window_p_correct_mean_P5": tr.get("window_diagnostics", {}).get("window_p_correct_mean_P5", float("nan")),
                "train_window_p_correct_mean_P6": tr.get("window_diagnostics", {}).get("window_p_correct_mean_P6", float("nan")),
                "train_window_p_correct_end_P4": tr.get("window_diagnostics", {}).get("window_p_correct_end_P4", float("nan")),
                "train_window_p_correct_end_P5": tr.get("window_diagnostics", {}).get("window_p_correct_end_P5", float("nan")),
                "train_window_p_correct_end_P6": tr.get("window_diagnostics", {}).get("window_p_correct_end_P6", float("nan")),
                "train_window_token_loss_P4": tr.get("window_diagnostics", {}).get("window_token_loss_P4", float("nan")),
                "train_window_token_loss_P5": tr.get("window_diagnostics", {}).get("window_token_loss_P5", float("nan")),
                "train_window_token_loss_P6": tr.get("window_diagnostics", {}).get("window_token_loss_P6", float("nan")),
                "train_mean_rt_tokens": tr["mean_rt_tokens"],
                "train_mean_rt_ms": tr["mean_rt_ms"],
                "train_rt_found": tr["rt_found"],
                "train_rt_miss": tr["rt_miss"],
                "train_rt_not_first": tr["rt_not_first"],
                "train_rt_negative": tr.get("rt_negative", 0),
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
                "train_lambda_online": tr.get("lambda_online", float("nan")),
                "train_lambda_end_current": tr.get("lambda_end_current", float("nan")),
                "train_weighted_token_loss": tr.get("weighted_token_loss", float("nan")),
                "train_weighted_online_decision_loss": tr.get("weighted_online_decision_loss", float("nan")),
                "train_weighted_online_ce_loss": tr.get("weighted_online_ce_loss", float("nan")),
                "train_aux_token_ce_loss": tr.get("aux_token_ce_loss", float("nan")),
                "train_weighted_aux_token_ce_loss": tr.get("weighted_aux_token_ce_loss", float("nan")),
                "train_pre_p4_uniformity_loss": tr.get("pre_p4_uniformity_loss", float("nan")),
                "train_weighted_pre_p4_uniformity_loss": tr.get("weighted_pre_p4_uniformity_loss", float("nan")),
                "train_pre_evidence_uniform_kl_loss": tr.get("pre_evidence_uniform_kl_loss", float("nan")),
                "train_weighted_pre_evidence_uniform_kl_loss": tr.get("weighted_pre_evidence_uniform_kl_loss", float("nan")),
                "train_preDevCostRaw": tr.get("preDevCostRaw", float("nan")),
                "train_wPreDevCost": tr.get("wPreDevCost", float("nan")),
                "train_preDevCostMode": tr.get("preDevCostMode", ""),
                "train_preDevCostWeight": tr.get("preDevCostWeight", float("nan")),
                "train_preDevCostMargin": tr.get("preDevCostMargin", float("nan")),
                "train_preDevCostScale": tr.get("preDevCostScale", float("nan")),
                "train_anti_immediate_stop_loss": tr.get("anti_immediate_stop_loss", float("nan")),
                "train_stop_entropy_bonus": tr.get("stop_entropy_bonus", float("nan")),
                "train_stop_prior_loss": tr.get("stop_prior_loss", float("nan")),
                "train_sampled_mean_rt_logged_ms": tr.get("sampled_mean_rt_logged_ms", float("nan")),
                "train_last_token_acc": tr.get("last_token_acc", float("nan")),
                "train_deviant_end_acc": tr.get("deviant_end_acc", float("nan")),

                "val_total_loss": va["total_loss"],
                "val_end_loss": va["end_loss"],
                "val_weighted_end_loss": va.get("weighted_end_loss", float("nan")),
                "val_token_loss": va["token_loss"],
                "val_online_loss_sum": va.get("online_loss_sum", float("nan")),
                "val_online_loss_mean": va.get("online_loss_mean", float("nan")),
                "val_online_valid_n": va.get("online_valid_n", float("nan")),
                "val_old_token_loss": va.get("old_token_loss", float("nan")),
                "val_windowed_correct_ce_loss": va.get("windowed_correct_ce_loss", float("nan")),
                "val_event_deviance_ce_loss": va.get("event_deviance_ce_loss", float("nan")),
                "val_anti_commit_loss": va["anti_commit_loss"],
                "val_acc": va["acc"],
                "val_f1_macro": va["f1_macro"],
                "val_auc_ovr": va["auc_ovr"],
                "val_window_acc": va.get("window_acc", float("nan")),
                "val_window_f1": va.get("window_f1", float("nan")),
                "val_window_auc": va.get("window_auc", float("nan")),
                "val_window_acc_first_token": va.get("window_acc_first_token", float("nan")),
                "val_window_acc_last_token": va.get("window_acc_last_token", float("nan")),
                "val_window_acc_mean_token": va.get("window_acc_mean_token", float("nan")),
                "val_window_auc_mean_token": va.get("window_auc_mean_token", float("nan")),
                "val_window_p_correct_mean": va.get("window_p_correct_mean", float("nan")),
                "val_window_p_correct_std": va.get("window_p_correct_std", float("nan")),
                "val_window_p_correct_p10": va.get("window_p_correct_p10", float("nan")),
                "val_window_p_correct_p50": va.get("window_p_correct_p50", float("nan")),
                "val_window_p_correct_p90": va.get("window_p_correct_p90", float("nan")),
                "val_early_p_correct_mean": va.get("early_p_correct_mean", float("nan")),
                "val_window_p_correct_end": va.get("window_p_correct_end", float("nan")),
                "val_window_p_correct_max": va.get("window_p_correct_max", float("nan")),
                "val_window_p_correct_auc": va.get("window_p_correct_auc", float("nan")),
                "val_window_prediction_mean": va.get("window_prediction_mean", float("nan")),
                "val_window_prediction_std": va.get("window_prediction_std", float("nan")),
                "val_window_prediction_min": va.get("window_prediction_min", float("nan")),
                "val_window_prediction_max": va.get("window_prediction_max", float("nan")),
                "val_window_y_true_frac_class0": va.get("window_y_true_frac_class0", float("nan")),
                "val_window_y_true_frac_class1": va.get("window_y_true_frac_class1", float("nan")),
                "val_window_y_true_frac_class2": va.get("window_y_true_frac_class2", float("nan")),
                "val_end_acc": va.get("end_acc", float("nan")),
                "val_end_f1": va.get("end_f1", float("nan")),
                "val_end_auc": va.get("end_auc", float("nan")),
                "val_window_acc_P4": va.get("window_diagnostics", {}).get("window_acc_P4", float("nan")),
                "val_window_acc_P5": va.get("window_diagnostics", {}).get("window_acc_P5", float("nan")),
                "val_window_acc_P6": va.get("window_diagnostics", {}).get("window_acc_P6", float("nan")),
                "val_window_f1_P4": va.get("window_diagnostics", {}).get("window_f1_P4", float("nan")),
                "val_window_f1_P5": va.get("window_diagnostics", {}).get("window_f1_P5", float("nan")),
                "val_window_f1_P6": va.get("window_diagnostics", {}).get("window_f1_P6", float("nan")),
                "val_window_auc_P4": va.get("window_diagnostics", {}).get("window_auc_P4", float("nan")),
                "val_window_auc_P5": va.get("window_diagnostics", {}).get("window_auc_P5", float("nan")),
                "val_window_auc_P6": va.get("window_diagnostics", {}).get("window_auc_P6", float("nan")),
                "val_window_p_correct_mean_P4": va.get("window_diagnostics", {}).get("window_p_correct_mean_P4", float("nan")),
                "val_window_p_correct_mean_P5": va.get("window_diagnostics", {}).get("window_p_correct_mean_P5", float("nan")),
                "val_window_p_correct_mean_P6": va.get("window_diagnostics", {}).get("window_p_correct_mean_P6", float("nan")),
                "val_window_p_correct_end_P4": va.get("window_diagnostics", {}).get("window_p_correct_end_P4", float("nan")),
                "val_window_p_correct_end_P5": va.get("window_diagnostics", {}).get("window_p_correct_end_P5", float("nan")),
                "val_window_p_correct_end_P6": va.get("window_diagnostics", {}).get("window_p_correct_end_P6", float("nan")),
                "val_window_token_loss_P4": va.get("window_diagnostics", {}).get("window_token_loss_P4", float("nan")),
                "val_window_token_loss_P5": va.get("window_diagnostics", {}).get("window_token_loss_P5", float("nan")),
                "val_window_token_loss_P6": va.get("window_diagnostics", {}).get("window_token_loss_P6", float("nan")),
                "val_mean_rt_tokens": va["mean_rt_tokens"],
                "val_mean_rt_ms": va["mean_rt_ms"],
                "val_rt_found": va["rt_found"],
                "val_rt_miss": va["rt_miss"],
                "val_rt_not_first": va["rt_not_first"],
                "val_rt_negative": va.get("rt_negative", 0),
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
                "val_lambda_online": va.get("lambda_online", float("nan")),
                "val_lambda_end_current": va.get("lambda_end_current", float("nan")),
                "val_weighted_token_loss": va.get("weighted_token_loss", float("nan")),
                "val_weighted_online_decision_loss": va.get("weighted_online_decision_loss", float("nan")),
                "val_weighted_online_ce_loss": va.get("weighted_online_ce_loss", float("nan")),
                "val_aux_token_ce_loss": va.get("aux_token_ce_loss", float("nan")),
                "val_weighted_aux_token_ce_loss": va.get("weighted_aux_token_ce_loss", float("nan")),
                "val_pre_p4_uniformity_loss": va.get("pre_p4_uniformity_loss", float("nan")),
                "val_weighted_pre_p4_uniformity_loss": va.get("weighted_pre_p4_uniformity_loss", float("nan")),
                "val_pre_evidence_uniform_kl_loss": va.get("pre_evidence_uniform_kl_loss", float("nan")),
                "val_weighted_pre_evidence_uniform_kl_loss": va.get("weighted_pre_evidence_uniform_kl_loss", float("nan")),
                "val_preDevCostRaw": va.get("preDevCostRaw", float("nan")),
                "val_wPreDevCost": va.get("wPreDevCost", float("nan")),
                "val_preDevCostMode": va.get("preDevCostMode", ""),
                "val_preDevCostWeight": va.get("preDevCostWeight", float("nan")),
                "val_preDevCostMargin": va.get("preDevCostMargin", float("nan")),
                "val_preDevCostScale": va.get("preDevCostScale", float("nan")),
                "val_anti_immediate_stop_loss": va.get("anti_immediate_stop_loss", float("nan")),
                "val_stop_entropy_bonus": va.get("stop_entropy_bonus", float("nan")),
                "val_stop_prior_loss": va.get("stop_prior_loss", float("nan")),
                "val_sampled_mean_rt_logged_ms": va.get("sampled_mean_rt_logged_ms", float("nan")),
                "val_last_token_acc": va.get("last_token_acc", float("nan")),
                "val_deviant_end_acc": va.get("deviant_end_acc", float("nan")),
                "collapse_invalid": bool(behavior_diag.get("collapse_invalid", False)),
                "collapse_reason": str(behavior_diag.get("collapse_reason", "")),
                "behavior_meanRT_ms": behavior_diag.get("meanRT_ms", float("nan")),
                "behavior_batch_meanRT_tokens": behavior_diag.get("batch_meanRT_tokens", float("nan")),
                "behavior_batch_meanRT_ms": behavior_diag.get("batch_meanRT_ms", float("nan")),
                "behavior_floor0": behavior_diag.get("floor0", float("nan")),
                "behavior_floor5": behavior_diag.get("floor5", float("nan")),
                "behavior_P4": behavior_diag.get("P4", float("nan")),
                "behavior_P5": behavior_diag.get("P5", float("nan")),
                "behavior_P6": behavior_diag.get("P6", float("nan")),
                "behavior_ordering": behavior_diag.get("ordering", False),
                "behavior_online_loss_mean": behavior_diag.get("online_loss_mean", float("nan")),
                "behavior_p_correct_p50": behavior_diag.get("p_correct_p50", float("nan")),
                "behavior_p_correct_p90": behavior_diag.get("p_correct_p90", float("nan")),
                "behavior_early_p_correct_mean": behavior_diag.get("early_p_correct_mean", float("nan")),
                "loss_best_epoch": stage_loss_best_epoch_global if stage_loss_best_epoch_global is not None else 0,
                "behavior_valid_best_epoch": stage_behavior_best_epoch_global if stage_behavior_best_epoch_global is not None else 0,

                "best_val": best_val,
                "best_epoch": best_epoch,
                "best_target_isi_val": best_target_val,
                "best_target_isi_epoch": best_target_epoch,
                "best_target_isi": best_target_isi,
                "rt_readout_mode": str(getattr(args, "rt_readout_mode", "both")),
                "cost_readout_window": str(getattr(args, "cost_readout_window", "deviant_onset_to_next_tone_onset")),
                "cost_w_list": str(getattr(args, "cost_w_list", "")),
                "cost_timeout_ms_list": str(getattr(args, "cost_timeout_ms_list", "")),
                "cost_threshold_list": str(getattr(args, "cost_threshold_list", "")),
                "p_correct_threshold_list": str(getattr(args, "p_correct_threshold_list", "")),
                "cost_k_consec_list": str(getattr(args, "cost_k_consec_list", "")),
                "model_selection_metric": selection_metric_name,
                "model_selection_metric_value": selection_metric_value,
                "early_stop_metric": early_stop_metric_name,
                "early_stop_metric_value": early_stop_metric_value,
                "rt_diagnostic_note": "train meanRT uses legacy/simple diagnostic; main RT evaluation should use post-hoc readout sweep",
                "time_elapsed_sec": elapsed,
            }
            row.update(hr_monitor_row)
            rt_candidate_pick = choose_rt_candidate_row(
                rt_rows=rt_diag_rows_epoch,
                val_metrics=va,
                prefer_mode=str(getattr(args, "rt_candidate_prefer_mode", "simple")),
                prefer_threshold=float(getattr(args, "rt_candidate_prefer_threshold", 0.90)),
                min_win_auc=float(getattr(args, "rt_candidate_min_win_auc", 0.60)),
            ) if bool(getattr(args, "select_rt_candidate", True)) else {"selected_row": None, "selected_score": float("-inf"), "selected_meta": None}
            selected_rt_row = rt_candidate_pick.get("selected_row")
            selected_rt_meta = rt_candidate_pick.get("selected_meta") or {}
            if selected_rt_row is not None:
                print(
                    f"[rt_candidate] epoch={int(epoch_global):04d} score={float(selected_rt_meta.get('rt_candidate_score', float('-inf'))):.2f} "
                    f"eligible={bool(selected_rt_meta.get('rt_candidate_eligible', False))} "
                    f"mode={str(selected_rt_row.get('readout_mode'))} "
                    f"p={_safe_float(selected_rt_row.get('p_threshold')):.2f} "
                    f"found={_safe_float(selected_rt_row.get('found_rate')):.2f} "
                    f"floor0={_safe_float(selected_rt_row.get('proportion_rt_floor_0ms')):.2f} "
                    f"floor5={_safe_float(selected_rt_row.get('proportion_rt_floor_5ms')):.2f} "
                    f"meanRT={_safe_float(selected_rt_row.get('mean_rt_ms')):.1f} "
                    f"P4={_safe_float(selected_rt_row.get('mean_rt_P4')):.1f} "
                    f"P5={_safe_float(selected_rt_row.get('mean_rt_P5')):.1f} "
                    f"P6={_safe_float(selected_rt_row.get('mean_rt_P6')):.1f}"
                )
            trajectory_payload = {
                "run_dir": str(run_dir.resolve()),
                "epoch_global": int(epoch_global),
                "stage_idx": int(stage),
                "stage_epoch_1based": int(stage_epoch_1based),
                "isi_ms": int(isi_ms),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "train": _extract_epoch_trajectory_payload(tr),
                "val": _extract_epoch_trajectory_payload(va),
            }
            epoch_traj_path = epoch_traj_dir / f"epoch_{int(epoch_global):04d}_trajectory.json"
            epoch_traj_path.write_text(json.dumps(_serialize_jsonable(trajectory_payload), indent=2), encoding="utf-8")

            valid_gate_mode = str(getattr(args, "epoch_save_start_valid", "fully_valid")).strip().lower()
            save_found_p_threshold = float(getattr(args, "epoch_save_found_p_threshold", 0.60))
            save_found_rate_min = float(getattr(args, "epoch_save_found_rate_min", 0.60))
            save_min_end_acc = float(getattr(args, "epoch_save_min_end_acc", 0.80))
            gate_k = int(getattr(args, "rt_k_consec", 1))
            gate_rt_row = next(
                (
                    r for r in (rt_diag_rows_epoch or [])
                    if str(r.get("readout_mode")) == "simple_threshold"
                    and abs(_safe_float(r.get("p_threshold"), float("nan")) - save_found_p_threshold) < 1e-8
                    and int(r.get("k_consec", 1)) == gate_k
                ),
                None,
            )
            current_val_win_acc = _safe_float((gate_rt_row or {}).get("acc_at_rt", float("nan")))
            current_mean_rt_ms = _safe_float(behavior_diag.get("meanRT_ms"))
            current_collapse_invalid = bool(behavior_diag.get("collapse_invalid", False))
            current_p4_acc = _safe_float((gate_rt_row or {}).get("acc_at_rt_P4", float("nan")))
            current_p5_acc = _safe_float((gate_rt_row or {}).get("acc_at_rt_P5", float("nan")))
            current_p6_acc = _safe_float((gate_rt_row or {}).get("acc_at_rt_P6", float("nan")))
            current_p4_found = _safe_float((gate_rt_row or {}).get("found_rate_P4", float("nan")))
            current_p5_found = _safe_float((gate_rt_row or {}).get("found_rate_P5", float("nan")))
            current_p6_found = _safe_float((gate_rt_row or {}).get("found_rate_P6", float("nan")))
            current_near_valid = (
                (not current_collapse_invalid)
                and np.isfinite(current_mean_rt_ms) and (current_mean_rt_ms >= 100.0)
                and np.isfinite(current_val_win_acc) and (current_val_win_acc >= 0.60)
            )
            current_fully_valid = (
                (not current_collapse_invalid)
                and np.isfinite(current_mean_rt_ms) and (current_mean_rt_ms >= 100.0)
                and np.isfinite(current_val_win_acc) and (current_val_win_acc >= 0.85)
                and np.isfinite(current_p4_acc) and (current_p4_acc >= 0.70)
                and np.isfinite(current_p5_acc) and (current_p5_acc >= 0.70)
                and np.isfinite(current_p6_acc) and (current_p6_acc >= 0.70)
            )
            current_strict_acc_only_valid = (
                np.isfinite(current_val_win_acc) and (current_val_win_acc >= 0.90)
                and np.isfinite(current_p4_acc) and (current_p4_acc >= 0.85)
                and np.isfinite(current_p5_acc) and (current_p5_acc >= 0.85)
                and np.isfinite(current_p6_acc) and (current_p6_acc >= 0.85)
            )
            current_strict_threshold_valid = (
                np.isfinite(current_p4_found) and (current_p4_found >= save_found_rate_min)
                and np.isfinite(current_p5_found) and (current_p5_found >= save_found_rate_min)
                and np.isfinite(current_p6_found) and (current_p6_found >= save_found_rate_min)
            )
            current_end_acc = _safe_float(va.get("end_acc", float("nan")))
            current_end_acc_valid = (
                np.isfinite(current_end_acc) and current_end_acc >= save_min_end_acc
            )
            if valid_gate_mode == "fully_valid":
                current_save_gate_valid = current_fully_valid
            elif valid_gate_mode == "near_valid":
                current_save_gate_valid = current_near_valid
            elif valid_gate_mode == "strict_threshold":
                current_save_gate_valid = current_strict_threshold_valid
            elif valid_gate_mode == "end_acc":
                current_save_gate_valid = current_end_acc_valid
            else:
                current_save_gate_valid = current_strict_acc_only_valid
            if save_epoch_after_valid and (not epoch_save_gate_open) and current_save_gate_valid:
                epoch_save_gate_open = True
                first_epoch_save_gate_epoch = int(epoch_global)
                print(
                    f"[epoch_ckpt_gate] opened at epoch={int(epoch_global)} "
                    f"mode={valid_gate_mode} near_valid={bool(current_near_valid)} "
                    f"fully_valid={bool(current_fully_valid)} strict_acc_only={bool(current_strict_acc_only_valid)} "
                    f"strict_threshold={bool(current_strict_threshold_valid)} "
                    f"end_acc_valid={bool(current_end_acc_valid)} val_end_acc={current_end_acc:.3f} "
                    f"min_end_acc={save_min_end_acc:.3f}"
                )
            if save_steps_after_valid and (not step_save_gate_open) and current_save_gate_valid:
                step_save_gate_open = True
                first_step_save_gate_epoch = int(epoch_global)
                first_step_save_gate_step = int(optimizer_step_state.get("count", 0))
                print(
                    f"[step_ckpt_gate] opened at epoch={int(epoch_global)} "
                    f"global_step={int(first_step_save_gate_step)} mode={valid_gate_mode} "
                    f"near_valid={bool(current_near_valid)} "
                    f"fully_valid={bool(current_fully_valid)} strict_acc_only={bool(current_strict_acc_only_valid)} "
                    f"strict_threshold={bool(current_strict_threshold_valid)} "
                    f"end_acc_valid={bool(current_end_acc_valid)} val_end_acc={current_end_acc:.3f} "
                    f"min_end_acc={save_min_end_acc:.3f}"
                )
            stop_after_valid_n = int(getattr(args, "stop_n_epochs_after_valid", 0))
            if stop_after_valid_n > 0 and stop_after_valid_trigger_epoch is None and current_save_gate_valid:
                stop_after_valid_trigger_epoch = int(epoch_global)
                print(
                    f"[stop_after_valid_gate] triggered at epoch={int(epoch_global)} "
                    f"mode={valid_gate_mode} will_stop_after_epoch={int(epoch_global) + stop_after_valid_n}"
                )

            epoch_ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict() if optim is not None else None,
                "optim_state": optim.state_dict() if optim is not None else None,
                "scheduler_state": None,
                "args": vars(args),
                "args_dict": vars(args),
                "epoch_global": int(epoch_global),
                "optimizer_global_step": int(optimizer_step_state.get("count", 0)),
                "stage_idx": int(stage),
                "stage_epoch_1based": int(stage_epoch_1based),
                "isi_ms": int(isi_ms),
                "metrics": {"train": tr, "val": va},
                "train_metrics": tr,
                "val_metrics": va,
                "rt_diag_rows": rt_diag_rows_epoch,
                "win_by_pos_rows": {
                    "train": tr.get("window_diagnostics_by_position", []),
                    "val": va.get("window_diagnostics_by_position", []),
                },
                "near_valid": bool(current_near_valid),
                "fully_valid": bool(current_fully_valid),
                "strict_acc_only_valid": bool(current_strict_acc_only_valid),
                "end_acc_valid": bool(current_end_acc_valid),
                "epoch_save_min_end_acc": float(save_min_end_acc),
                "epoch_save_gate_open": bool(epoch_save_gate_open),
                "epoch_save_gate_mode": str(valid_gate_mode),
                "first_epoch_save_gate_epoch": (
                    int(first_epoch_save_gate_epoch) if first_epoch_save_gate_epoch is not None else None
                ),
                "step_save_gate_open": bool(step_save_gate_open),
                "first_step_save_gate_epoch": (
                    int(first_step_save_gate_epoch) if first_step_save_gate_epoch is not None else None
                ),
                "first_step_save_gate_step": (
                    int(first_step_save_gate_step) if first_step_save_gate_step is not None else None
                ),
                "stop_after_valid_trigger_epoch": (
                    int(stop_after_valid_trigger_epoch) if stop_after_valid_trigger_epoch is not None else None
                ),
                "trajectory_path": str(epoch_traj_path),
                "sweep": sweep_info,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            if bool(getattr(args, "save_every_epoch", True)) or (save_epoch_after_valid and epoch_save_gate_open):
                epoch_ckpt_path = epoch_ckpt_dir / f"epoch_{int(epoch_global):04d}.pt"
                torch.save(epoch_ckpt, epoch_ckpt_path)
                print(f"[epoch_ckpt] saved {epoch_ckpt_path.relative_to(run_dir)}")

            if selected_rt_row is not None and bool(selected_rt_meta.get("rt_candidate_eligible", False)):
                selected_score = float(selected_rt_meta.get("rt_candidate_score", float("-inf")))
                if selected_score > float(best_rt_candidate_score):
                    best_rt_candidate_score = selected_score
                    torch.save(epoch_ckpt, run_dir / "best_rt_candidate.pt")
                    if int(isi_ms) == int(best_target_isi):
                        torch.save(epoch_ckpt, run_dir / f"best_isi{int(isi_ms)}_rt_candidate.pt")
                    best_rt_candidate_summary = {
                        "selected_epoch": int(epoch_global),
                        "stage_idx": int(stage),
                        "stage_epoch_1based": int(stage_epoch_1based),
                        "isi_ms": int(isi_ms),
                        "rt_candidate_score": selected_score,
                        "eligible": True,
                        "notes": str(selected_rt_meta.get("rt_candidate_notes", "")),
                        "val_window_acc": _safe_float(va.get("window_acc")),
                        "val_window_f1": _safe_float(va.get("window_f1")),
                        "val_window_auc": _safe_float(va.get("window_auc")),
                        "val_token_loss": _safe_float(va.get("token_loss")),
                        "chosen_rt_diag": _serialize_jsonable(selected_rt_row),
                    }
                    print(
                        f"[rt_candidate_best] updated best_rt_candidate.pt epoch={int(epoch_global):04d} "
                        f"score={selected_score:.2f}"
                    )
            if bool(getattr(args, "rt_candidate_save_json", True)):
                (run_dir / "best_rt_candidate_summary.json").write_text(
                    json.dumps(_serialize_jsonable(best_rt_candidate_summary), indent=2),
                    encoding="utf-8",
                )

            run_meta = {
                "run_dir": str(run_dir.resolve()),
                "epoch_global": int(epoch_global),
                "stage_idx": int(stage),
                "isi_ms": int(isi_ms),
                "encoding_mode": str(getattr(args, "encoding_mode", "onehot")),
                "sigma_rf": float(getattr(args, "sigma_rf", 1.0)),
                "sigma_rf_noise": float(getattr(args, "sigma_rf_noise", 0.0)),
                "sigma_other_noise": float(getattr(args, "sigma_other_noise", 0.0)),
                "end_loss_weight": float(getattr(args, "end_loss_weight", 1.0)),
                "lambda_token": float(getattr(args, "lambda_token", 0.0)),
                "token_loss_mode": str(getattr(args, "token_loss_mode", "old")),
                "correct_ce_window": str(getattr(args, "correct_ce_window", "")),
            }
            best_key = (
                str(selected_rt_row.get("readout_mode")) if selected_rt_row is not None else None,
                _safe_float(selected_rt_row.get("w")) if selected_rt_row is not None else float("nan"),
                _safe_float(selected_rt_row.get("timeout_ms")) if selected_rt_row is not None else float("nan"),
                _safe_float(selected_rt_row.get("cost_threshold")) if selected_rt_row is not None else float("nan"),
                _safe_float(selected_rt_row.get("p_threshold")) if selected_rt_row is not None else float("nan"),
                int(selected_rt_row.get("k_consec")) if selected_rt_row is not None else None,
            )
            for rt_row in rt_diag_rows_epoch:
                score_meta = compute_rt_candidate_score_for_row(
                    rt_row=rt_row,
                    val_metrics=va,
                    min_win_auc=float(getattr(args, "rt_candidate_min_win_auc", 0.60)),
                )
                row_key = (
                    str(rt_row.get("readout_mode")),
                    _safe_float(rt_row.get("w")),
                    _safe_float(rt_row.get("timeout_ms")),
                    _safe_float(rt_row.get("cost_threshold")),
                    _safe_float(rt_row.get("p_threshold")),
                    int(rt_row.get("k_consec")),
                )
                epoch_rt_diag_rows_all.append({
                    **run_meta,
                    "mode": str(rt_row.get("readout_mode")),
                    "threshold_p": _safe_float(rt_row.get("p_threshold")),
                    "cost_timeout": _safe_float(rt_row.get("timeout_ms")),
                    "cost_threshold": _safe_float(rt_row.get("cost_threshold")),
                    "k_consec": int(rt_row.get("k_consec")),
                    "found": _safe_float(rt_row.get("found_rate")),
                    "miss": _safe_float(rt_row.get("miss_rate")),
                    "floor0": _safe_float(rt_row.get("proportion_rt_floor_0ms")),
                    "floor5": _safe_float(rt_row.get("proportion_rt_floor_5ms")),
                    "meanRT": _safe_float(rt_row.get("mean_rt_ms")),
                    "medianRT": _safe_float(rt_row.get("median_rt_ms")),
                    "P4": _safe_float(rt_row.get("mean_rt_P4")),
                    "P5": _safe_float(rt_row.get("mean_rt_P5")),
                    "P6": _safe_float(rt_row.get("mean_rt_P6")),
                    "ordering": bool(rt_row.get("rt_ordering_P4_gt_P5_gt_P6", False)),
                    "val_loss": _safe_float(va.get("total_loss")),
                    "val_tok": _safe_float(va.get("token_loss")),
                    "val_win_acc": _safe_float(va.get("window_acc")),
                    "val_win_f1": _safe_float(va.get("window_f1")),
                    "val_win_auc": _safe_float(va.get("window_auc")),
                    "val_end_acc": _safe_float(va.get("end_acc")),
                    "val_end_f1": _safe_float(va.get("end_f1")),
                    "val_end_auc": _safe_float(va.get("end_auc")),
                    "rt_candidate_score": float(score_meta["rt_candidate_score"]),
                    "rt_candidate_eligible": bool(score_meta["rt_candidate_eligible"]),
                    "is_best_rt_candidate": bool(row_key == best_key and bool(score_meta["rt_candidate_eligible"])),
                })
            append_rows_to_csv(epoch_rt_diag_csv, epoch_rt_diag_rows_all)

            for split_name, split_metrics in [("train", tr), ("val", va)]:
                split_loss = _safe_float(split_metrics.get("total_loss"))
                for pos_row in split_metrics.get("window_diagnostics_by_position", []) or []:
                    epoch_window_by_pos_rows_all.append({
                        **run_meta,
                        "split": str(split_name),
                        "position": int(pos_row.get("position")),
                        "win_acc": _safe_float(pos_row.get("window_acc")),
                        "win_auc": _safe_float(pos_row.get("window_auc_class_ovr")),
                        "pEnd": _safe_float(pos_row.get("p_correct_at_window_end")),
                        "split_loss": split_loss,
                    })
            append_rows_to_csv(epoch_window_by_pos_csv, epoch_window_by_pos_rows_all)

            if bool(getattr(args, "pre_p4_audit_enable", False)):
                for audit_row in va.get("pre_p4_probability_audit_rows", []) or []:
                    pre_p4_probability_audit_rows_all.append({
                        **run_meta,
                        "split": "val",
                        "token_ms": int(ds.token_ms),
                        "token_loss_mode": str(args.token_loss_mode),
                        "correct_ce_window": str(getattr(args, "correct_ce_window", "deviant_onset_to_next_tone_offset")),
                        "encoding_mode": str(getattr(args, "encoding_mode", "onehot")),
                        "sigma_rf": float(getattr(args, "sigma_rf", 1.0)),
                        "sigma_rf_noise": float(getattr(args, "sigma_rf_noise", 0.0)),
                        **audit_row,
                    })
                append_rows_to_csv(pre_p4_probability_audit_csv, pre_p4_probability_audit_rows_all)
                if bool(getattr(args, "debug", False)):
                    print(
                        f"[pre_p4_prob write] path={pre_p4_probability_audit_csv} "
                        f"n_rows={len(pre_p4_probability_audit_rows_all)}"
                    )

            for split_name, split_metrics in [("train", tr), ("val", va)]:
                wd = split_metrics.get("window_diagnostics", {}) or {}
                wd_row = {
                    "epoch_global": epoch_global,
                    "stage": stage,
                    "isi_ms": isi_ms,
                    "split": split_name,
                    "token_ms": int(ds.token_ms),
                    "token_loss_mode": str(args.token_loss_mode),
                    "correct_ce_window": str(getattr(args, "correct_ce_window", "deviant_onset_to_next_tone_offset")),
                    "correct_ce_weighting": str(getattr(args, "correct_ce_weighting", "equal")),
                    "encoding_mode": str(getattr(args, "encoding_mode", "onehot")),
                    "sigma_rf": float(getattr(args, "sigma_rf", 1.0)),
                    "sigma_rf_noise": float(getattr(args, "sigma_rf_noise", 0.0)),
                }
                for k, v in wd.items():
                    if k in ("trajectory_rel_ms", "trajectory_by_position", "by_position_rows", "per_trial_position", "per_trial_p_end", "per_trial_mean", "per_trial_auc", "per_trial_crossing_ms"):
                        continue
                    wd_row[k] = v
                wd_row["trajectory_rel_ms"] = json.dumps(np.array(wd.get("trajectory_rel_ms", []), dtype=float).tolist())
                wd_row["trajectory_by_position"] = json.dumps({
                    str(k): np.array(v, dtype=float).tolist() for k, v in (wd.get("trajectory_by_position", {}) or {}).items()
                })
                window_diag_rows.append(wd_row)

                for pos_row in split_metrics.get("window_diagnostics_by_position", []) or []:
                    out_pos = dict(pos_row)
                    out_pos.update({
                        "epoch_global": epoch_global,
                        "stage": stage,
                        "isi_ms": isi_ms,
                        "split": split_name,
                        "token_ms": int(ds.token_ms),
                        "token_loss_mode": str(args.token_loss_mode),
                        "correct_ce_window": str(getattr(args, "correct_ce_window", "deviant_onset_to_next_tone_offset")),
                        "encoding_mode": str(getattr(args, "encoding_mode", "onehot")),
                        "sigma_rf": float(getattr(args, "sigma_rf", 1.0)),
                        "sigma_rf_noise": float(getattr(args, "sigma_rf_noise", 0.0)),
                    })
                    window_diag_by_pos_rows.append(out_pos)

            rf_row = dict(rf_diag_stage)
            rf_row.update({
                "epoch_global": epoch_global,
                "stage": stage,
                "isi_ms": isi_ms,
                "token_ms": int(ds.token_ms),
                "encoding_mode": str(getattr(args, "encoding_mode", "onehot")),
            })
            rf_diag_rows.append(rf_row)

            write_jsonl(jsonl_path, row)
            write_csv_row(csv_path, header, row)
            collapse_diag_rows_all.append({
                "epoch_global": int(epoch_global),
                "stage": int(stage),
                "isi_ms": int(isi_ms),
                "val_loss": _safe_float(va.get("total_loss")),
                "val_win_acc": _safe_float(va.get("window_acc")),
                "val_win_f1": _safe_float(va.get("window_f1")),
                "val_win_auc": _safe_float(va.get("window_auc")),
                "meanRT_ms": _safe_float(behavior_diag.get("meanRT_ms")),
                "batch_meanRT_tokens": _safe_float(behavior_diag.get("batch_meanRT_tokens")),
                "batch_meanRT_ms": _safe_float(behavior_diag.get("batch_meanRT_ms")),
                "floor0": _safe_float(behavior_diag.get("floor0")),
                "floor5": _safe_float(behavior_diag.get("floor5")),
                "P4": _safe_float(behavior_diag.get("P4")),
                "P5": _safe_float(behavior_diag.get("P5")),
                "P6": _safe_float(behavior_diag.get("P6")),
                "ordering": bool(behavior_diag.get("ordering", False)),
                "online_loss_mean": _safe_float(behavior_diag.get("online_loss_mean")),
                "p_correct_p50": _safe_float(behavior_diag.get("p_correct_p50")),
                "p_correct_p90": _safe_float(behavior_diag.get("p_correct_p90")),
                "early_p_correct_mean": _safe_float(behavior_diag.get("early_p_correct_mean")),
                "collapse_invalid": bool(behavior_diag.get("collapse_invalid", False)),
                "collapse_reason": str(behavior_diag.get("collapse_reason", "")),
                "loss_best_epoch": int(stage_loss_best_epoch_global) if stage_loss_best_epoch_global is not None else 0,
                "behavior_valid_best_epoch": int(stage_behavior_best_epoch_global) if stage_behavior_best_epoch_global is not None else 0,
            })
            append_rows_to_csv(collapse_diag_csv, collapse_diag_rows_all)
            for rt_diag_row in rt_diag_rows_epoch:
                rt_diag_row.update({
                    "stage": int(stage),
                    "split": "val",
                    "encoding_mode": str(getattr(args, "encoding_mode", "onehot")),
                    "sigma_rf": float(getattr(args, "sigma_rf", 1.0)),
                    "sigma_rf_noise": float(getattr(args, "sigma_rf_noise", 0.0)),
                    "readout_window": str(getattr(args, "train_rt_diag_window", "deviant_onset_to_next_tone_onset")),
                })
            train_rt_diag_rows.extend(rt_diag_rows_epoch)
            append_rows_to_csv(train_rt_diag_csv, train_rt_diag_rows)
            for rt_diag_row in rt_diag_rows_epoch:
                write_jsonl(train_rt_diag_jsonl, rt_diag_row)
            append_rows_to_csv(window_diag_csv, window_diag_rows)
            append_rows_to_csv(window_diag_by_pos_csv, window_diag_by_pos_rows)
            append_rows_to_csv(rf_diag_csv, rf_diag_rows)
            history.append(row)
            try:
                plot_window_diagnostics(window_diag_rows, window_diag_by_pos_rows, rf_diag_rows, run_dir / "plots")
            except Exception as e:
                print(f"[plot warn] plot_window_diagnostics failed: {e}")

            if int(getattr(args, "plot_every", 0)) > 0 and (epoch_global % int(args.plot_every) == 0):
                try:
                    plot_history(history, run_dir / "plots" / "dummy.png")
                except Exception as e:
                    print(f"[plot warn] plot_history failed: {e}")

            if bool(optimizer_step_state.get("request_stop", False)):
                print(
                    f"[behavior_step_early_stop] stage {stage} (isi={isi_ms}) "
                    f"stop after epoch {epoch_global}: "
                    f"{str(optimizer_step_state.get('request_stop_reason', 'requested'))}"
                )
                break

            # 新增：检查是否达到stage性能阈值（可对最后stage关闭）
            if stage_threshold_active and not stage_reached_threshold:
                if va.get('end_acc', va['acc']) >= stage_perf_threshold:
                    stage_reached_threshold = True
                    current_stage_epochs = epoch_global - stage_start_epoch + 1
                    
                    print(f"\n🎯 Stage {stage} (isi={isi_ms}) reached performance threshold {stage_perf_threshold:.2%} "
                          f"at epoch {epoch_global} with val_end_acc={va.get('end_acc', va['acc']):.2%}")
                    
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
            if int(getattr(args, "stop_after_optimizer_steps", 0)) > 0 and optimizer_step_state["count"] >= int(getattr(args, "stop_after_optimizer_steps", 0)):
                print(f"[stop_after_optimizer_steps] reached optimizer steps: {optimizer_step_state['count']}")
                break
            stop_after_valid_n = int(getattr(args, "stop_n_epochs_after_valid", 0))
            if (
                stop_after_valid_n > 0
                and stop_after_valid_trigger_epoch is not None
                and int(epoch_global) >= int(stop_after_valid_trigger_epoch) + stop_after_valid_n
            ):
                print(
                    f"[stop_after_valid] reached epoch {int(epoch_global)} "
                    f"(trigger={int(stop_after_valid_trigger_epoch)}, extra_epochs={stop_after_valid_n})"
                )
                break

        if stage_loss_best_epoch_global is not None:
            lbm = stage_loss_best_metrics or {}
            lbb = stage_loss_best_behavior or {}
            print(
                f"[stage_best_loss_only] stage={stage} isi={isi_ms} "
                f"epoch={stage_loss_best_epoch_global} val_loss={stage_loss_best_val:.4f} "
                f"win_acc={_safe_float(lbm.get('window_acc')):.4f} "
                f"meanRT={_safe_float(lbb.get('meanRT_ms')):.1f}ms "
                f"floor0={_safe_float(lbb.get('floor0')):.2f} "
                f"collapse_invalid={bool(lbb.get('collapse_invalid', False))}"
            )
        if stage_behavior_best_epoch_global is not None:
            bbm = stage_behavior_best_metrics or {}
            bbb = stage_behavior_best_behavior or {}
            print(
                f"[stage_best_behavior_valid] stage={stage} isi={isi_ms} "
                f"epoch={stage_behavior_best_epoch_global} val_loss={_safe_float(bbm.get('total_loss')):.4f} "
                f"win_acc={_safe_float(bbm.get('window_acc')):.4f} "
                f"meanRT={_safe_float(bbb.get('meanRT_ms')):.1f}ms "
                f"floor0={_safe_float(bbb.get('floor0')):.2f} "
                f"collapse_invalid={bool(bbb.get('collapse_invalid', False))}"
            )
        if (
            stage_restore_mode == "behavior_valid_then_loss"
            and stage_loss_best_epoch_global is not None
            and stage_behavior_best_epoch_global is not None
            and int(stage_loss_best_epoch_global) != int(stage_behavior_best_epoch_global)
        ):
            print("[WARN] loss-best differs from behavior-valid-best; restoring behavior-valid-best.")

        if stage_restore_mode == "selection_metric":
            selected_restore_state = stage_best_state
            selected_restore_optim = stage_best_optim
            selected_restore_epoch = stage_best_epoch_global
            selected_restore_mode_label = f"selection_metric:{selection_metric_name}"
        elif stage_restore_mode == "loss_only":
            selected_restore_state = stage_loss_best_state
            selected_restore_optim = stage_loss_best_optim
            selected_restore_epoch = stage_loss_best_epoch_global
            selected_restore_mode_label = "loss_best"
        else:
            selected_restore_state = stage_behavior_best_state if stage_behavior_best_state is not None else stage_loss_best_state
            selected_restore_optim = stage_behavior_best_optim if stage_behavior_best_state is not None else stage_loss_best_optim
            selected_restore_epoch = stage_behavior_best_epoch_global if stage_behavior_best_state is not None else stage_loss_best_epoch_global
            selected_restore_mode_label = "behavior_valid_best" if stage_behavior_best_state is not None else "loss_best_fallback"

        if selected_restore_state is not None:
            model.load_state_dict(selected_restore_state, strict=True)
            if selected_restore_optim is not None:
                optim.load_state_dict(selected_restore_optim)
            print(
                f"[stage_best_restore] stage={stage} isi={isi_ms} "
                f"mode={selected_restore_mode_label} epoch={selected_restore_epoch}"
            )
        else:
            print(
                f"[stage_best] WARNING: no stage-best snapshot captured for stage={stage} isi={isi_ms}. "
                f"(If you set epochs_per_isi=0, or val crashed, this can happen.)"
            )

        stage_best_summary_rows_all.append({
            "run_dir": str(run_dir.resolve()),
            "stage": int(stage),
            "isi_ms": int(isi_ms),
            "rf": float(getattr(args, "sigma_rf", float("nan"))),
            "sigma_rf_noise": float(getattr(args, "sigma_rf_noise", float("nan"))),
            "noise_rho": float(getattr(args, "noise_rho", float("nan"))),
            "lambda_end": float(getattr(args, "lambda_end", float("nan"))),
            "loss_best_epoch": int(stage_loss_best_epoch_global) if stage_loss_best_epoch_global is not None else 0,
            "loss_best_val_loss": float(stage_loss_best_val) if stage_loss_best_epoch_global is not None else float("nan"),
            "loss_best_win_acc": _safe_float((stage_loss_best_metrics or {}).get("window_acc")),
            "loss_best_meanRT_ms": _safe_float((stage_loss_best_behavior or {}).get("meanRT_ms")),
            "loss_best_floor0": _safe_float((stage_loss_best_behavior or {}).get("floor0")),
            "loss_best_collapse_invalid": bool((stage_loss_best_behavior or {}).get("collapse_invalid", False)),
            "behavior_best_epoch": int(stage_behavior_best_epoch_global or stage_loss_best_epoch_global or 0),
            "behavior_best_val_loss": _safe_float((stage_behavior_best_metrics or stage_loss_best_metrics or {}).get("total_loss")),
            "behavior_best_win_acc": _safe_float((stage_behavior_best_metrics or stage_loss_best_metrics or {}).get("window_acc")),
            "behavior_best_meanRT_ms": _safe_float((stage_behavior_best_behavior or stage_loss_best_behavior or {}).get("meanRT_ms")),
            "behavior_best_floor0": _safe_float((stage_behavior_best_behavior or stage_loss_best_behavior or {}).get("floor0")),
            "behavior_best_collapse_invalid": bool((stage_behavior_best_behavior or stage_loss_best_behavior or {}).get("collapse_invalid", False)),
            "collapse_epoch_first": int(stage_collapse_epoch_first) if stage_collapse_epoch_first is not None else 0,
            "collapse_epoch_count": int(stage_collapse_epoch_count),
            "best_valid_found": bool(stage_behavior_best_found),
            "final_selected_checkpoint": str((stage_behavior_best_path if stage_behavior_best_state is not None else stage_loss_best_path).resolve()) if selected_restore_state is not None else "",
        })
        append_rows_to_csv(stage_best_summary_csv, stage_best_summary_rows_all)

        if bool(getattr(args, "debug_overfit_tiny", False)) and optimizer_step_state["count"] >= int(getattr(args, "debug_max_steps", 500)):
            break
        if int(getattr(args, "stop_after_optimizer_steps", 0)) > 0 and optimizer_step_state["count"] >= int(getattr(args, "stop_after_optimizer_steps", 0)):
            break

    try:
        plot_history(history, run_dir / "plots" / "dummy.png")
    except Exception as e:
        print(f"[plot warn] plot_history_final failed: {e}")
    final_step = int(optimizer_step_state.get("count", 0))
    if final_step > 0:
        _save_step_checkpoint(
            {
                "epoch_global": int(epoch_global),
                "global_opt_step": final_step,
                "step_in_epoch": 0,
                "phase_name": "final",
                "isi_ms": int(isi_ms),
                "end_loss": float("nan"),
                "token_loss": float("nan"),
                "anti_commit_loss": float("nan"),
                "total_loss": float("nan"),
                "end_acc": float("nan"),
                "model": model,
                "optimizer": optim,
            }
        )
    print(f"[done] Saved run to: {run_dir.resolve()}")
    if csv_path.exists():
        shutil.copyfile(csv_path, run_dir / "training_log.csv")
    if train_rt_diag_csv.exists():
        shutil.copyfile(train_rt_diag_csv, run_dir / "rt_summary.csv")
    print("  - best.pt / last.pt / latest.pt")
    if (run_dir / "training_log.csv").exists():
        print("  - training_log.csv")
    if (run_dir / "rt_summary.csv").exists():
        print("  - rt_summary.csv")
    compact_fig = run_dir / "figures" / "compact_trial_timeline.png"
    if compact_fig.exists():
        print(f"  - figure path: {compact_fig}")
    aggregate_fig = run_dir / "figures" / "aggregate_probability_by_position.png"
    if aggregate_fig.exists():
        print(f"  - aggregate figure path: {aggregate_fig}")
    saved_stage_ckpts = []
    for isi_ms in args.isi_schedule:
        p = run_dir / f"best_isi{int(isi_ms)}.pt"
        if p.exists():
            saved_stage_ckpts.append(p.name)
    for name in saved_stage_ckpts:
        print(f"  - {name}")
    print("  - logs/metrics.jsonl + logs/metrics.csv")
    if bool(getattr(args, "pre_p4_audit_enable", False)):
        print("  - pre_p4_probability_audit.csv")
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
    p.add_argument("--data_dir", type=str, default="data_online",
                   help="训练数据目录（会读取 input_blocks.pt/labels_blocks.pt 或 input_tensor.pt/labels_tensor.pt）")
    p.add_argument("--save_dir", type=str, default="runs_strict_online_p4_smoke")
    p.add_argument("--output_dir", type=str, default="",
                   help="Alias of --save_dir for launcher compatibility.")

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
    p.add_argument(
        "--stimuli_exclude_pairs",
        type=str,
        nargs="*",
        default=[],
        help="Directed std,dev pair exclusions, e.g. 1455,1500 1500,1455. Single frequencies are still allowed unless --stimuli_exclude_freqs excludes them.",
    )
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
    p.add_argument("--use_prerendered_tokens", type=str2bool, default=False,
                   help="Read rendered_input_blocks.pt from data_dir instead of rendering token inputs online.")

    # model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_norm", action="store_true")

    # tbptt + losses
    p.add_argument("--chunk_len", type=int, default=512)
    p.add_argument("--lambda_token", type=float, default=0.5)
    p.add_argument("--end_loss_weight", type=float, default=1.0)
    p.add_argument("--lambda_online", type=float, default=1.0,
                   help="Main optimized online/window masked-mean loss weight.")
    p.add_argument("--lambda_end", type=float, default=0.05,
                   help="Auxiliary end-logit scaffold loss weight.")
    p.add_argument("--lambda_end_anneal", action="store_true",
                   help="Linearly anneal lambda_end to zero over the first half of each stage.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--supervision_mode", type=str, default="post_deviant", choices=["post_deviant", "strict_online_p4"])

    p.add_argument("--token_loss_mode", type=str, default="old",
                   choices=["old", "windowed_correct_ce", "strict_p4_causal_ce", "causal_hazard_ce", "event_deviance_ce", "tone_event_ce", "deviance_event_ce", "uniform", "exp"],
                   help="old preserves the current token_loss behavior; event_deviance_ce trains only local standard/deviant tone events without P4/P5/P6 posterior targets.")
    p.add_argument("--token_tau", type=float, default=50.0)
    p.add_argument("--token_w_min", type=float, default=0.05)
    p.add_argument("--correct_ce_window", type=str, default="deviant_onset_to_next_tone_offset",
                   choices=[
                       "deviant_onset_to_deviant_offset",
                       "deviant_onset_to_next_tone_onset",
                       "deviant_onset_to_trial_end",
                       "previous_standard_onset_to_next_tone_onset",
                       "first_possible_deviant_onset_to_next_tone_onset",
                       "first_possible_deviant_onset_to_trial_end",
                       "deviant_onset_to_next_tone_offset",
                       "deviant_onset_to_second_next_tone_onset",
                       "deviant_onset_to_second_next_tone_offset",
                   ])
    p.add_argument("--correct_ce_weighting", type=str, default="equal", choices=["equal"])
    p.add_argument("--disable_aux_online_ce", action="store_true",
                   help="Disable auxiliary online CE in single-phase non-Bayesian training.")

    # timing (ms)
    p.add_argument("--tone_ms", type=int, default=50)
    p.add_argument("--ramp_ms", type=int, default=5)  # kept
    p.add_argument("--token_ms", type=int, default=10)

    # response / readout reference point
    p.add_argument("--response_start", type=str, default="deviant_offset",
                   choices=["deviant_onset", "deviant_offset"],
                   help="When readout scanning starts relative to deviant. "
                        "deviant_onset: RT measured from deviant onset (0ms = onset). "
                        "deviant_offset: RT measured from deviant offset (backward-compatible).")

    # Curriculum
    p.add_argument("--isi_schedule", type=str, default="0,50,300,700",
                   help="ISI 课程学习列表，逗号分隔，例如 '0,50,300,700'")
    p.add_argument("--epochs", type=int, default=0, help="Alias for --epochs_per_isi when >0.")
    p.add_argument("--epochs_per_isi", type=int, default=5,
                   help="每个 ISI 阶段训练多少个 epoch（总 epoch = len(isi_schedule)*epochs_per_isi）")

    # ERB one-hot space
    p.add_argument("--f_min_hz", type=float, default=1300.0)
    p.add_argument("--f_max_hz", type=float, default=1700.0)
    p.add_argument("--n_bins", type=int, default=128)
    p.add_argument("--add_eos", action="store_true")
    p.add_argument("--add_bos", action="store_true", help="Append a dedicated BOS boundary marker after EOS so continuous hidden sees BOS before the next trial.")
    p.add_argument("--eos_mode", type=str, default="separate", choices=["separate", "mixed"],
                   help="separate: append an EOS token; mixed: legacy mode that tags the last real token.")
    p.add_argument("--encoding_mode", type=str, default="onehot", choices=["onehot", "gaussian_rf"])
    p.add_argument("--sigma_rf", type=float, default=1.0,
                   help="[legacy] Gaussian receptive field width in ERB-bin space. Use --sigma_rf_width instead.")
    p.add_argument("--sigma_rf_width", type=float, default=None,
                   help="Gaussian receptive field width in ERB-bin space (canonical name).")
    p.add_argument("--rf_normalization", type=str, default="peak", choices=["peak", "sum", "none"])
    p.add_argument("--sigma_rf_noise", type=float, default=0.0,
                   help="[legacy] Additive Gaussian noise on Gaussian RF vectors before clipping to [0,1]. Use --sigma_input_noise instead.")
    p.add_argument("--sigma_input_noise", type=float, default=None,
                   help="Gaussian noise std added to RF-encoded input representation (canonical name).")
    p.add_argument("--rf_noise_per_token", type=str2bool, default=True,
                   help="If true, resample Gaussian RF noise for each tone token; else once per tone.")
    p.add_argument("--noise_mode", type=str, default="per_token",
                   choices=["per_token", "smoothed", "fixed", "per_tone"],
                   help="Temporal structure for additive sensory noise in ERB bins.")
    p.add_argument("--noise_rho", type=float, default=0.0,
                   help="AR(1) temporal correlation for smoothed sensory noise; main sweep uses 0.95.")

    # noise (single-run values; can be overridden by sweeps)
    p.add_argument("--sigma_other_noise", type=float, default=0.05,
                   help="[legacy] Input noise for onehot encoding. Maps to sigma_input_noise.")
    p.add_argument("--p_other_noise", type=float, default=1.0)
    p.add_argument("--sigma_silence_noise", type=float, default=0.0)
    p.add_argument("--resample_noise_per_epoch", action="store_true",
                   help="训练集在每个 epoch 重采样输入噪声；验证集保持固定噪声。")
    p.add_argument("--hidden_noise_std", type=float, default=0.0,
                   help="训练期加在 layer-norm 后 hidden states 上的高斯噪声标准差。")

    # RT criterion (eval only; can be overridden by sweeps)
    p.add_argument("--rt_p_thresh", type=float, default=0.7)
    p.add_argument("--rt_k_consec", type=int, default=1)
    p.add_argument("--rt_readout_mode", type=str, default="both",
                   choices=["simple_threshold", "simple_threshold_pmax", "masked_logits_threshold", "p_correct_argmax", "p_correct_argmax_correct", "p_correct_accumulator", "baseline_corrected_simple_threshold", "baseline_corrected_dynamic_margin", "baseline_corrected_dynamic_pmax", "baseline_corrected_ec", "bayesian_cost", "bayes_cost_argmin", "both"])
    p.add_argument("--cost_w_list", type=str, default="0.00005 0.0001 0.0002 0.000333 0.0005 0.001")
    p.add_argument("--cost_timeout_ms_list", type=str, default="")
    p.add_argument("--cost_threshold_list", type=str, default="0.30 0.40 0.50 0.60 0.70 0.80")
    p.add_argument("--p_correct_threshold_list", type=str, default="0.50 0.60 0.70 0.80 0.90 0.95")
    p.add_argument("--cost_k_consec_list", type=str, default="1 3 5")
    p.add_argument("--cost_readout_window", type=str, default="deviant_onset_to_next_tone_onset",
                   choices=[
                       "deviant_onset_to_deviant_offset",
                       "deviant_onset_to_next_tone_onset",
                       "previous_standard_offset_to_next_tone_onset",
                       "previous_standard_onset_to_next_tone_onset",
                       "first_possible_deviant_onset_to_next_tone_onset",
                       "first_possible_deviant_onset_to_trial_end",
                       "deviant_onset_to_next_tone_offset",
                       "deviant_onset_to_second_next_tone_onset",
                       "deviant_onset_to_second_next_tone_offset",
                   ])
    p.add_argument("--train_rt_diag_enable", type=str2bool, default=True)
    p.add_argument("--train_rt_diag_every_epoch", type=str2bool, default=True)
    p.add_argument("--train_rt_diag_every_n_epochs", type=int, default=1)
    p.add_argument("--pre_p4_audit_enable", type=str2bool, default=False)
    p.add_argument("--train_rt_diag_window", type=str, default="deviant_onset_to_next_tone_onset",
                   choices=[
                       "deviant_onset_to_deviant_offset",
                       "deviant_onset_to_next_tone_onset",
                       "previous_standard_offset_to_next_tone_onset",
                       "previous_standard_onset_to_next_tone_onset",
                       "first_possible_deviant_onset_to_next_tone_onset",
                       "first_possible_deviant_onset_to_trial_end",
                       "deviant_onset_to_next_tone_offset",
                       "deviant_onset_to_second_next_tone_onset",
                       "deviant_onset_to_second_next_tone_offset",
                   ])
    p.add_argument("--train_rt_diag_modes", type=str, default="simple_threshold,bayesian_cost")
    p.add_argument("--train_rt_diag_p_thresholds", type=str, default="0.50 0.60 0.70 0.80 0.90 0.95 0.99")
    p.add_argument("--train_rt_diag_cost_timeouts_ms", type=str, default="3000 5000 10000")
    p.add_argument("--train_rt_diag_cost_thresholds", type=str, default="0.50 0.60 0.70")
    p.add_argument("--train_rt_diag_k_consec", type=str, default="1 3")
    p.add_argument("--train_rt_diag_max_trials", type=int, default=5000)
    p.add_argument("--save_epoch_checkpoints", type=str2bool, default=False)
    p.add_argument("--save_epoch_checkpoints_every", type=int, default=1)
    p.add_argument("--save_every_epoch", type=str2bool, default=False)
    p.add_argument("--save_epoch_after_valid", type=str2bool, default=False)
    p.add_argument(
        "--epoch_save_start_valid",
        type=str,
        default="fully_valid",
        choices=["near_valid", "fully_valid", "strict_acc_only", "strict_threshold", "end_acc"],
        help="When --save_epoch_after_valid=true, start saving epoch checkpoints once this validity gate is first reached.",
    )
    p.add_argument(
        "--epoch_save_found_p_threshold",
        type=float,
        default=0.60,
        help="Probability threshold used by epoch_save_start_valid=strict_threshold. Supported reliably for thresholds already tracked in window diagnostics, e.g. 0.50/0.55/0.60/0.70/0.90.",
    )
    p.add_argument(
        "--epoch_save_found_rate_min",
        type=float,
        default=0.60,
        help="Minimum per-position found rate required by epoch_save_start_valid=strict_threshold.",
    )
    p.add_argument(
        "--epoch_save_min_end_acc",
        type=float,
        default=0.80,
        help="Minimum validation end_acc required by epoch_save_start_valid=end_acc.",
    )
    p.add_argument(
        "--stop_n_epochs_after_valid",
        type=int,
        default=0,
        help="If >0, once the chosen valid gate is first reached, continue for N more epochs and then stop.",
    )
    p.add_argument("--epoch_ckpt_dir", type=str, default="checkpoints_by_epoch")
    p.add_argument("--save_step_checkpoints", type=str2bool, default=False)
    p.add_argument("--save_step_checkpoints_every", type=int, default=0)
    p.add_argument("--save_steps", type=str, default="",
                   help="Exact optimizer steps to checkpoint, e.g. '0,500,1000,2000'. Saves ckpt_stepXXXXXX.pt.")
    p.add_argument(
        "--save_steps_after_valid",
        type=str2bool,
        default=False,
        help="If true, interpret --save_steps and --save_step_checkpoints_every relative to the optimizer step at which the chosen valid gate is first reached.",
    )
    p.add_argument("--step_ckpt_dir", type=str, default="checkpoints_by_step")
    p.add_argument("--amp", type=str2bool, default=False)
    p.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--persistent_workers", type=str2bool, default=True)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--pin_memory", type=str2bool, default=True)
    p.add_argument("--stop_after_optimizer_steps", type=int, default=0)
    p.add_argument(
        "--behavior_eval_every_steps",
        type=int,
        default=0,
        help="If >0, run participant-level behavioral eval every N optimizer steps.",
    )
    p.add_argument(
        "--behavior_patience_evals",
        type=int,
        default=0,
        help="If >0, stop after this many step-level behavioral evals without improvement.",
    )
    p.add_argument("--select_rt_candidate", type=str2bool, default=True)
    p.add_argument("--rt_candidate_prefer_mode", type=str, default="simple", choices=["simple", "cost", "any"])
    p.add_argument("--rt_candidate_prefer_threshold", type=float, default=0.90)
    p.add_argument("--rt_candidate_min_win_auc", type=float, default=0.60)
    p.add_argument("--rt_candidate_save_json", type=str2bool, default=True)
    p.add_argument("--pre_p4_uniformity_weight", type=float, default=0.0)
    p.add_argument("--pre_evidence_uniform_kl_weight", type=float, default=0.0)
    p.add_argument(
        "--pre_evidence_uniform_kl_window",
        type=str,
        default="trial_start_to_p4_onset",
        choices=["trial_start_to_deviant_onset", "trial_start_to_p4_onset"],
    )

    # resume
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt (best.pt/last.pt)")
    p.add_argument(
        "--resume_override_lr",
        type=str2bool,
        default=False,
        help="When resuming, overwrite restored optimizer lr with --lr after loading optimizer state.",
    )
    p.add_argument(
        "--freeze_variant",
        type=str,
        default="full_finetune",
        choices=["full_finetune", "freeze_recurrent_core", "freeze_output_head"],
        help="Finetune mode for controlled ISI700 experiments. Use with --init_from, not --resume.",
    )

    # debug / sanity
    p.add_argument("--debug", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_n_trials", type=int, default=64)
    p.add_argument("--smoke_epochs", type=int, default=2)
    p.add_argument("--debug_steps", type=int, default=3)
    p.add_argument("--max_blocks", type=int, default=0)
    p.add_argument("--debug_loss_check", action="store_true")
    p.add_argument("--debug_overfit_tiny", action="store_true")
    p.add_argument("--debug_n_blocks", type=int, default=16)
    p.add_argument("--debug_max_steps", type=int, default=500)
    p.add_argument("--debug_disable_val", action="store_true")
    p.add_argument("--debug_no_noise", action="store_true")
    p.add_argument("--debug_window_first_n_trials", type=int, default=6)
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
    p.add_argument("--debug_rt_timing_geometry", action="store_true")
    p.add_argument("--debug_position_aware_min_start", action="store_true")
    p.add_argument(
        "--position_aware_start_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "pos4_onset",
            "actual_deviant_onset",
            "actual_deviant_midpoint",
            "actual_deviant_end",
        ],
    )
    p.add_argument("--tok_window_ms", type=int, default=300)
    p.add_argument("--tok_start_offset_ms", type=int, default=0)
    p.add_argument("--token_supervision_reference", type=str, default="deviant_offset",
                   choices=["deviant_onset", "deviant_offset"],
                   help="Reference point for token CE supervision.")
    p.add_argument("--include_anchor_token", action="store_true",
                   help="When set, token supervision starts at >= reference token instead of > reference token.")
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

    # ---- human RT correlation monitoring during training ----
    p.add_argument("--human_csv_for_rt_corr", type=str, default="",
                   help="human_trial.csv 的路径，用于训练期间每 N 个 epoch 监控 model RT vs human RT 相关性")
    p.add_argument("--monitor_human_rt_every", type=int, default=3,
                   help="每隔多少个 epoch 计算一次 human RT 相关性（0=关闭）")
    p.add_argument("--human_rt_monitor_readout_mode", type=str, default="simple_threshold",
                   choices=["simple_threshold", "advisor_expected_cost_dp", "expected_cost_threshold"],
                   help="训练期间 human RT 监控所用的 RT readout。")
    p.add_argument("--human_rt_monitor_readout_start", type=str, default="deviant_onset",
                   choices=["deviant_onset", "deviant_offset", "previous_standard_onset", "previous_standard_offset", "previous_tone_onset", "previous_tone_offset", "first_possible_deviant_onset", "advisor_position_conditioned_start"])
    p.add_argument("--human_rt_monitor_readout_end", type=str, default="trial_end",
                   choices=["trial_end", "next_tone_onset", "next_tone_offset"])
    p.add_argument("--human_rt_monitor_rt_reference", type=str, default="deviant_onset",
                   choices=["deviant_onset", "deviant_offset"])
    p.add_argument("--human_rt_monitor_advisor_time_cost", type=float, default=0.0005,
                   help="advisor_expected_cost_dp 下的时间代价权重 w，例如 1/2000=0.0005。")
    p.add_argument("--human_rt_monitor_expected_cost_threshold", type=float, default=0.5,
                   help="expected_cost_threshold 下的阈值 theta；当 expected cost <= theta 时作答。")
    p.add_argument("--human_rt_monitor_advisor_force_deadline", type=str2bool, default=False)
    p.add_argument("--human_rt_monitor_decision_not_before", type=str, default="window_start",
                   choices=["window_start", "p4_onset", "deviant_onset"])
    p.add_argument("--human_rt_monitor_cost_elapsed_reference", type=str, default="window_start",
                   choices=["window_start", "p4_onset", "deviant_onset"])
    p.add_argument("--human_rt_monitor_decision_min_elapsed_ms", type=float, default=0.0)
    p.add_argument("--windowed_correct_ce_average", type=str2bool, default=False,
                   help="是否对 windowed_correct_ce 在 window 内按 supervised token 平均。False=raw sum。")
    p.add_argument(
        "--model_selection_metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "human_rt_r2", "delta_r2"],
        help="best.pt / best epoch 的选择指标。delta_r2 = a_R2 - c_R2。",
    )
    p.add_argument(
        "--early_stop_metric",
        type=str,
        default="auto",
        choices=["auto", "val_loss", "human_rt_r2", "delta_r2"],
        help="early stopping 使用的指标。auto 表示跟随 --model_selection_metric。",
    )
    p.add_argument("--early_stop_min_found_rate", type=float, default=0.5,
                   help="行为指标（human_rt_r2 / delta_r2）只有在 found_rate >= 该阈值后才开始 early stopping / best 选择。")
    p.add_argument(
        "--stage_restore_mode",
        type=str,
        default="auto",
        choices=["auto", "selection_metric", "behavior_valid_then_loss", "loss_only"],
        help="每个 curriculum stage 结束后恢复哪一类 best checkpoint。auto: 非 loss 指标时恢复 selection best，否则保持旧行为。",
    )

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
    p.add_argument("--use_event_head", action="store_true")
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
                   choices=["trial_onset", "deviant_start", "deviant_onset", "deviant_end"])
    p.add_argument("--online_supervision_start", type=str, default="deviant_start",
                   choices=["trial_start", "deviant_start", "deviant_onset", "deviant_end"])
    p.add_argument("--online_supervision_end", type=str, default="trial_end",
                   choices=["trial_end", "stimulus_end", "sequence_end", "next_tone_onset", "next_tone_offset", "second_next_tone_onset", "second_next_tone_offset"])
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
                   choices=["trial_end", "stimulus_end", "sequence_end", "next_tone_onset", "next_tone_offset", "second_next_tone_onset", "second_next_tone_offset"])
    p.add_argument("--decision_softmin_tau", type=float, default=0.05)
    p.add_argument("--online_warmup_epochs", type=int, default=2)
    p.add_argument("--policy_gradient_baseline", type=str, default="running_mean",
                   choices=["none", "running_mean", "value_head"])
    p.add_argument("--policy_entropy_weight", type=float, default=0.01)
    p.add_argument("--block_context_training", action="store_true")
    p.add_argument("--detach_hidden_between_trials", action="store_true")
    p.add_argument("--detach_hidden_every_n_trials", type=int, default=1,
                   help="When block_context_training and detach_hidden_between_trials are both enabled, carry gradients across this many consecutive trials before detaching hidden. 1 = detach every trial boundary.")
    p.add_argument("--hidden_carryover_rho", type=float, default=1.0,
                   help="Scale carried hidden state at trial boundaries during block_context_training. 1.0=full carry, 0.0=reset-like carry, values between damp across-trial hidden.")
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
    p.add_argument("--pre_devend_stop_weight", type=float, default=0.0)
    p.add_argument("--pre_devend_cost_weight", type=float, default=0.0)
    p.add_argument("--pre_devend_cost_mode", type=str, default="none",
                   choices=["none", "flat", "linear", "quadratic", "sigmoid_ramp"])
    p.add_argument("--pre_devend_cost_margin_ms", type=float, default=0.0)
    p.add_argument("--pre_devend_cost_scale_ms", type=float, default=50.0)
    p.add_argument("--stop_entropy_weight", type=float, default=0.0)
    p.add_argument("--stop_prior_weight", type=float, default=0.0)
    p.add_argument("--stop_prior_target", type=float, default=0.05)
    p.add_argument("--gap_curriculum", action="store_true")
    p.add_argument("--gap_schedule", type=str, default="25,10,1")
    p.add_argument("--epochs_per_gap", type=int, default=10)
    args = p.parse_args()
    if str(getattr(args, "output_dir", "")).strip():
        args.save_dir = str(getattr(args, "output_dir")).strip()
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
    if int(getattr(args, "epochs", 0)) > 0:
        args.epochs_per_isi = int(args.epochs)

    # ─── name resolution: old CLI args → canonical names ───
    # sigma_rf_width  (canonical) vs sigma_rf  (legacy)
    _provided_sigma_rf_width = getattr(args, "sigma_rf_width", None)
    _provided_sigma_rf = ("--sigma_rf" in __import__("sys").argv)
    if _provided_sigma_rf_width is not None:
        if _provided_sigma_rf and abs(float(args.sigma_rf) - float(args.sigma_rf_width)) > 1e-8:
            raise ValueError(
                f"Conflicting args: --sigma_rf={args.sigma_rf} and "
                f"--sigma_rf_width={args.sigma_rf_width}. "
                "Use only one (--sigma_rf_width is canonical)."
            )
        args.sigma_rf = float(args.sigma_rf_width)
    # always set canonical attribute
    args.sigma_rf_width = float(args.sigma_rf)

    # sigma_input_noise (canonical) vs sigma_rf_noise / sigma_other_noise (legacy)
    _provided_sigma_input_noise = getattr(args, "sigma_input_noise", None)
    _provided_sigma_rf_noise = ("--sigma_rf_noise" in __import__("sys").argv)
    _provided_sigma_other = ("--sigma_other_noise" in __import__("sys").argv)

    if _provided_sigma_input_noise is not None:
        # canonical arg provided — check for conflicts with legacy
        if _provided_sigma_rf_noise and abs(float(args.sigma_rf_noise) - float(args.sigma_input_noise)) > 1e-8:
            raise ValueError(
                f"Conflicting args: --sigma_rf_noise={args.sigma_rf_noise} and "
                f"--sigma_input_noise={args.sigma_input_noise}. "
                "Use only one (--sigma_input_noise is canonical)."
            )
        if _provided_sigma_other and abs(float(args.sigma_other_noise) - float(args.sigma_input_noise)) > 1e-8:
            raise ValueError(
                f"Conflicting args: --sigma_other_noise={args.sigma_other_noise} and "
                f"--sigma_input_noise={args.sigma_input_noise}. "
                "Use only one (--sigma_input_noise is canonical)."
            )
        args.sigma_rf_noise = float(args.sigma_input_noise)
        args.sigma_other_noise = float(args.sigma_input_noise)
    else:
        # no canonical arg — derive from legacy args
        # sigma_rf_noise and sigma_other_noise are already set from CLI or defaults
        if _provided_sigma_rf_noise and _provided_sigma_other:
            if abs(float(args.sigma_rf_noise) - float(args.sigma_other_noise)) > 1e-8:
                print(
                    f"[WARN] --sigma_rf_noise={args.sigma_rf_noise} and "
                    f"--sigma_other_noise={args.sigma_other_noise} differ. "
                    "Using sigma_rf_noise as sigma_input_noise for gaussian_rf mode; "
                    "sigma_other_noise for onehot mode."
                )
                args.sigma_input_noise = float(args.sigma_rf_noise)  # prefer rf_noise for canonical
            else:
                args.sigma_input_noise = float(args.sigma_rf_noise)
        elif _provided_sigma_rf_noise:
            args.sigma_input_noise = float(args.sigma_rf_noise)
        elif _provided_sigma_other:
            args.sigma_input_noise = float(args.sigma_other_noise)
        else:
            # neither legacy arg explicitly provided — use default sigma_rf_noise as canonical
            args.sigma_input_noise = float(args.sigma_rf_noise)

    argv_now = __import__("sys").argv
    args.supervision_mode = str(getattr(args, "supervision_mode", "post_deviant"))
    if bool(getattr(args, "smoke", False)):
        args.max_blocks = max(1, int(np.ceil(float(getattr(args, "smoke_n_trials", 64)) / 10.0)))
        args.epochs_per_isi = int(getattr(args, "smoke_epochs", 2))
        args.early_stop_patience = 0
        args.save_every_epoch = True
        args.save_epoch_checkpoints = True
        print(
            "[smoke] "
            f"supervision_mode={args.supervision_mode} max_blocks={args.max_blocks} "
            f"epochs_per_isi={args.epochs_per_isi}"
        )
    if args.supervision_mode == "strict_online_p4":
        if "--lambda_end" not in argv_now and "--end_loss_weight" not in argv_now:
            args.lambda_end = 0.0
            args.end_loss_weight = 0.0
        if "--response_start" not in argv_now:
            args.response_start = "deviant_onset"
        if "--rt_p_thresh" not in argv_now and "--decision_threshold" in argv_now:
            args.rt_p_thresh = float(args.decision_threshold)
        if "--train_rt_diag_modes" not in argv_now:
            args.train_rt_diag_modes = "bayes_cost_argmin"
        if "--train_rt_diag_window" not in argv_now:
            args.train_rt_diag_window = "first_possible_deviant_onset_to_trial_end"
        if "--cost_readout_window" not in argv_now:
            args.cost_readout_window = "first_possible_deviant_onset_to_trial_end"
        print("[supervision] strict_online_p4: 3-class head, loss mask fixed P4 onset -> trial end")
    if "--lambda_online" not in argv_now:
        args.lambda_online = float(args.lambda_token)
    if "--lambda_end" not in argv_now:
        args.lambda_end = float(args.end_loss_weight)
    if "--noise_mode" not in argv_now:
        args.noise_mode = "per_token"
    if "--noise_rho" not in argv_now:
        args.noise_rho = 0.0 if str(args.noise_mode) == "per_token" else float(args.noise_rho)
    args.lambda_token = float(args.lambda_online)
    args.end_loss_weight = float(args.lambda_end)

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

    if _normalize_token_loss_mode(str(getattr(args, "token_loss_mode", "old"))) == "event_deviance_ce":
        if not bool(getattr(args, "use_event_head", False)):
            print("[event_deviance_ce] enabling --use_event_head automatically")
        args.use_event_head = True

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
        args.token_supervision_reference = "deviant_onset"
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

    tok_anchor_cli_used = "--tok_anchor" in __import__("sys").argv
    token_reference_cli_used = "--token_supervision_reference" in __import__("sys").argv
    if tok_anchor_cli_used and not token_reference_cli_used:
        args.token_supervision_reference = "deviant_onset" if str(args.tok_anchor) == "deviant_onset" else "deviant_offset"

    active_optional_loss = False
    if float(getattr(args, "lambda_anti_commit", 0.0)) != 0.0:
        active_optional_loss = True
    if bool(getattr(args, "online_decision_training", False)) and float(getattr(args, "online_loss_weight", 0.0)) != 0.0:
        active_optional_loss = True
    if bool(getattr(args, "online_decision_training", False)) and (not bool(getattr(args, "disable_aux_online_ce", False))) and float(getattr(args, "online_ce_weight", 0.0)) != 0.0:
        active_optional_loss = True
    if float(getattr(args, "aux_token_ce_weight", 0.0)) != 0.0:
        active_optional_loss = True
    if bool(getattr(args, "anti_immediate_stop", False)) and float(getattr(args, "anti_immediate_stop_weight", 0.0)) != 0.0:
        active_optional_loss = True
    if float(getattr(args, "pre_devend_stop_weight", 0.0)) != 0.0:
        active_optional_loss = True
    if float(getattr(args, "pre_devend_cost_weight", 0.0)) != 0.0:
        active_optional_loss = True
    if float(getattr(args, "stop_entropy_weight", 0.0)) != 0.0:
        active_optional_loss = True
    if float(getattr(args, "stop_prior_weight", 0.0)) != 0.0:
        active_optional_loss = True
    if float(getattr(args, "pre_evidence_uniform_kl_weight", 0.0)) != 0.0:
        active_optional_loss = True
    if (
        float(getattr(args, "end_loss_weight", 1.0)) == 0.0
        and float(getattr(args, "lambda_token", 0.0)) == 0.0
        and not active_optional_loss
    ):
        raise ValueError("No active loss terms: end_loss_weight=0 and lambda_token=0 and no optional losses.")

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
            exclude_pairs=parse_exclude_pairs(args.stimuli_exclude_pairs),
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
            "sigma_input_noise": float(getattr(args, "sigma_input_noise",
                                                float(getattr(args, "sigma_rf_noise", 0.0)))),
            "sigma_rf_width": float(getattr(args, "sigma_rf_width",
                                            float(getattr(args, "sigma_rf", 1.0)))),
            "p_other_noise": float(args.p_other_noise),
            "sigma_silence_noise": float(args.sigma_silence_noise),
            "end_loss_weight": float(getattr(args, "end_loss_weight", 1.0)),
            "lambda_token": float(args.lambda_token),
            "encoding_mode": str(getattr(args, "encoding_mode", "onehot")),
            "sigma_rf": float(getattr(args, "sigma_rf", 1.0)),
            "rf_normalization": str(getattr(args, "rf_normalization", "peak")),
            "sigma_rf_noise": float(getattr(args, "sigma_rf_noise", 0.0)),
            "rf_noise_per_token": bool(getattr(args, "rf_noise_per_token", True)),
            "noise_mode": str(getattr(args, "noise_mode", "per_token")),
            "noise_rho": float(getattr(args, "noise_rho", 0.0)),
            "legacy_sigma_rf": float(getattr(args, "sigma_rf", 1.0)),
            "legacy_sigma_rf_noise": float(getattr(args, "sigma_rf_noise", 0.0)),
            "legacy_sigma_other_noise": float(args.sigma_other_noise),
            "response_start": str(getattr(args, "response_start", "deviant_offset")),
            "end_loss_weight": float(getattr(args, "end_loss_weight", 1.0)),
            "lambda_token": float(getattr(args, "lambda_token", 0.5)),
            "lambda_online": float(getattr(args, "lambda_online", float(getattr(args, "lambda_token", 0.5)))),
            "lambda_end": float(getattr(args, "lambda_end", float(getattr(args, "end_loss_weight", 1.0)))),
            "token_loss_mode": str(getattr(args, "token_loss_mode", "old")),
            "correct_ce_window": str(getattr(args, "correct_ce_window", "deviant_onset_to_next_tone_offset")),
            "correct_ce_weighting": str(getattr(args, "correct_ce_weighting", "equal")),
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
            "endw": float(getattr(run_args, "end_loss_weight", 1.0)),
            "tokw": float(run_args.lambda_token),
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
            "encoding_mode": str(getattr(run_args, "encoding_mode", "onehot")),
            "sigma_rf": float(getattr(run_args, "sigma_rf", 1.0)),
            "sigma_rf_width": float(getattr(run_args, "sigma_rf_width",
                                             float(getattr(run_args, "sigma_rf", 1.0)))),
            "rf_normalization": str(getattr(run_args, "rf_normalization", "peak")),
            "sigma_rf_noise": float(getattr(run_args, "sigma_rf_noise", 0.0)),
            "sigma_input_noise": float(getattr(run_args, "sigma_input_noise",
                                                float(getattr(run_args, "sigma_rf_noise", 0.0)))),
            "rf_noise_per_token": bool(getattr(run_args, "rf_noise_per_token", True)),
            "noise_mode": str(getattr(run_args, "noise_mode", "per_token")),
            "noise_rho": float(getattr(run_args, "noise_rho", 0.0)),
            "legacy_sigma_rf": float(getattr(run_args, "sigma_rf", 1.0)),
            "legacy_sigma_rf_noise": float(getattr(run_args, "sigma_rf_noise", 0.0)),
            "legacy_sigma_other_noise": float(sigma_other),
            "response_start": str(getattr(run_args, "response_start", "deviant_offset")),
            "end_loss_weight": float(getattr(run_args, "end_loss_weight", 1.0)),
            "lambda_token": float(getattr(run_args, "lambda_token", 0.5)),
            "lambda_online": float(getattr(run_args, "lambda_online", float(getattr(run_args, "lambda_token", 0.5)))),
            "lambda_end": float(getattr(run_args, "lambda_end", float(getattr(run_args, "end_loss_weight", 1.0)))),
            "token_loss_mode": str(getattr(run_args, "token_loss_mode", "old")),
            "correct_ce_window": str(getattr(run_args, "correct_ce_window", "deviant_onset_to_next_tone_offset")),
            "correct_ce_weighting": str(getattr(run_args, "correct_ce_weighting", "equal")),
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
