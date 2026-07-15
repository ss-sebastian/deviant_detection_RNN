from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


NOT_FOUND_IMPUTED_RT_MS = 1500.0


def _nanmean_or_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return float("nan")
    return float(np.nanmean(x))


def labels_to_class_index(y_456: torch.Tensor) -> torch.Tensor:
    return (y_456.long() - 4).long()


def deviant_onset_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    dev_idx = (y_pos_456.long() - 1).clamp(min=0)
    step = int(tone_T + isi_T)
    return dev_idx * step


def deviant_offset_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    return deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T) + int(tone_T) - 1


def next_tone_onset_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    return deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T) + int(tone_T + isi_T)


def next_tone_offset_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    return next_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T) + int(tone_T) - 1


def previous_tone_onset_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    return deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T) - int(tone_T + isi_T)


def previous_tone_offset_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    return previous_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T) + int(tone_T) - 1


def first_possible_deviant_onset_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    return torch.zeros_like(y_pos_456.long(), dtype=torch.long) + int(3 * (tone_T + isi_T))


def second_next_tone_onset_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    return deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T) + 2 * int(tone_T + isi_T)


def second_next_tone_offset_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    return second_next_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T) + int(tone_T) - 1


def _event_onset_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
    event_name: str,
) -> torch.Tensor:
    if event_name == "advisor_position_conditioned_start":
        dev_on = deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
        prev_off = previous_tone_offset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
        return torch.where(y_pos_456.long() == 4, dev_on, prev_off)
    if event_name == "deviant_onset":
        return deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
    if event_name == "previous_standard_onset" or event_name == "previous_tone_onset":
        return previous_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
    if event_name == "first_possible_deviant_onset":
        return first_possible_deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
    if event_name == "next_tone_onset":
        return next_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
    if event_name == "deviant_offset":
        return deviant_offset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
    if event_name == "previous_standard_offset" or event_name == "previous_tone_offset":
        return previous_tone_offset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
    if event_name == "next_tone_offset":
        return next_tone_offset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
    raise ValueError(f"Unknown event_name: {event_name}")


def _event_time_ms_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
    token_ms: int,
    event_name: str,
) -> torch.Tensor:
    if event_name == "advisor_position_conditioned_start":
        dev_on = _event_time_ms_in_trial(
            y_pos_456=y_pos_456,
            tone_T=tone_T,
            isi_T=isi_T,
            token_ms=token_ms,
            event_name="deviant_onset",
        )
        prev_off = _event_time_ms_in_trial(
            y_pos_456=y_pos_456,
            tone_T=tone_T,
            isi_T=isi_T,
            token_ms=token_ms,
            event_name="previous_tone_offset",
        )
        return torch.where(y_pos_456.long() == 4, dev_on, prev_off)
    if event_name == "deviant_onset":
        return deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T).float() * float(token_ms)
    if event_name == "deviant_offset":
        return (
            deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T).float() + float(tone_T)
        ) * float(token_ms)
    if event_name == "previous_standard_onset" or event_name == "previous_tone_onset":
        return previous_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T).float() * float(token_ms)
    if event_name == "first_possible_deviant_onset":
        return first_possible_deviant_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T).float() * float(token_ms)
    if event_name == "previous_standard_offset" or event_name == "previous_tone_offset":
        return (
            previous_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T).float() + float(tone_T)
        ) * float(token_ms)
    if event_name == "next_tone_onset":
        return next_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T).float() * float(token_ms)
    if event_name == "next_tone_offset":
        return (
            next_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T).float() + float(tone_T)
        ) * float(token_ms)
    raise ValueError(f"Unknown event_name: {event_name}")


def compute_readout_window_bounds_in_trial(
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    readout_start: str,
    readout_end: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    start = _event_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T, event_name=readout_start)
    if readout_end == "trial_end":
        end_excl = torch.full_like(start, int(trial_T_tokens))
    elif readout_end == "next_tone_onset":
        end_excl = next_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T)
    elif readout_end == "next_tone_offset":
        end_excl = next_tone_onset_token_in_trial(y_pos_456=y_pos_456, tone_T=tone_T, isi_T=isi_T) + int(tone_T)
    else:
        raise ValueError(f"Unknown readout_end: {readout_end}")
    end_excl = end_excl.clamp(min=0, max=int(trial_T_tokens))
    start = start.clamp(min=0, max=int(trial_T_tokens))
    end_excl = torch.maximum(end_excl, start + 1)
    return start.long(), end_excl.long()


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    xv = x[m]
    yv = y[m]
    if np.nanstd(xv) < 1e-12 or np.nanstd(yv) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def prior_prob_from_position(position_456: np.ndarray) -> np.ndarray:
    pos = np.asarray(position_456, dtype=int)
    out = np.full(pos.shape, np.nan, dtype=float)
    out[pos == 4] = 1.0 / 3.0
    out[pos == 5] = 1.0 / 2.0
    out[pos == 6] = 1.0
    return out


def prior_surprisal_from_position(position_456: np.ndarray) -> np.ndarray:
    p = prior_prob_from_position(position_456)
    return -np.log(p)


@dataclass
class PreparedRTReadout:
    logits_deviant: np.ndarray  # (N,T,3), raw logits for P4/P5/P6
    probs_correct: np.ndarray  # (N,T)
    probs_max: np.ndarray      # (N,T)
    entropy: np.ndarray        # (N,T), raw 3-class entropy in nats
    logits_margin: np.ndarray  # (N,T), top1 - top2 logits
    pred_class: np.ndarray     # (N,T)
    y_cls: np.ndarray          # (N,)
    y_pos_456: np.ndarray      # (N,)
    window_start: np.ndarray   # (N,)
    window_end_exclusive: np.ndarray  # (N,)
    window_start_time_ms: np.ndarray
    window_end_time_ms: np.ndarray
    rt_reference_time_ms: np.ndarray
    deviant_onset_ms: np.ndarray
    deviant_offset_ms: np.ndarray
    previous_tone_onset_ms: np.ndarray
    previous_tone_offset_ms: np.ndarray
    next_tone_onset_ms: np.ndarray
    next_tone_offset_ms: np.ndarray
    p4_onset_ms: np.ndarray
    token_ms: float
    p_correct_at_dev_onset: np.ndarray
    p_correct_at_window_end: np.ndarray
    mean_p_correct_in_window: np.ndarray
    max_p_correct_in_window: np.ndarray
    auc_p_correct_in_window: np.ndarray
    p_correct_at_window_25pct: np.ndarray
    p_correct_at_window_50pct: np.ndarray
    p_correct_at_window_75pct: np.ndarray
    p_max_at_dev_onset: np.ndarray
    p_max_at_window_end: np.ndarray
    mean_p_max_in_window: np.ndarray
    max_p_max_in_window: np.ndarray
    auc_p_max_in_window: np.ndarray
    trajectory_rel_ms: List[float]
    trajectory_by_position: Dict[int, List[float]]
    overall_metrics: Dict[str, Any]
    by_position_metrics: Dict[int, Dict[str, Any]]
    supervision_mode: str = "post_deviant"


def prepare_rt_readout(
    logits_trial: torch.Tensor | np.ndarray,  # (N,Tt,C)
    y_cls: torch.Tensor | np.ndarray,         # (N,)
    y_pos_456: torch.Tensor | np.ndarray,     # (N,)
    tone_T: int,
    isi_T: int,
    token_ms: int,
    readout_start: str = "deviant_onset",
    readout_end: str = "trial_end",
    rt_reference: str = "deviant_onset",
    supervision_mode: str = "post_deviant",
) -> PreparedRTReadout:
    if isinstance(logits_trial, np.ndarray):
        logits_t = torch.from_numpy(logits_trial).float()
    else:
        logits_t = logits_trial.detach().float().cpu()
    if isinstance(y_cls, np.ndarray):
        y_cls_t = torch.from_numpy(y_cls).long()
    else:
        y_cls_t = y_cls.detach().long().cpu()
    if isinstance(y_pos_456, np.ndarray):
        y_pos_t = torch.from_numpy(y_pos_456).long()
    else:
        y_pos_t = y_pos_456.detach().long().cpu()

    if logits_t.ndim != 3 or logits_t.shape[-1] < 3:
        raise ValueError(f"logits_trial must be (N,T,C>=3), got {tuple(logits_t.shape)}")

    N, Tt, C = logits_t.shape
    probs = torch.softmax(logits_t, dim=-1)
    deviant_probs = probs[..., :3]
    deviant_logits = logits_t[..., :3]
    p_max_all = deviant_probs.max(dim=-1).values.numpy()
    entropy_all = (-(deviant_probs * torch.log(deviant_probs.clamp_min(1e-12))).sum(dim=-1)).numpy()
    pred_class = deviant_probs.argmax(dim=-1).numpy().astype(np.int64)
    top2_logits = torch.topk(deviant_logits, k=2, dim=-1).values
    logits_margin = (top2_logits[..., 0] - top2_logits[..., 1]).numpy()
    gather_idx = y_cls_t.view(-1, 1, 1).expand(-1, Tt, 1)
    p_corr = deviant_probs.gather(dim=-1, index=gather_idx).squeeze(-1).numpy()

    start_t, end_t = compute_readout_window_bounds_in_trial(
        y_pos_456=y_pos_t,
        trial_T_tokens=int(Tt),
        tone_T=int(tone_T),
        isi_T=int(isi_T),
        readout_start=str(readout_start),
        readout_end=str(readout_end),
    )
    start = start_t.numpy().astype(np.int64)
    end_excl = end_t.numpy().astype(np.int64)
    y_pos_np = y_pos_t.numpy().astype(np.int64)
    y_cls_np = y_cls_t.numpy().astype(np.int64)
    dev_on_ms = _event_time_ms_in_trial(y_pos_t, tone_T=tone_T, isi_T=isi_T, token_ms=token_ms, event_name="deviant_onset").numpy()
    dev_off_ms = _event_time_ms_in_trial(y_pos_t, tone_T=tone_T, isi_T=isi_T, token_ms=token_ms, event_name="deviant_offset").numpy()
    prev_on_ms = _event_time_ms_in_trial(y_pos_t, tone_T=tone_T, isi_T=isi_T, token_ms=token_ms, event_name="previous_tone_onset").numpy()
    prev_off_ms = _event_time_ms_in_trial(y_pos_t, tone_T=tone_T, isi_T=isi_T, token_ms=token_ms, event_name="previous_tone_offset").numpy()
    next_on_ms = _event_time_ms_in_trial(y_pos_t, tone_T=tone_T, isi_T=isi_T, token_ms=token_ms, event_name="next_tone_onset").numpy()
    next_off_ms = _event_time_ms_in_trial(y_pos_t, tone_T=tone_T, isi_T=isi_T, token_ms=token_ms, event_name="next_tone_offset").numpy()
    p4_on_ms = _event_time_ms_in_trial(y_pos_t, tone_T=tone_T, isi_T=isi_T, token_ms=token_ms, event_name="first_possible_deviant_onset").numpy()
    start_ms = start.astype(float) * float(token_ms)
    end_ms = end_excl.astype(float) * float(token_ms)
    ref_ms = _event_time_ms_in_trial(y_pos_t, tone_T=tone_T, isi_T=isi_T, token_ms=token_ms, event_name=str(rt_reference)).numpy()

    p_dev_on = np.full((N,), np.nan, dtype=float)
    p_end = np.full((N,), np.nan, dtype=float)
    p25 = np.full((N,), np.nan, dtype=float)
    p50 = np.full((N,), np.nan, dtype=float)
    p75 = np.full((N,), np.nan, dtype=float)
    mean_p = np.full((N,), np.nan, dtype=float)
    max_p = np.full((N,), np.nan, dtype=float)
    auc_p = np.full((N,), np.nan, dtype=float)
    pmax_dev_on = np.full((N,), np.nan, dtype=float)
    pmax_end = np.full((N,), np.nan, dtype=float)
    mean_pmax = np.full((N,), np.nan, dtype=float)
    max_pmax = np.full((N,), np.nan, dtype=float)
    auc_pmax = np.full((N,), np.nan, dtype=float)

    max_len = int(np.max(np.maximum(1, end_excl - start)))
    traj_sum: Dict[int, np.ndarray] = {4: np.zeros((max_len,), dtype=float), 5: np.zeros((max_len,), dtype=float), 6: np.zeros((max_len,), dtype=float)}
    traj_cnt: Dict[int, np.ndarray] = {4: np.zeros((max_len,), dtype=float), 5: np.zeros((max_len,), dtype=float), 6: np.zeros((max_len,), dtype=float)}

    for i in range(N):
        s = int(start[i])
        e = int(end_excl[i])
        w = p_corr[i, s:e]
        w_max = p_max_all[i, s:e]
        if w.size == 0:
            continue
        p_dev_on[i] = float(w[0])
        p_end[i] = float(w[-1])
        idx25 = int(round((w.size - 1) * 0.25))
        idx50 = int(round((w.size - 1) * 0.50))
        idx75 = int(round((w.size - 1) * 0.75))
        p25[i] = float(w[idx25])
        p50[i] = float(w[idx50])
        p75[i] = float(w[idx75])
        mean_p[i] = float(np.mean(w))
        max_p[i] = float(np.max(w))
        pmax_dev_on[i] = float(w_max[0])
        pmax_end[i] = float(w_max[-1])
        mean_pmax[i] = float(np.mean(w_max))
        max_pmax[i] = float(np.max(w_max))
        if hasattr(np, "trapezoid"):
            auc_p[i] = float(np.trapezoid(w, dx=float(token_ms)))
            auc_pmax[i] = float(np.trapezoid(w_max, dx=float(token_ms)))
        else:
            auc_p[i] = float(np.trapz(w, dx=float(token_ms)))
            auc_pmax[i] = float(np.trapz(w_max, dx=float(token_ms)))
        pos = int(y_pos_np[i])
        if pos in traj_sum:
            traj_sum[pos][: w.size] += w
            traj_cnt[pos][: w.size] += 1.0

    trajectory_rel_ms = [float(i * token_ms) for i in range(max_len)]
    traj_by_pos: Dict[int, List[float]] = {}
    by_pos: Dict[int, Dict[str, Any]] = {}
    for pos in [4, 5, 6]:
        traj = np.divide(traj_sum[pos], np.maximum(traj_cnt[pos], 1.0))
        traj[traj_cnt[pos] == 0] = np.nan
        traj_by_pos[pos] = traj.tolist()
        m = y_pos_np == pos
        by_pos[pos] = {
            "n_trials": int(np.sum(m)),
            "p_correct_at_dev_onset": _nanmean_or_nan(p_dev_on[m]) if np.any(m) else float("nan"),
            "p_correct_at_window_end": _nanmean_or_nan(p_end[m]) if np.any(m) else float("nan"),
            "mean_p_correct_in_window": _nanmean_or_nan(mean_p[m]) if np.any(m) else float("nan"),
            "auc_p_correct_in_window": _nanmean_or_nan(auc_p[m]) if np.any(m) else float("nan"),
            "p_max_at_dev_onset": _nanmean_or_nan(pmax_dev_on[m]) if np.any(m) else float("nan"),
            "p_max_at_window_end": _nanmean_or_nan(pmax_end[m]) if np.any(m) else float("nan"),
            "mean_p_max_in_window": _nanmean_or_nan(mean_pmax[m]) if np.any(m) else float("nan"),
            "auc_p_max_in_window": _nanmean_or_nan(auc_pmax[m]) if np.any(m) else float("nan"),
        }

    overall = {
        "p_correct_at_dev_onset": _nanmean_or_nan(p_dev_on),
        "p_correct_at_window_25pct": _nanmean_or_nan(p25),
        "p_correct_at_window_50pct": _nanmean_or_nan(p50),
        "p_correct_at_window_75pct": _nanmean_or_nan(p75),
        "p_correct_at_window_end": _nanmean_or_nan(p_end),
        "mean_p_correct_in_window": _nanmean_or_nan(mean_p),
        "max_p_correct_in_window": _nanmean_or_nan(max_p),
        "auc_p_correct_in_window": _nanmean_or_nan(auc_p),
        "p_max_at_dev_onset": _nanmean_or_nan(pmax_dev_on),
        "p_max_at_window_end": _nanmean_or_nan(pmax_end),
        "mean_p_max_in_window": _nanmean_or_nan(mean_pmax),
        "max_p_max_in_window": _nanmean_or_nan(max_pmax),
        "auc_p_max_in_window": _nanmean_or_nan(auc_pmax),
        "p_correct_ordering_ok": bool(
            np.isfinite(by_pos[4]["p_correct_at_window_end"])
            and np.isfinite(by_pos[5]["p_correct_at_window_end"])
            and np.isfinite(by_pos[6]["p_correct_at_window_end"])
            and (by_pos[6]["p_correct_at_window_end"] > by_pos[5]["p_correct_at_window_end"] > by_pos[4]["p_correct_at_window_end"])
        ),
    }

    return PreparedRTReadout(
        logits_deviant=deviant_logits.numpy(),
        probs_correct=p_corr,
        probs_max=p_max_all,
        entropy=entropy_all,
        logits_margin=logits_margin,
        pred_class=pred_class,
        y_cls=y_cls_np,
        y_pos_456=y_pos_np,
        window_start=start,
        window_end_exclusive=end_excl,
        window_start_time_ms=start_ms,
        window_end_time_ms=end_ms,
        rt_reference_time_ms=ref_ms,
        deviant_onset_ms=dev_on_ms,
        deviant_offset_ms=dev_off_ms,
        previous_tone_onset_ms=prev_on_ms,
        previous_tone_offset_ms=prev_off_ms,
        next_tone_onset_ms=next_on_ms,
        next_tone_offset_ms=next_off_ms,
        p4_onset_ms=p4_on_ms,
        token_ms=float(token_ms),
        p_correct_at_dev_onset=p_dev_on,
        p_correct_at_window_end=p_end,
        mean_p_correct_in_window=mean_p,
        max_p_correct_in_window=max_p,
        auc_p_correct_in_window=auc_p,
        p_correct_at_window_25pct=p25,
        p_correct_at_window_50pct=p50,
        p_correct_at_window_75pct=p75,
        p_max_at_dev_onset=pmax_dev_on,
        p_max_at_window_end=pmax_end,
        mean_p_max_in_window=mean_pmax,
        max_p_max_in_window=max_pmax,
        auc_p_max_in_window=auc_pmax,
        trajectory_rel_ms=trajectory_rel_ms,
        trajectory_by_position=traj_by_pos,
        overall_metrics=overall,
        by_position_metrics=by_pos,
        supervision_mode=str(supervision_mode),
    )


def _first_k_consecutive(mask: np.ndarray, k: int) -> int:
    run = 0
    for i, ok in enumerate(mask.tolist()):
        if bool(ok):
            run += 1
            if run >= k:
                return i - k + 1
        else:
            run = 0
    return -1


def _condition_rt_stats(rt_ms: np.ndarray, found: np.ndarray, pos: np.ndarray) -> Dict[str, Any]:
    found = found.astype(bool)
    valid = found & np.isfinite(rt_ms)
    vals = rt_ms[valid]
    out: Dict[str, Any] = {
        "n_trials": int(rt_ms.size),
        "found_rate": float(np.mean(found.astype(float))) if rt_ms.size > 0 else float("nan"),
        "miss_rate": float(np.mean((~found).astype(float))) if rt_ms.size > 0 else float("nan"),
        "mean_rt_ms": float(np.mean(vals)) if vals.size > 0 else float("nan"),
        "median_rt_ms": float(np.median(vals)) if vals.size > 0 else float("nan"),
        "std_rt_ms": float(np.std(vals)) if vals.size > 0 else float("nan"),
        "iqr_rt_ms": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if vals.size > 0 else float("nan"),
        "proportion_rt_floor_0ms": float(np.mean(vals <= 0.0)) if vals.size > 0 else float("nan"),
        "proportion_rt_floor_5ms": float(np.mean(vals <= 5.0)) if vals.size > 0 else float("nan"),
        "proportion_rt_floor_10ms": float(np.mean(vals <= 10.0)) if vals.size > 0 else float("nan"),
        "proportion_rt_floor_20ms": float(np.mean(vals <= 20.0)) if vals.size > 0 else float("nan"),
    }
    means = {}
    medians = {}
    found_rates = {}
    for p in [4, 5, 6]:
        mp = pos == p
        vals_p = rt_ms[valid & mp]
        out[f"n_P{p}"] = int(np.sum(mp))
        means[p] = float(np.mean(vals_p)) if vals_p.size > 0 else float("nan")
        medians[p] = float(np.median(vals_p)) if vals_p.size > 0 else float("nan")
        found_rates[p] = float(np.mean(found[mp].astype(float))) if np.any(mp) else float("nan")
        out[f"mean_rt_P{p}"] = means[p]
        out[f"median_rt_P{p}"] = medians[p]
        out[f"std_rt_P{p}"] = float(np.std(vals_p)) if vals_p.size > 0 else float("nan")
        out[f"found_rate_P{p}"] = found_rates[p]
        out[f"floor_rate_5ms_P{p}"] = float(np.mean(vals_p <= 5.0)) if vals_p.size > 0 else float("nan")
    out["rt_ordering_P4_gt_P5_gt_P6"] = bool(
        np.isfinite(means[4]) and np.isfinite(means[5]) and np.isfinite(means[6]) and (means[4] > means[5] > means[6])
    )
    return out


def _decision_validity_stats(
    *,
    decision_time_ms: np.ndarray,
    p4_onset_ms: float,
    deviant_onset_ms: np.ndarray,
    found: np.ndarray,
) -> Dict[str, Any]:
    found = found.astype(bool)
    valid = found & np.isfinite(decision_time_ms)
    is_negative = valid & ((decision_time_ms - deviant_onset_ms) < 0.0)
    is_pre_p4 = valid & (decision_time_ms < float(p4_onset_ms))
    is_pre_dev = valid & (decision_time_ms < deviant_onset_ms)
    is_valid_after_p4 = valid & (decision_time_ms >= float(p4_onset_ms))
    return {
        "negative_rt_rate": float(np.mean(is_negative[valid])) if np.any(valid) else float("nan"),
        "pre_p4_crossing_rate": float(np.mean(is_pre_p4[valid])) if np.any(valid) else float("nan"),
        "pre_deviant_crossing_rate": float(np.mean(is_pre_dev[valid])) if np.any(valid) else float("nan"),
        "valid_after_p4_rate": float(np.mean(is_valid_after_p4[valid])) if np.any(valid) else float("nan"),
    }


def compute_bayesian_cost_rt(
    probs_over_time: np.ndarray,
    time_ms: np.ndarray,
    timeout_ms: float,
    *,
    w: Optional[float] = None,
    kappa: float = 0.5,
    min_time_ms: Optional[float] = None,
    return_diagnostics: bool = True,
    confidence_source: str = "p_correct",
) -> Dict[str, Any]:
    """Online deterministic expected-cost first-crossing readout.

    Decision-theoretic interpretation:
    - EC(t) combines error cost and time cost.
    - w converts elapsed time into error-cost units.
    - w = 1 / timeout_ms means waiting until timeout costs one error unit.
    - kappa is the acceptable expected-cost criterion.
    - The response is the first online time point where EC(t) < kappa.

    Parameters
    ----------
    probs_over_time:
        Array with shape [n_timepoints, n_classes] if using ``p_max``, or
        [n_timepoints] / [n_timepoints, 1] if using ``p_correct``.
    time_ms:
        Elapsed time since the chosen reference, in milliseconds.
    timeout_ms:
        Maximum allowed response time.
    confidence_source:
        ``"p_correct"`` or ``"p_max"``.
    """
    probs = np.asarray(probs_over_time, dtype=float)
    t_ms = np.asarray(time_ms, dtype=float).reshape(-1)
    if probs.ndim == 2 and probs.shape[0] != t_ms.shape[0]:
        raise ValueError("probs_over_time and time_ms must agree on n_timepoints")
    if probs.ndim == 1 and probs.shape[0] != t_ms.shape[0]:
        raise ValueError("probs_over_time and time_ms must agree on n_timepoints")
    if probs.ndim not in (1, 2):
        raise ValueError("probs_over_time must be 1D or 2D")

    timeout_ms = float(timeout_ms)
    if not np.isfinite(timeout_ms) or timeout_ms <= 0.0:
        raise ValueError("timeout_ms must be positive and finite")
    if w is None:
        w = 1.0 / timeout_ms
    w = float(w)
    kappa = float(kappa)
    min_time_ms = None if min_time_ms is None else float(min_time_ms)

    if confidence_source == "p_max":
        if probs.ndim != 2:
            raise ValueError("p_max readout requires probs_over_time shape [T, C]")
        confidence = np.nanmax(probs, axis=1)
        choice_over_time = np.nanargmax(probs, axis=1)
    elif confidence_source == "p_correct":
        if probs.ndim == 2:
            if probs.shape[1] != 1:
                raise ValueError("p_correct readout expects [T] or [T,1]")
            confidence = probs[:, 0]
        else:
            confidence = probs
        choice_over_time = None
    else:
        raise ValueError(f"Unknown confidence_source: {confidence_source}")

    ec = (1.0 - confidence) + (w * t_ms)
    eligible = np.isfinite(ec) & np.isfinite(t_ms) & (t_ms <= timeout_ms)
    if min_time_ms is not None:
        eligible &= (t_ms >= min_time_ms)
    hit_mask = eligible & (ec < kappa)
    hit_idx = int(np.flatnonzero(hit_mask)[0]) if np.any(hit_mask) else -1

    crossed = hit_idx >= 0
    selected_idx = hit_idx if crossed else -1
    selected_rt_ms = float(t_ms[hit_idx]) if crossed else float(timeout_ms)
    selected_choice = int(choice_over_time[hit_idx]) if (crossed and choice_over_time is not None) else -1
    selected_ec = float(ec[hit_idx]) if crossed else float("nan")
    selected_conf = float(confidence[hit_idx]) if crossed else float("nan")

    out = {
        "crossed": bool(crossed),
        "selected_index": int(selected_idx),
        "selected_rt_ms": float(selected_rt_ms),
        "selected_choice": int(selected_choice),
        "selected_ec": float(selected_ec),
        "selected_confidence": float(selected_conf),
        "timeout": bool(not crossed),
        "w": float(w),
        "kappa": float(kappa),
        "timeout_ms": float(timeout_ms),
        "confidence_source": str(confidence_source),
    }
    if return_diagnostics:
        out["ec_over_time"] = ec
        out["confidence_over_time"] = confidence
    return out


def run_rt_readout_sweeps(
    prepared: PreparedRTReadout,
    rt_readout_mode: str = "both",
    p_threshold_list: Sequence[float] = (0.5,),
    logit_margin_threshold_list: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    msprt_threshold_list: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0),
    cost_w_list: Sequence[float] = (0.001,),
    cost_timeout_ms_list: Sequence[float] = (),
    cost_threshold_list: Sequence[float] = (0.5,),
    k_consec_list: Sequence[int] = (1, 3, 5),
    bayes_error_cost: float = 1.0,
    bayes_time_cost_list: Sequence[float] = (0.00025,),
    bayes_threshold_start_list: Sequence[float] = (0.90,),
    bayes_threshold_min_list: Sequence[float] = (0.34,),
    bayes_urgency_slope_list: Sequence[float] = (0.0005,),
    bayes_evidence_bound_list: Sequence[float] = (25.0,),
    bayes_leak_list: Sequence[float] = (0.0,),
    bayes_wait_lag_ms_list: Sequence[float] = (100.0,),
    bayes_min_p_list: Sequence[float] = (0.5,),
    bayes_k_consec_list: Sequence[int] = (),
    stochastic_beta_list: Sequence[float] = (10.0,),
    stochastic_b0_list: Sequence[float] = (1.0,),
    stochastic_seed_list: Sequence[int] = (0,),
    bayes_force_deadline: bool = True,
    decision_not_before: str = "window_start",
    cost_elapsed_reference: str = "window_start",
    decision_min_elapsed_ms: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    p_corr = prepared.probs_correct
    p_max = prepared.probs_max
    logits_dev = np.asarray(prepared.logits_deviant, dtype=float)
    entropy = prepared.entropy
    logit_margin = prepared.logits_margin
    start = prepared.window_start
    end_excl = prepared.window_end_exclusive
    pos = prepared.y_pos_456
    N = p_corr.shape[0]

    if cost_timeout_ms_list:
        w_specs = [(1.0 / float(t), float(t)) for t in cost_timeout_ms_list]
    else:
        w_specs = [(float(w), (1.0 / float(w) if float(w) > 0 else float("nan"))) for w in cost_w_list]

    trial_rows: List[Dict[str, Any]] = []
    cond_rows: List[Dict[str, Any]] = []
    include_simple = rt_readout_mode in ("simple_threshold", "simple_threshold_pmax", "both")
    include_masked_logits = rt_readout_mode in ("masked_logits_threshold", "both")
    include_pcorrect_argmax = rt_readout_mode in ("p_correct_argmax", "both")
    include_pcorrect_argmax_correct = rt_readout_mode in ("p_correct_argmax_correct", "both")
    include_entropy = rt_readout_mode == "entropy_threshold"
    include_margin = rt_readout_mode == "simple_threshold_logit_margin"
    include_msprt = rt_readout_mode == "msprt_threshold"
    include_cost = rt_readout_mode in ("bayesian_cost", "expected_cost_threshold", "both")
    include_argmin = rt_readout_mode in ("bayes_cost_argmin", "both")
    include_deadline = rt_readout_mode == "bayesian_deadline"
    include_accumulator = rt_readout_mode == "bayesian_accumulator"
    include_pcorrect_accumulator = rt_readout_mode == "p_correct_accumulator"
    include_baseline_simple = rt_readout_mode == "baseline_corrected_simple_threshold"
    include_baseline_dynamic_margin = rt_readout_mode == "baseline_corrected_dynamic_margin"
    include_baseline_dynamic_pmax = rt_readout_mode == "baseline_corrected_dynamic_pmax"
    include_baseline_ec = rt_readout_mode == "baseline_corrected_ec"
    include_marginal_wait = rt_readout_mode == "bayesian_marginal_wait"
    include_online_cost = rt_readout_mode == "bayesian_online_cost"
    include_stochastic_online = rt_readout_mode == "stochastic_online_commit"
    include_advisor_stochastic = rt_readout_mode == "advisor_bayes_stochastic"
    include_advisor_dp = rt_readout_mode == "advisor_expected_cost_dp"
    if not (include_simple or include_masked_logits or include_pcorrect_argmax or include_pcorrect_argmax_correct or include_entropy or include_margin or include_msprt or include_cost or include_argmin or include_deadline or include_accumulator or include_pcorrect_accumulator or include_baseline_simple or include_baseline_dynamic_margin or include_baseline_dynamic_pmax or include_baseline_ec or include_marginal_wait or include_online_cost or include_stochastic_online or include_advisor_stochastic or include_advisor_dp):
        raise ValueError(f"Unknown rt_readout_mode: {rt_readout_mode}")
    p4_onset_ms_arr = np.asarray(prepared.p4_onset_ms, dtype=float)
    p4_onset_ms = float(np.nanmedian(p4_onset_ms_arr)) if np.any(np.isfinite(p4_onset_ms_arr)) else float("nan")

    def event_tokens(name: str) -> np.ndarray:
        if name == "window_start":
            return start.astype(np.int64)
        if name == "deviant_onset":
            return np.ceil(np.asarray(prepared.deviant_onset_ms, dtype=float) / float(prepared.token_ms)).astype(np.int64)
        if name == "p4_onset":
            return np.ceil(np.asarray(prepared.p4_onset_ms, dtype=float) / float(prepared.token_ms)).astype(np.int64)
        raise ValueError(f"Unknown decision_not_before/cost_elapsed_reference: {name}")

    decision_floor = event_tokens(str(decision_not_before))
    min_elapsed_tokens = int(max(0, math.ceil(float(decision_min_elapsed_ms) / float(prepared.token_ms))))
    effective_start = np.maximum(start.astype(np.int64), decision_floor.astype(np.int64))
    effective_start = np.maximum(effective_start, start.astype(np.int64) + min_elapsed_tokens)

    def elapsed_for_cost(token_indices: np.ndarray, i: int) -> np.ndarray:
        token_time = np.asarray(token_indices, dtype=float) * float(prepared.token_ms)
        ref = str(cost_elapsed_reference)
        if ref == "window_start":
            return token_time - float(prepared.window_start_time_ms[i])
        if ref == "deviant_onset":
            return np.maximum(0.0, token_time - float(prepared.deviant_onset_ms[i]))
        if ref == "p4_onset":
            return token_time - float(prepared.p4_onset_ms[i])
        raise ValueError(f"Unknown cost_elapsed_reference: {ref}")

    def variant_meta() -> Dict[str, Any]:
        return {
            "decision_not_before": str(decision_not_before),
            "cost_elapsed_reference": str(cost_elapsed_reference),
            "decision_min_elapsed_ms": float(decision_min_elapsed_ms),
        }

    # Baseline-corrected readouts subtract the inherited trial-start logits.
    # This prevents carryover priors from producing zero-latency raw-confidence crossings.
    baseline_token = np.zeros((N,), dtype=int)
    z_start = logits_dev[np.arange(N), baseline_token, :]
    delta_logits = logits_dev - z_start[:, None, :]
    delta_shift = delta_logits - np.max(delta_logits, axis=-1, keepdims=True)
    delta_exp = np.exp(delta_shift)
    delta_probs = delta_exp / np.maximum(np.sum(delta_exp, axis=-1, keepdims=True), 1e-12)
    delta_pred = delta_probs.argmax(axis=-1).astype(np.int64)
    delta_pmax = delta_probs.max(axis=-1)
    delta_sorted = np.sort(delta_probs, axis=-1)
    delta_margin = delta_sorted[..., -1] - delta_sorted[..., -2]
    delta_entropy = -(delta_probs * np.log(np.clip(delta_probs, 1e-12, 1.0))).sum(axis=-1)
    raw_start_probs = np.exp(z_start - np.max(z_start, axis=-1, keepdims=True))
    raw_start_probs = raw_start_probs / np.maximum(np.sum(raw_start_probs, axis=-1, keepdims=True), 1e-12)
    raw_start_pmax = raw_start_probs.max(axis=-1)
    raw_start_pred = raw_start_probs.argmax(axis=-1).astype(np.int64)
    raw_start_entropy = -(raw_start_probs * np.log(np.clip(raw_start_probs, 1e-12, 1.0))).sum(axis=-1)
    start_saturated = raw_start_pmax > 0.95
    delta_p_corr = delta_probs[
        np.arange(N)[:, None],
        np.arange(delta_probs.shape[1])[None, :],
        prepared.y_cls[:, None],
    ]

    if include_simple:
        simple_signal = p_corr if rt_readout_mode != "simple_threshold_pmax" else p_max
        simple_mode_name = "simple_threshold_pmax" if rt_readout_mode == "simple_threshold_pmax" else "simple_threshold"
        for thr in p_threshold_list:
            for k in k_consec_list:
                rt_ms = np.full((N,), np.nan, dtype=float)
                found = np.zeros((N,), dtype=bool)
                decision_token = np.full((N,), -1, dtype=int)
                p_at_rt = np.full((N,), np.nan, dtype=float)
                pmax_at_rt = np.full((N,), np.nan, dtype=float)
                for i in range(N):
                    s = int(effective_start[i]); e = int(end_excl[i])
                    w = simple_signal[i, s:e]
                    idx = _first_k_consecutive(w >= float(thr), int(k))
                    if idx >= 0:
                        found[i] = True
                        decision_token[i] = s + idx
                        rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                        p_at_rt[i] = float(w[idx])
                        pmax_at_rt[i] = float(p_max[i, s:e][idx])
                cond_meta = {
                    "readout_mode": simple_mode_name,
                    "w": float("nan"),
                    "timeout_ms": float("nan"),
                    "cost_threshold": float("nan"),
                    "p_threshold": float(thr),
                    "k_consec": int(k),
                    **variant_meta(),
                }
                cond_row = {
                    **cond_meta,
                    **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                    **_decision_validity_stats(
                        decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                        p4_onset_ms=p4_onset_ms,
                        deviant_onset_ms=prepared.deviant_onset_ms,
                        found=found,
                    ),
                    "accuracy_at_rt": float(
                        np.mean(
                            (
                                prepared.pred_class[np.arange(N)[found], decision_token[found]]
                                == prepared.y_cls[found]
                            ).astype(float)
                        )
                    ) if np.any(found) else float("nan"),
                    "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                    "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                    "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                    "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                    "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                    "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                    "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                }
                cond_rows.append(cond_row)
                for i in range(N):
                    trial_rows.append({
                        **cond_meta,
                        "trial_index": int(i),
                        "position": int(pos[i]),
                        "model_rt_ms": float(rt_ms[i]),
                        "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                        "found_flag": bool(found[i]),
                        "decision_token": int(decision_token[i]),
                        "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                        "window_start_token": int(prepared.window_start[i]),
                        "window_end_token": int(prepared.window_end_exclusive[i]),
                        "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                        "p4_onset_ms": float(p4_onset_ms),
                        "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                        "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                        "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                        "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                        "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                        "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                        "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                        "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                        "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                        "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                        "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                        "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                        "p_correct_at_rt": float(p_at_rt[i]),
                        "p_max_at_rt": float(pmax_at_rt[i]),
                        "expected_cost_at_rt": float("nan"),
                        "pred_class": int(prepared.pred_class[i, decision_token[i]]) if decision_token[i] >= 0 else -1,
                        "true_class": int(prepared.y_cls[i]),
                        "correct_at_rt": bool(decision_token[i] >= 0 and int(prepared.pred_class[i, decision_token[i]]) == int(prepared.y_cls[i])),
                        "found": bool(found[i]),
                        "forced_deadline": False,
                        "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                        "threshold_t": float(thr),
                        "cost_answer_now": float("nan"),
                        "bayes_error_cost": float("nan"),
                        "bayes_time_cost": float("nan"),
                        "bayes_threshold_start": float("nan"),
                        "bayes_threshold_min": float("nan"),
                        "bayes_urgency_slope": float("nan"),
                        "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                        "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                        "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                        "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                        "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                        "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                        "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                        "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                    })

    if include_baseline_simple:
        for thr in p_threshold_list:
            for k in k_consec_list:
                rt_ms = np.full((N,), np.nan, dtype=float)
                found = np.zeros((N,), dtype=bool)
                decision_token = np.full((N,), -1, dtype=int)
                p_delta_at_rt = np.full((N,), np.nan, dtype=float)
                pmax_delta_at_rt = np.full((N,), np.nan, dtype=float)
                p_raw_at_rt = np.full((N,), np.nan, dtype=float)
                pmax_raw_at_rt = np.full((N,), np.nan, dtype=float)
                delta_choice_at_rt = np.full((N,), -1, dtype=int)
                raw_choice_at_rt = np.full((N,), -1, dtype=int)
                for i in range(N):
                    s = int(effective_start[i])
                    e = int(end_excl[i])
                    if e <= s:
                        continue
                    w = delta_p_corr[i, s:e]
                    idx = _first_k_consecutive(w >= float(thr), int(k))
                    if idx < 0:
                        continue
                    t = s + idx
                    found[i] = True
                    decision_token[i] = t
                    rt_ms[i] = float((t * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                    p_delta_at_rt[i] = float(delta_p_corr[i, t])
                    pmax_delta_at_rt[i] = float(delta_pmax[i, t])
                    p_raw_at_rt[i] = float(p_corr[i, t])
                    pmax_raw_at_rt[i] = float(p_max[i, t])
                    delta_choice_at_rt[i] = int(delta_pred[i, t])
                    raw_choice_at_rt[i] = int(prepared.pred_class[i, t])
                cond_meta = {
                    "readout_mode": "baseline_corrected_simple_threshold",
                    "w": float("nan"),
                    "timeout_ms": float("nan"),
                    "cost_threshold": float("nan"),
                    "p_threshold": float(thr),
                    "k_consec": int(k),
                    "readout_type": "baseline_corrected_delta_logits_simple_threshold",
                    **variant_meta(),
                }
                cond_row = {
                    **cond_meta,
                    **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                    **_decision_validity_stats(
                        decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                        p4_onset_ms=p4_onset_ms,
                        deviant_onset_ms=prepared.deviant_onset_ms,
                        found=found,
                    ),
                    "accuracy_at_rt": float(np.mean((delta_choice_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                    "raw_accuracy_at_rt": float(np.mean((raw_choice_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                    "mean_raw_start_pmax": float(np.nanmean(raw_start_pmax)),
                    "mean_raw_start_entropy": float(np.nanmean(raw_start_entropy)),
                    "start_saturated_rate": float(np.mean(start_saturated.astype(float))),
                    "mean_p_delta_correct_at_rt": float(np.nanmean(p_delta_at_rt)) if np.any(np.isfinite(p_delta_at_rt)) else float("nan"),
                    "mean_pmax_delta_at_rt": float(np.nanmean(pmax_delta_at_rt)) if np.any(np.isfinite(pmax_delta_at_rt)) else float("nan"),
                    "mean_p_raw_correct_at_rt": float(np.nanmean(p_raw_at_rt)) if np.any(np.isfinite(p_raw_at_rt)) else float("nan"),
                    "mean_pmax_raw_at_rt": float(np.nanmean(pmax_raw_at_rt)) if np.any(np.isfinite(pmax_raw_at_rt)) else float("nan"),
                    "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                    "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                    "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                    "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                    "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                    "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                    "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                }
                cond_rows.append(cond_row)
                for i in range(N):
                    t = int(decision_token[i])
                    trial_rows.append({
                        **cond_meta,
                        "trial_index": int(i),
                        "position": int(pos[i]),
                        "model_rt_ms": float(rt_ms[i]),
                        "decision_time_ms": float(t * prepared.token_ms) if t >= 0 else float("nan"),
                        "found_flag": bool(found[i]),
                        "decision_token": int(t),
                        "decision_token_relative_to_dev_onset": int(t - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if t >= 0 else -1,
                        "window_start_token": int(prepared.window_start[i]),
                        "window_end_token": int(prepared.window_end_exclusive[i]),
                        "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                        "p4_onset_ms": float(p4_onset_ms),
                        "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                        "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                        "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                        "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                        "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                        "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                        "decision_minus_p4_onset_ms": float((t * prepared.token_ms) - p4_onset_ms) if t >= 0 else float("nan"),
                        "decision_minus_deviant_onset_ms": float((t * prepared.token_ms) - prepared.deviant_onset_ms[i]) if t >= 0 else float("nan"),
                        "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                        "is_pre_p4_crossing": bool(t >= 0 and (t * prepared.token_ms) < p4_onset_ms),
                        "is_pre_deviant_crossing": bool(t >= 0 and (t * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                        "is_valid_after_p4_onset": bool(t >= 0 and (t * prepared.token_ms) >= p4_onset_ms),
                        "p_correct_at_rt": float(p_delta_at_rt[i]),
                        "p_max_at_rt": float(pmax_delta_at_rt[i]),
                        "p_delta_correct_at_rt": float(p_delta_at_rt[i]),
                        "pmax_delta_at_rt": float(pmax_delta_at_rt[i]),
                        "p_raw_correct_at_rt": float(p_raw_at_rt[i]),
                        "pmax_raw_at_rt": float(pmax_raw_at_rt[i]),
                        "raw_start_p4": float(raw_start_probs[i, 0]),
                        "raw_start_p5": float(raw_start_probs[i, 1]),
                        "raw_start_p6": float(raw_start_probs[i, 2]),
                        "raw_start_pmax": float(raw_start_pmax[i]),
                        "raw_start_entropy": float(raw_start_entropy[i]),
                        "raw_start_pred_class": int(raw_start_pred[i]),
                        "expected_cost_at_rt": float("nan"),
                        "pred_class": int(delta_choice_at_rt[i]),
                        "raw_pred_class": int(raw_choice_at_rt[i]),
                        "true_class": int(prepared.y_cls[i]),
                        "correct_at_rt": bool(found[i] and int(delta_choice_at_rt[i]) == int(prepared.y_cls[i])),
                        "correct_at_rt_raw": bool(found[i] and int(raw_choice_at_rt[i]) == int(prepared.y_cls[i])),
                        "found": bool(found[i]),
                        "forced_deadline": False,
                        "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                        "threshold_t": float(thr),
                        "cost_answer_now": float("nan"),
                        "bayes_error_cost": float("nan"),
                        "bayes_time_cost": float("nan"),
                        "bayes_threshold_start": float("nan"),
                        "bayes_threshold_min": float("nan"),
                        "bayes_urgency_slope": float("nan"),
                        "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                        "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                        "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                        "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                        "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                        "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                        "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                        "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                    })

    if include_masked_logits:
        token_indices = np.arange(logits_dev.shape[1], dtype=int)
        active_mask_by_t = np.zeros((logits_dev.shape[1], 3), dtype=bool)
        active_mask_by_t[token_indices < 60, :] = True
        active_mask_by_t[(token_indices >= 60) & (token_indices < 75), 1:] = True
        active_mask_by_t[token_indices >= 75, 2] = True

        masked_logits = np.where(active_mask_by_t[None, :, :], logits_dev, -1e9)
        shifted = masked_logits - np.max(masked_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        denom = np.sum(exp_logits, axis=-1, keepdims=True)
        masked_probs = exp_logits / np.maximum(denom, 1e-12)
        masked_pred = masked_probs.argmax(axis=-1).astype(np.int64)
        masked_pmax = masked_probs.max(axis=-1)
        masked_pcorr = np.zeros((N, logits_dev.shape[1]), dtype=float)
        for i in range(N):
            y = int(prepared.y_cls[i])
            active_y = active_mask_by_t[:, y]
            masked_pcorr[i, active_y] = masked_probs[i, active_y, y]

        for thr in p_threshold_list:
            rt_ms = np.full((N,), np.nan, dtype=float)
            found = np.zeros((N,), dtype=bool)
            decision_token = np.full((N,), -1, dtype=int)
            p_at_rt = np.full((N,), np.nan, dtype=float)
            pmax_at_rt = np.full((N,), np.nan, dtype=float)
            pred_at_rt = np.full((N,), -1, dtype=int)
            raw_p_at_rt = np.full((N,), np.nan, dtype=float)
            raw_pmax_at_rt = np.full((N,), np.nan, dtype=float)
            raw_pred_at_rt = np.full((N,), -1, dtype=int)
            active_classes_at_rt = np.full((N,), "", dtype=object)
            for i in range(N):
                s = int(effective_start[i])
                e = int(end_excl[i])
                if e <= s:
                    continue
                w = masked_pcorr[i, s:e]
                hits = np.isfinite(w) & (w >= float(thr))
                if not np.any(hits):
                    continue
                idx = int(np.argmax(hits))
                t = s + idx
                found[i] = True
                decision_token[i] = t
                rt_ms[i] = float((t * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                p_at_rt[i] = float(masked_pcorr[i, t])
                pmax_at_rt[i] = float(masked_pmax[i, t])
                pred_at_rt[i] = int(masked_pred[i, t])
                raw_p_at_rt[i] = float(p_corr[i, t])
                raw_pmax_at_rt[i] = float(p_max[i, t])
                raw_pred_at_rt[i] = int(prepared.pred_class[i, t])
                active_classes_at_rt[i] = ",".join(str(c) for c in np.where(active_mask_by_t[t])[0].tolist())
            cond_meta = {
                "readout_mode": "masked_logits_threshold",
                "w": float("nan"),
                "timeout_ms": float("nan"),
                "cost_threshold": float("nan"),
                "p_threshold": float(thr),
                "k_consec": 1,
                **variant_meta(),
            }
            cond_row = {
                **cond_meta,
                **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                **_decision_validity_stats(
                    decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                    p4_onset_ms=p4_onset_ms,
                    deviant_onset_ms=prepared.deviant_onset_ms,
                    found=found,
                ),
                "accuracy_at_rt": float(np.mean((pred_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                "raw_accuracy_at_rt": float(np.mean((raw_pred_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                "p_correct_masked_at_rt_mean": float(np.nanmean(p_at_rt)) if np.any(np.isfinite(p_at_rt)) else float("nan"),
                "p_max_masked_at_rt_mean": float(np.nanmean(pmax_at_rt)) if np.any(np.isfinite(pmax_at_rt)) else float("nan"),
                "p_correct_raw_at_rt_mean": float(np.nanmean(raw_p_at_rt)) if np.any(np.isfinite(raw_p_at_rt)) else float("nan"),
                "p_max_raw_at_rt_mean": float(np.nanmean(raw_pmax_at_rt)) if np.any(np.isfinite(raw_pmax_at_rt)) else float("nan"),
                "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
            }
            cond_rows.append(cond_row)
            for i in range(N):
                trial_rows.append({
                    **cond_meta,
                    "trial_index": int(i),
                    "position": int(pos[i]),
                    "model_rt_ms": float(rt_ms[i]),
                    "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                    "found_flag": bool(found[i]),
                    "decision_token": int(decision_token[i]),
                    "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                    "window_start_token": int(prepared.window_start[i]),
                    "window_end_token": int(prepared.window_end_exclusive[i]),
                    "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                    "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                    "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                    "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                    "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                    "p4_onset_ms": float(p4_onset_ms),
                    "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                    "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                    "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                    "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                    "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                    "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                    "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                    "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                    "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                    "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                    "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                    "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                    "p_correct_at_rt": float(p_at_rt[i]),
                    "p_max_at_rt": float(pmax_at_rt[i]),
                    "p_correct_masked_at_rt": float(p_at_rt[i]),
                    "p_max_masked_at_rt": float(pmax_at_rt[i]),
                    "masked_pred_at_rt": int(pred_at_rt[i]),
                    "p_correct_raw_at_rt": float(raw_p_at_rt[i]),
                    "p_max_raw_at_rt": float(raw_pmax_at_rt[i]),
                    "raw_pred_at_rt": int(raw_pred_at_rt[i]),
                    "active_classes_at_rt": str(active_classes_at_rt[i]),
                    "expected_cost_at_rt": float("nan"),
                    "pred_class": int(pred_at_rt[i]),
                    "true_class": int(prepared.y_cls[i]),
                    "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_at_rt[i]) == int(prepared.y_cls[i])),
                    "correct_at_rt_masked": bool(decision_token[i] >= 0 and int(pred_at_rt[i]) == int(prepared.y_cls[i])),
                    "correct_at_rt_raw": bool(decision_token[i] >= 0 and int(raw_pred_at_rt[i]) == int(prepared.y_cls[i])),
                    "found": bool(found[i]),
                    "forced_deadline": False,
                    "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                    "threshold_t": float(thr),
                    "cost_answer_now": float("nan"),
                    "bayes_error_cost": float("nan"),
                    "bayes_time_cost": float("nan"),
                    "bayes_threshold_start": float("nan"),
                    "bayes_threshold_min": float("nan"),
                    "bayes_urgency_slope": float("nan"),
                    "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                    "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                    "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                    "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                    "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                    "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                    "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                    "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                })

    if include_pcorrect_argmax:
        rt_ms = np.full((N,), np.nan, dtype=float)
        found = np.zeros((N,), dtype=bool)
        decision_token = np.full((N,), -1, dtype=int)
        p_true_at_rt = np.full((N,), np.nan, dtype=float)
        pmax_at_rt = np.full((N,), np.nan, dtype=float)
        pred_class_at_rt = np.full((N,), -1, dtype=int)
        argmax_at_window_start = np.zeros((N,), dtype=bool)
        argmax_at_window_end = np.zeros((N,), dtype=bool)
        n_window_tokens = np.zeros((N,), dtype=int)
        for i in range(N):
            s = int(effective_start[i])
            e = int(end_excl[i])
            w = p_corr[i, s:e]
            n_window_tokens[i] = int(w.size)
            if w.size == 0:
                continue
            finite = np.isfinite(w)
            if not np.any(finite):
                continue
            idx = int(np.nanargmax(w))
            found[i] = True
            decision_token[i] = s + idx
            rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
            p_true_at_rt[i] = float(p_corr[i, decision_token[i]])
            pmax_at_rt[i] = float(p_max[i, decision_token[i]])
            pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
            argmax_at_window_start[i] = bool(idx == 0)
            argmax_at_window_end[i] = bool(idx == w.size - 1)
        cond_meta = {
            "readout_mode": "p_correct_argmax",
            "w": float("nan"),
            "timeout_ms": float("nan"),
            "cost_threshold": float("nan"),
            "p_threshold": float("nan"),
            "k_consec": 1,
            **variant_meta(),
        }
        cond_row = {
            **cond_meta,
            **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
            **_decision_validity_stats(
                decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                p4_onset_ms=p4_onset_ms,
                deviant_onset_ms=prepared.deviant_onset_ms,
                found=found,
            ),
            "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
            "argmax_at_window_start_rate": float(np.mean(argmax_at_window_start[found].astype(float))) if np.any(found) else float("nan"),
            "argmax_at_window_end_rate": float(np.mean(argmax_at_window_end[found].astype(float))) if np.any(found) else float("nan"),
            "n_unique_model_rt": int(np.unique(rt_ms[found & np.isfinite(rt_ms)]).size),
            "model_rt_std": float(np.std(rt_ms[found & np.isfinite(rt_ms)])) if np.any(found & np.isfinite(rt_ms)) else float("nan"),
            "mean_p_correct_at_argmax": float(np.nanmean(p_true_at_rt)) if np.any(np.isfinite(p_true_at_rt)) else float("nan"),
            "mean_p_max_at_argmax": float(np.nanmean(pmax_at_rt)) if np.any(np.isfinite(pmax_at_rt)) else float("nan"),
            "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
            "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
            "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
            "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
            "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
            "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
            "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
        }
        cond_rows.append(cond_row)
        for i in range(N):
            trial_rows.append({
                **cond_meta,
                "trial_index": int(i),
                "position": int(pos[i]),
                "model_rt_ms": float(rt_ms[i]),
                "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                "found_flag": bool(found[i]),
                "decision_token": int(decision_token[i]),
                "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                "window_start_token": int(prepared.window_start[i]),
                "window_end_token": int(prepared.window_end_exclusive[i]),
                "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                "p4_onset_ms": float(p4_onset_ms),
                "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                "p_correct_at_rt": float(p_true_at_rt[i]),
                "p_max_at_rt": float(pmax_at_rt[i]),
                "expected_cost_at_rt": float("nan"),
                "pred_class": int(pred_class_at_rt[i]),
                "true_class": int(prepared.y_cls[i]),
                "correct_at_rt": bool(found[i] and pred_class_at_rt[i] == prepared.y_cls[i]),
                "found": bool(found[i]),
                "forced_deadline": False,
                "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                "threshold_t": float("nan"),
                "cost_answer_now": float("nan"),
                "bayes_error_cost": float("nan"),
                "bayes_time_cost": float("nan"),
                "bayes_threshold_start": float("nan"),
                "bayes_threshold_min": float("nan"),
                "bayes_urgency_slope": float("nan"),
                "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                "argmax_at_window_start": bool(argmax_at_window_start[i]),
                "argmax_at_window_end": bool(argmax_at_window_end[i]),
                "argmax_token_index": int(decision_token[i]),
                "n_window_tokens": int(n_window_tokens[i]),
            })

    if include_pcorrect_argmax_correct:
        rt_ms = np.full((N,), np.nan, dtype=float)
        found = np.zeros((N,), dtype=bool)
        decision_token = np.full((N,), -1, dtype=int)
        p_true_at_rt = np.full((N,), np.nan, dtype=float)
        pmax_at_rt = np.full((N,), np.nan, dtype=float)
        pred_class_at_rt = np.full((N,), -1, dtype=int)
        argmax_at_window_start = np.zeros((N,), dtype=bool)
        argmax_at_window_end = np.zeros((N,), dtype=bool)
        n_window_tokens = np.zeros((N,), dtype=int)
        n_correct_argmax_tokens = np.zeros((N,), dtype=int)
        for i in range(N):
            s = int(effective_start[i])
            e = int(end_excl[i])
            w_true = p_corr[i, s:e]
            w_max = p_max[i, s:e]
            n_window_tokens[i] = int(w_true.size)
            if w_true.size == 0:
                continue
            true_is_max = np.isfinite(w_true) & np.isfinite(w_max) & np.isclose(w_true, w_max, rtol=1e-7, atol=1e-8)
            n_correct_argmax_tokens[i] = int(np.sum(true_is_max))
            if not np.any(true_is_max):
                continue
            masked = np.where(true_is_max, w_true, -np.inf)
            idx = int(np.argmax(masked))
            found[i] = True
            decision_token[i] = s + idx
            rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
            p_true_at_rt[i] = float(p_corr[i, decision_token[i]])
            pmax_at_rt[i] = float(p_max[i, decision_token[i]])
            pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
            argmax_at_window_start[i] = bool(idx == 0)
            argmax_at_window_end[i] = bool(idx == w_true.size - 1)
        cond_meta = {
            "readout_mode": "p_correct_argmax_correct",
            "w": float("nan"),
            "timeout_ms": float("nan"),
            "cost_threshold": float("nan"),
            "p_threshold": float("nan"),
            "k_consec": 1,
            **variant_meta(),
        }
        cond_row = {
            **cond_meta,
            **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
            **_decision_validity_stats(
                decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                p4_onset_ms=p4_onset_ms,
                deviant_onset_ms=prepared.deviant_onset_ms,
                found=found,
            ),
            "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
            "argmax_at_window_start_rate": float(np.mean(argmax_at_window_start[found].astype(float))) if np.any(found) else float("nan"),
            "argmax_at_window_end_rate": float(np.mean(argmax_at_window_end[found].astype(float))) if np.any(found) else float("nan"),
            "n_unique_model_rt": int(np.unique(rt_ms[found & np.isfinite(rt_ms)]).size),
            "model_rt_std": float(np.std(rt_ms[found & np.isfinite(rt_ms)])) if np.any(found & np.isfinite(rt_ms)) else float("nan"),
            "mean_p_correct_at_argmax": float(np.nanmean(p_true_at_rt)) if np.any(np.isfinite(p_true_at_rt)) else float("nan"),
            "mean_p_max_at_argmax": float(np.nanmean(pmax_at_rt)) if np.any(np.isfinite(pmax_at_rt)) else float("nan"),
            "mean_correct_argmax_tokens": float(np.mean(n_correct_argmax_tokens.astype(float))) if n_correct_argmax_tokens.size else float("nan"),
            "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
            "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
            "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
            "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
            "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
            "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
            "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
        }
        cond_rows.append(cond_row)
        for i in range(N):
            trial_rows.append({
                **cond_meta,
                "trial_index": int(i),
                "position": int(pos[i]),
                "model_rt_ms": float(rt_ms[i]),
                "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                "found_flag": bool(found[i]),
                "decision_token": int(decision_token[i]),
                "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                "window_start_token": int(prepared.window_start[i]),
                "window_end_token": int(prepared.window_end_exclusive[i]),
                "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                "p4_onset_ms": float(p4_onset_ms),
                "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                "p_correct_at_rt": float(p_true_at_rt[i]),
                "p_max_at_rt": float(pmax_at_rt[i]),
                "expected_cost_at_rt": float("nan"),
                "pred_class": int(pred_class_at_rt[i]),
                "true_class": int(prepared.y_cls[i]),
                "correct_at_rt": bool(found[i] and np.isclose(p_true_at_rt[i], pmax_at_rt[i], rtol=1e-7, atol=1e-8)),
                "found": bool(found[i]),
                "forced_deadline": False,
                "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                "threshold_t": float("nan"),
                "cost_answer_now": float("nan"),
                "bayes_error_cost": float("nan"),
                "bayes_time_cost": float("nan"),
                "bayes_threshold_start": float("nan"),
                "bayes_threshold_min": float("nan"),
                "bayes_urgency_slope": float("nan"),
                "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                "argmax_at_window_start": bool(argmax_at_window_start[i]),
                "argmax_at_window_end": bool(argmax_at_window_end[i]),
                "argmax_token_index": int(decision_token[i]),
                "n_window_tokens": int(n_window_tokens[i]),
                "n_correct_argmax_tokens": int(n_correct_argmax_tokens[i]),
            })

    if include_baseline_dynamic_margin or include_baseline_dynamic_pmax:
        mode_name = "baseline_corrected_dynamic_margin" if include_baseline_dynamic_margin else "baseline_corrected_dynamic_pmax"
        signal_all = delta_margin if include_baseline_dynamic_margin else delta_pmax
        signal_col = "margin_delta_at_rt" if include_baseline_dynamic_margin else "pmax_delta_at_rt"
        for threshold_start in bayes_threshold_start_list:
            for threshold_min in bayes_threshold_min_list:
                for urgency_slope in bayes_urgency_slope_list:
                    for k in k_consec_list:
                        rt_ms = np.full((N,), np.nan, dtype=float)
                        found = np.zeros((N,), dtype=bool)
                        decision_token = np.full((N,), -1, dtype=int)
                        signal_at_rt = np.full((N,), np.nan, dtype=float)
                        threshold_at_rt = np.full((N,), np.nan, dtype=float)
                        p_delta_at_rt = np.full((N,), np.nan, dtype=float)
                        pmax_delta_at_rt = np.full((N,), np.nan, dtype=float)
                        entropy_delta_at_rt = np.full((N,), np.nan, dtype=float)
                        p_raw_at_rt = np.full((N,), np.nan, dtype=float)
                        pmax_raw_at_rt = np.full((N,), np.nan, dtype=float)
                        delta_choice_at_rt = np.full((N,), -1, dtype=int)
                        raw_choice_at_rt = np.full((N,), -1, dtype=int)
                        for i in range(N):
                            s = int(effective_start[i])
                            e = int(end_excl[i])
                            if e <= s:
                                continue
                            token_idx = np.arange(s, e, dtype=float)
                            elapsed_ms = token_idx * float(prepared.token_ms) - float(prepared.rt_reference_time_ms[i])
                            threshold_curve = np.maximum(
                                float(threshold_min),
                                float(threshold_start) - float(urgency_slope) * elapsed_ms,
                            )
                            sig = signal_all[i, s:e]
                            hits = np.isfinite(sig) & (sig > threshold_curve)
                            idx = _first_k_consecutive(hits, int(k))
                            if idx < 0:
                                continue
                            t = s + idx
                            found[i] = True
                            decision_token[i] = t
                            rt_ms[i] = float((t * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                            signal_at_rt[i] = float(sig[idx])
                            threshold_at_rt[i] = float(threshold_curve[idx])
                            delta_choice_at_rt[i] = int(delta_pred[i, t])
                            raw_choice_at_rt[i] = int(prepared.pred_class[i, t])
                            p_delta_at_rt[i] = float(delta_probs[i, t, int(prepared.y_cls[i])])
                            pmax_delta_at_rt[i] = float(delta_pmax[i, t])
                            entropy_delta_at_rt[i] = float(delta_entropy[i, t])
                            p_raw_at_rt[i] = float(p_corr[i, t])
                            pmax_raw_at_rt[i] = float(p_max[i, t])
                        cond_meta = {
                            "readout_mode": mode_name,
                            "w": float("nan"),
                            "timeout_ms": float("nan"),
                            "cost_threshold": float("nan"),
                            "kappa": float("nan"),
                            "p_threshold": float("nan"),
                            "k_consec": int(k),
                            "bayes_error_cost": float("nan"),
                            "bayes_time_cost": float("nan"),
                            "bayes_threshold_start": float(threshold_start),
                            "bayes_threshold_min": float(threshold_min),
                            "bayes_urgency_slope": float(urgency_slope),
                            "readout_type": "baseline_corrected_delta_logits",
                            "effective_threshold_formula": "delta_signal(t)>max(threshold_min,threshold_start-urgency_slope*elapsed_ms)",
                            **variant_meta(),
                        }
                        cond_row = {
                            **cond_meta,
                            **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                            **_decision_validity_stats(
                                decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                                p4_onset_ms=p4_onset_ms,
                                deviant_onset_ms=prepared.deviant_onset_ms,
                                found=found,
                            ),
                            "accuracy_at_rt": float(np.mean((delta_choice_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                            "raw_accuracy_at_rt": float(np.mean((raw_choice_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                            "mean_raw_start_pmax": float(np.nanmean(raw_start_pmax)),
                            "mean_raw_start_entropy": float(np.nanmean(raw_start_entropy)),
                            "start_saturated_rate": float(np.mean(start_saturated.astype(float))),
                            "zero_ms_crossing_rate": float(np.mean((found & np.isfinite(rt_ms) & (rt_ms <= max(0.0, float(decision_min_elapsed_ms)))).astype(float))),
                            f"mean_{signal_col}": float(np.nanmean(signal_at_rt)) if np.any(np.isfinite(signal_at_rt)) else float("nan"),
                            "mean_threshold_at_rt": float(np.nanmean(threshold_at_rt)) if np.any(np.isfinite(threshold_at_rt)) else float("nan"),
                            "mean_p_delta_correct_at_rt": float(np.nanmean(p_delta_at_rt)) if np.any(np.isfinite(p_delta_at_rt)) else float("nan"),
                            "mean_pmax_delta_at_rt": float(np.nanmean(pmax_delta_at_rt)) if np.any(np.isfinite(pmax_delta_at_rt)) else float("nan"),
                            "mean_p_raw_correct_at_rt": float(np.nanmean(p_raw_at_rt)) if np.any(np.isfinite(p_raw_at_rt)) else float("nan"),
                            "mean_pmax_raw_at_rt": float(np.nanmean(pmax_raw_at_rt)) if np.any(np.isfinite(pmax_raw_at_rt)) else float("nan"),
                            "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                            "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                            "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                            "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                            "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                            "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                            "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                        }
                        cond_rows.append(cond_row)
                        for i in range(N):
                            t = int(decision_token[i])
                            trial_rows.append({
                                **cond_meta,
                                "trial_index": int(i),
                                "position": int(pos[i]),
                                "model_rt_ms": float(rt_ms[i]),
                                "decision_time_ms": float(t * prepared.token_ms) if t >= 0 else float("nan"),
                                "found_flag": bool(found[i]),
                                "decision_token": int(t),
                                "decision_token_relative_to_dev_onset": int(t - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if t >= 0 else -1,
                                "window_start_token": int(prepared.window_start[i]),
                                "window_end_token": int(prepared.window_end_exclusive[i]),
                                "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                                "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                                "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                                "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                                "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                                "p4_onset_ms": float(p4_onset_ms),
                                "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                                "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                                "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                                "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                                "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                                "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                                "decision_minus_p4_onset_ms": float((t * prepared.token_ms) - p4_onset_ms) if t >= 0 else float("nan"),
                                "decision_minus_deviant_onset_ms": float((t * prepared.token_ms) - prepared.deviant_onset_ms[i]) if t >= 0 else float("nan"),
                                "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                                "is_pre_p4_crossing": bool(t >= 0 and (t * prepared.token_ms) < p4_onset_ms),
                                "is_pre_deviant_crossing": bool(t >= 0 and (t * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                                "is_valid_after_p4_onset": bool(t >= 0 and (t * prepared.token_ms) >= p4_onset_ms),
                                "p_correct_at_rt": float(p_delta_at_rt[i]),
                                "p_max_at_rt": float(pmax_delta_at_rt[i]),
                                "p_delta_correct_at_rt": float(p_delta_at_rt[i]),
                                "pmax_delta_at_rt": float(pmax_delta_at_rt[i]),
                                "entropy_delta_at_rt": float(entropy_delta_at_rt[i]),
                                signal_col: float(signal_at_rt[i]),
                                "threshold_t": float(threshold_at_rt[i]),
                                "p_raw_correct_at_rt": float(p_raw_at_rt[i]),
                                "pmax_raw_at_rt": float(pmax_raw_at_rt[i]),
                                "raw_start_p4": float(raw_start_probs[i, 0]),
                                "raw_start_p5": float(raw_start_probs[i, 1]),
                                "raw_start_p6": float(raw_start_probs[i, 2]),
                                "raw_start_pmax": float(raw_start_pmax[i]),
                                "raw_start_entropy": float(raw_start_entropy[i]),
                                "raw_start_pred_class": int(raw_start_pred[i]),
                                "start_saturated": bool(start_saturated[i]),
                                "zero_ms_crossing": bool(found[i] and np.isfinite(rt_ms[i]) and rt_ms[i] <= max(0.0, float(decision_min_elapsed_ms))),
                                "expected_cost_at_rt": float("nan"),
                                "pred_class": int(delta_choice_at_rt[i]),
                                "raw_pred_class": int(raw_choice_at_rt[i]),
                                "true_class": int(prepared.y_cls[i]),
                                "correct_at_rt": bool(found[i] and int(delta_choice_at_rt[i]) == int(prepared.y_cls[i])),
                                "correct_at_rt_raw": bool(found[i] and int(raw_choice_at_rt[i]) == int(prepared.y_cls[i])),
                                "found": bool(found[i]),
                                "forced_deadline": False,
                                "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                                "cost_answer_now": float("nan"),
                                "bayes_error_cost": float("nan"),
                                "bayes_time_cost": float("nan"),
                                "bayes_threshold_start": float(threshold_start),
                                "bayes_threshold_min": float(threshold_min),
                                "bayes_urgency_slope": float(urgency_slope),
                                "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                                "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                                "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                                "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                                "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                                "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                                "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                                "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                            })

    if include_baseline_ec:
        for w, timeout_ms in w_specs:
            for kappa in cost_threshold_list:
                rt_ms = np.full((N,), np.nan, dtype=float)
                found = np.zeros((N,), dtype=bool)
                decision_token = np.full((N,), -1, dtype=int)
                ec_at_rt = np.full((N,), np.nan, dtype=float)
                pmax_delta_at_rt = np.full((N,), np.nan, dtype=float)
                p_delta_at_rt = np.full((N,), np.nan, dtype=float)
                delta_choice_at_rt = np.full((N,), -1, dtype=int)
                raw_choice_at_rt = np.full((N,), -1, dtype=int)
                for i in range(N):
                    s = int(effective_start[i])
                    e = int(end_excl[i])
                    if e <= s:
                        continue
                    token_idx = np.arange(s, e, dtype=float)
                    elapsed_ms = elapsed_for_cost(token_idx, i)
                    ec = (1.0 - delta_pmax[i, s:e]) + float(w) * elapsed_ms
                    hits = np.isfinite(ec) & (ec < float(kappa))
                    idx = _first_k_consecutive(hits, 1)
                    if idx < 0:
                        continue
                    t = s + idx
                    found[i] = True
                    decision_token[i] = t
                    rt_ms[i] = float((t * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                    ec_at_rt[i] = float(ec[idx])
                    pmax_delta_at_rt[i] = float(delta_pmax[i, t])
                    p_delta_at_rt[i] = float(delta_probs[i, t, int(prepared.y_cls[i])])
                    delta_choice_at_rt[i] = int(delta_pred[i, t])
                    raw_choice_at_rt[i] = int(prepared.pred_class[i, t])
                cond_meta = {
                    "readout_mode": "baseline_corrected_ec",
                    "w": float(w),
                    "timeout_ms": float(timeout_ms),
                    "cost_threshold": float(kappa),
                    "kappa": float(kappa),
                    "p_threshold": float("nan"),
                    "k_consec": 1,
                    "readout_type": "baseline_corrected_delta_logits_ec",
                    "effective_threshold_formula": "EC_delta(t)=1-pmax_delta(t)+w*t; respond when EC_delta<kappa",
                    **variant_meta(),
                }
                cond_row = {
                    **cond_meta,
                    **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                    **_decision_validity_stats(
                        decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                        p4_onset_ms=p4_onset_ms,
                        deviant_onset_ms=prepared.deviant_onset_ms,
                        found=found,
                    ),
                    "accuracy_at_rt": float(np.mean((delta_choice_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                    "raw_accuracy_at_rt": float(np.mean((raw_choice_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                    "mean_raw_start_pmax": float(np.nanmean(raw_start_pmax)),
                    "mean_raw_start_entropy": float(np.nanmean(raw_start_entropy)),
                    "start_saturated_rate": float(np.mean(start_saturated.astype(float))),
                    "zero_ms_crossing_rate": float(np.mean((found & np.isfinite(rt_ms) & (rt_ms <= max(0.0, float(decision_min_elapsed_ms)))).astype(float))),
                    "mean_expected_cost_at_rt": float(np.nanmean(ec_at_rt)) if np.any(np.isfinite(ec_at_rt)) else float("nan"),
                    "mean_p_delta_correct_at_rt": float(np.nanmean(p_delta_at_rt)) if np.any(np.isfinite(p_delta_at_rt)) else float("nan"),
                    "mean_pmax_delta_at_rt": float(np.nanmean(pmax_delta_at_rt)) if np.any(np.isfinite(pmax_delta_at_rt)) else float("nan"),
                }
                cond_rows.append(cond_row)
                for i in range(N):
                    t = int(decision_token[i])
                    trial_rows.append({
                        **cond_meta,
                        "trial_index": int(i),
                        "position": int(pos[i]),
                        "model_rt_ms": float(rt_ms[i]),
                        "decision_time_ms": float(t * prepared.token_ms) if t >= 0 else float("nan"),
                        "found_flag": bool(found[i]),
                        "decision_token": int(t),
                        "decision_token_relative_to_dev_onset": int(t - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if t >= 0 else -1,
                        "window_start_token": int(prepared.window_start[i]),
                        "window_end_token": int(prepared.window_end_exclusive[i]),
                        "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                        "p4_onset_ms": float(p4_onset_ms),
                        "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                        "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                        "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                        "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                        "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                        "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                        "decision_minus_p4_onset_ms": float((t * prepared.token_ms) - p4_onset_ms) if t >= 0 else float("nan"),
                        "decision_minus_deviant_onset_ms": float((t * prepared.token_ms) - prepared.deviant_onset_ms[i]) if t >= 0 else float("nan"),
                        "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                        "is_pre_p4_crossing": bool(t >= 0 and (t * prepared.token_ms) < p4_onset_ms),
                        "is_pre_deviant_crossing": bool(t >= 0 and (t * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                        "is_valid_after_p4_onset": bool(t >= 0 and (t * prepared.token_ms) >= p4_onset_ms),
                        "p_correct_at_rt": float(p_delta_at_rt[i]),
                        "p_max_at_rt": float(pmax_delta_at_rt[i]),
                        "p_delta_correct_at_rt": float(p_delta_at_rt[i]),
                        "pmax_delta_at_rt": float(pmax_delta_at_rt[i]),
                        "expected_cost_at_rt": float(ec_at_rt[i]),
                        "raw_start_p4": float(raw_start_probs[i, 0]),
                        "raw_start_p5": float(raw_start_probs[i, 1]),
                        "raw_start_p6": float(raw_start_probs[i, 2]),
                        "raw_start_pmax": float(raw_start_pmax[i]),
                        "raw_start_entropy": float(raw_start_entropy[i]),
                        "raw_start_pred_class": int(raw_start_pred[i]),
                        "start_saturated": bool(start_saturated[i]),
                        "zero_ms_crossing": bool(found[i] and np.isfinite(rt_ms[i]) and rt_ms[i] <= max(0.0, float(decision_min_elapsed_ms))),
                        "pred_class": int(delta_choice_at_rt[i]),
                        "raw_pred_class": int(raw_choice_at_rt[i]),
                        "true_class": int(prepared.y_cls[i]),
                        "correct_at_rt": bool(found[i] and int(delta_choice_at_rt[i]) == int(prepared.y_cls[i])),
                        "correct_at_rt_raw": bool(found[i] and int(raw_choice_at_rt[i]) == int(prepared.y_cls[i])),
                        "found": bool(found[i]),
                        "forced_deadline": False,
                        "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                        "threshold_t": float(kappa),
                        "cost_answer_now": float(ec_at_rt[i]),
                        "bayes_error_cost": float("nan"),
                        "bayes_time_cost": float(w),
                        "bayes_threshold_start": float("nan"),
                        "bayes_threshold_min": float("nan"),
                        "bayes_urgency_slope": float("nan"),
                        "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                        "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                        "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                        "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                        "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                        "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                        "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                        "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                    })

    if include_entropy:
        for thr in p_threshold_list:
            for k in k_consec_list:
                rt_ms = np.full((N,), np.nan, dtype=float)
                found = np.zeros((N,), dtype=bool)
                decision_token = np.full((N,), -1, dtype=int)
                entropy_at_rt = np.full((N,), np.nan, dtype=float)
                p_true_at_rt = np.full((N,), np.nan, dtype=float)
                pmax_at_rt = np.full((N,), np.nan, dtype=float)
                pred_class_at_rt = np.full((N,), -1, dtype=int)
                for i in range(N):
                    s = int(effective_start[i]); e = int(end_excl[i])
                    w = entropy[i, s:e]
                    idx = _first_k_consecutive(w < float(thr), int(k))
                    if idx >= 0:
                        found[i] = True
                        decision_token[i] = s + idx
                        rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                        entropy_at_rt[i] = float(w[idx])
                        p_true_at_rt[i] = float(p_corr[i, decision_token[i]])
                        pmax_at_rt[i] = float(p_max[i, decision_token[i]])
                        pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                cond_meta = {
                    "readout_mode": "entropy_threshold",
                    "w": float("nan"),
                    "timeout_ms": float("nan"),
                    "cost_threshold": float("nan"),
                    "entropy_threshold": float(thr),
                    "p_threshold": float("nan"),
                    "k_consec": int(k),
                    **variant_meta(),
                }
                cond_row = {
                    **cond_meta,
                    **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                    **_decision_validity_stats(
                        decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                        p4_onset_ms=p4_onset_ms,
                        deviant_onset_ms=prepared.deviant_onset_ms,
                        found=found,
                    ),
                    "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                    "mean_entropy_at_rt": float(np.nanmean(entropy_at_rt)) if np.any(np.isfinite(entropy_at_rt)) else float("nan"),
                    "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                    "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                    "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                    "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                    "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                    "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                    "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                }
                cond_rows.append(cond_row)
                for i in range(N):
                    trial_rows.append({
                        **cond_meta,
                        "trial_index": int(i),
                        "position": int(pos[i]),
                        "model_rt_ms": float(rt_ms[i]),
                        "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                        "found_flag": bool(found[i]),
                        "decision_token": int(decision_token[i]),
                        "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                        "window_start_token": int(prepared.window_start[i]),
                        "window_end_token": int(prepared.window_end_exclusive[i]),
                        "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                        "p4_onset_ms": float(p4_onset_ms),
                        "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                        "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                        "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                        "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                        "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                        "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                        "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                        "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                        "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                        "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                        "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                        "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                        "entropy_at_rt": float(entropy_at_rt[i]),
                        "p_correct_at_rt": float(p_true_at_rt[i]),
                        "p_max_at_rt": float(pmax_at_rt[i]),
                        "expected_cost_at_rt": float("nan"),
                        "pred_class": int(pred_class_at_rt[i]) if decision_token[i] >= 0 else -1,
                        "true_class": int(prepared.y_cls[i]),
                        "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                        "found": bool(found[i]),
                        "forced_deadline": False,
                        "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                        "threshold_t": float(thr),
                        "cost_answer_now": float("nan"),
                        "bayes_error_cost": float("nan"),
                        "bayes_time_cost": float("nan"),
                        "bayes_threshold_start": float("nan"),
                        "bayes_threshold_min": float("nan"),
                        "bayes_urgency_slope": float("nan"),
                        "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                        "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                        "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                        "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                        "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                        "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                        "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                        "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                    })

    if include_margin:
        for thr in logit_margin_threshold_list:
            for k in k_consec_list:
                rt_ms = np.full((N,), np.nan, dtype=float)
                found = np.zeros((N,), dtype=bool)
                decision_token = np.full((N,), -1, dtype=int)
                margin_at_rt = np.full((N,), np.nan, dtype=float)
                p_true_at_rt = np.full((N,), np.nan, dtype=float)
                pmax_at_rt = np.full((N,), np.nan, dtype=float)
                pred_class_at_rt = np.full((N,), -1, dtype=int)
                for i in range(N):
                    s = int(effective_start[i]); e = int(end_excl[i])
                    m = logit_margin[i, s:e]
                    idx = _first_k_consecutive(m > float(thr), int(k))
                    if idx >= 0:
                        found[i] = True
                        decision_token[i] = s + idx
                        rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                        margin_at_rt[i] = float(m[idx])
                        p_true_at_rt[i] = float(p_corr[i, decision_token[i]])
                        pmax_at_rt[i] = float(p_max[i, decision_token[i]])
                        pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                cond_meta = {
                    "readout_mode": "simple_threshold_logit_margin",
                    "w": float("nan"),
                    "timeout_ms": float("nan"),
                    "cost_threshold": float("nan"),
                    "logit_margin_threshold": float(thr),
                    "p_threshold": float("nan"),
                    "k_consec": int(k),
                    **variant_meta(),
                }
                cond_row = {
                    **cond_meta,
                    **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                    **_decision_validity_stats(
                        decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                        p4_onset_ms=p4_onset_ms,
                        deviant_onset_ms=prepared.deviant_onset_ms,
                        found=found,
                    ),
                    "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                    "mean_logit_margin_at_rt": float(np.nanmean(margin_at_rt)) if np.any(np.isfinite(margin_at_rt)) else float("nan"),
                    "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                    "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                    "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                    "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                    "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                    "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                    "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                }
                cond_rows.append(cond_row)
                for i in range(N):
                    trial_rows.append({
                        **cond_meta,
                        "trial_index": int(i),
                        "position": int(pos[i]),
                        "model_rt_ms": float(rt_ms[i]),
                        "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                        "found_flag": bool(found[i]),
                        "decision_token": int(decision_token[i]),
                        "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                        "window_start_token": int(prepared.window_start[i]),
                        "window_end_token": int(prepared.window_end_exclusive[i]),
                        "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                        "p4_onset_ms": float(p4_onset_ms),
                        "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                        "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                        "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                        "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                        "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                        "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                        "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                        "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                        "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                        "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                        "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                        "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                        "p_correct_at_rt": float(p_true_at_rt[i]),
                        "p_max_at_rt": float(pmax_at_rt[i]),
                        "logit_margin_at_rt": float(margin_at_rt[i]),
                        "expected_cost_at_rt": float("nan"),
                        "pred_class": int(pred_class_at_rt[i]),
                        "true_class": int(prepared.y_cls[i]),
                        "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                        "found": bool(found[i]),
                        "forced_deadline": False,
                        "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                        "threshold_t": float(thr),
                        "cost_answer_now": float("nan"),
                        "bayes_error_cost": float("nan"),
                        "bayes_time_cost": float("nan"),
                        "bayes_threshold_start": float("nan"),
                        "bayes_threshold_min": float("nan"),
                        "bayes_urgency_slope": float("nan"),
                        "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                        "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                        "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                        "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                        "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                        "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                        "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                        "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                    })

    if include_msprt:
        for theta in msprt_threshold_list:
            rt_ms = np.full((N,), np.nan, dtype=float)
            found = np.zeros((N,), dtype=bool)
            forced_deadline = np.zeros((N,), dtype=bool)
            decision_token = np.full((N,), -1, dtype=int)
            margin_at_rt = np.full((N,), np.nan, dtype=float)
            p_true_at_rt = np.full((N,), np.nan, dtype=float)
            pmax_at_rt = np.full((N,), np.nan, dtype=float)
            pred_class_at_rt = np.full((N,), -1, dtype=int)
            for i in range(N):
                s = int(effective_start[i]); e = int(end_excl[i])
                m = logit_margin[i, s:e]
                if m.size == 0:
                    continue
                idx = _first_k_consecutive(m >= float(theta), 1)
                if idx < 0:
                    idx = m.size - 1
                    forced_deadline[i] = True
                found[i] = True
                decision_token[i] = s + idx
                rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                margin_at_rt[i] = float(m[idx])
                p_true_at_rt[i] = float(p_corr[i, decision_token[i]])
                pmax_at_rt[i] = float(p_max[i, decision_token[i]])
                pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
            cond_meta = {
                "readout_mode": "msprt_threshold",
                "w": float("nan"),
                "timeout_ms": float("nan"),
                "cost_threshold": float(theta),
                "msprt_threshold": float(theta),
                "p_threshold": float("nan"),
                "k_consec": 1,
                **variant_meta(),
            }
            cond_row = {
                **cond_meta,
                **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                **_decision_validity_stats(
                    decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                    p4_onset_ms=p4_onset_ms,
                    deviant_onset_ms=prepared.deviant_onset_ms,
                    found=found,
                ),
                "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                "forced_deadline_rate": float(np.mean(forced_deadline.astype(float))) if forced_deadline.size > 0 else float("nan"),
                "mean_logit_margin_at_rt": float(np.nanmean(margin_at_rt)) if np.any(np.isfinite(margin_at_rt)) else float("nan"),
                "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
            }
            cond_rows.append(cond_row)
            for i in range(N):
                trial_rows.append({
                    **cond_meta,
                    "trial_index": int(i),
                    "position": int(pos[i]),
                    "model_rt_ms": float(rt_ms[i]),
                    "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                    "found_flag": bool(found[i]),
                    "decision_token": int(decision_token[i]),
                    "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                    "window_start_token": int(prepared.window_start[i]),
                    "window_end_token": int(prepared.window_end_exclusive[i]),
                    "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                    "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                    "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                    "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                    "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                    "p4_onset_ms": float(p4_onset_ms),
                    "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                    "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                    "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                    "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                    "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                    "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                    "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                    "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                    "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                    "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                    "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                    "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                    "p_correct_at_rt": float(p_true_at_rt[i]),
                    "p_max_at_rt": float(pmax_at_rt[i]),
                    "msprt_logit_margin_at_rt": float(margin_at_rt[i]),
                    "expected_cost_at_rt": float("nan"),
                    "pred_class": int(pred_class_at_rt[i]),
                    "true_class": int(prepared.y_cls[i]),
                    "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                    "found": bool(found[i]),
                    "forced_deadline": bool(forced_deadline[i]),
                    "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                    "threshold_t": float(theta),
                    "cost_answer_now": float("nan"),
                    "bayes_error_cost": float("nan"),
                    "bayes_time_cost": float("nan"),
                    "bayes_threshold_start": float("nan"),
                    "bayes_threshold_min": float("nan"),
                    "bayes_urgency_slope": float("nan"),
                    "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                    "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                    "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                    "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                    "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                    "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                    "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                    "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                })

    if include_cost:
        for w, timeout_ms in w_specs:
            for thr in cost_threshold_list:
                for k in k_consec_list:
                    rt_ms = np.full((N,), np.nan, dtype=float)
                    found = np.zeros((N,), dtype=bool)
                    timeout = np.zeros((N,), dtype=bool)
                    decision_token = np.full((N,), -1, dtype=int)
                    p_at_rt = np.full((N,), np.nan, dtype=float)
                    pmax_at_rt = np.full((N,), np.nan, dtype=float)
                    cost_at_rt = np.full((N,), np.nan, dtype=float)
                    pred_class_at_rt = np.full((N,), -1, dtype=int)
                    for i in range(N):
                        s = int(effective_start[i]); e = int(end_excl[i])
                        wpc = p_corr[i, s:e]
                        token_idx = np.arange(s, e, dtype=float)
                        t_ms = elapsed_for_cost(token_idx, i)
                        # Expected cost combines unit error cost with time cost.
                        # Waiting until timeout costs one error unit when w=1/timeout_ms.
                        exp_cost = (1.0 - wpc) + float(w) * t_ms
                        idx = _first_k_consecutive(exp_cost < float(thr), int(k))
                        if idx >= 0:
                            found[i] = True
                            decision_token[i] = s + idx
                            rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                            p_at_rt[i] = float(wpc[idx])
                            pmax_at_rt[i] = float(p_max[i, decision_token[i]])
                            cost_at_rt[i] = float(exp_cost[idx])
                            pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                        else:
                            timeout[i] = True
                            rt_ms[i] = float(timeout_ms)
                    cond_meta = {
                        "readout_mode": "expected_cost_threshold" if rt_readout_mode == "expected_cost_threshold" else "bayesian_cost",
                        "readout_type": "bayesian_expected_cost_threshold_pcorrect",
                        "w": float(w),
                        "timeout_ms": float(timeout_ms),
                        "cost_threshold": float(thr),
                        "kappa": float(thr),
                        "p_threshold": float("nan"),
                        "k_consec": int(k),
                        "effective_threshold_formula": "EC(t)=1-p_correct(t)+w*t; respond when EC(t)<kappa",
                        **variant_meta(),
                    }
                    cond_row = {
                        **cond_meta,
                        **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                        **_decision_validity_stats(
                            decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                            p4_onset_ms=p4_onset_ms,
                            deviant_onset_ms=prepared.deviant_onset_ms,
                            found=found,
                        ),
                        "accuracy_at_rt": float(np.mean((pred_class_at_rt == prepared.y_cls).astype(float))) if pred_class_at_rt.size > 0 else float("nan"),
                        "accuracy_at_valid_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                        "timeout_rate": float(np.mean(timeout.astype(float))) if timeout.size > 0 else float("nan"),
                        "n_timeout_trials": int(np.sum(timeout)),
                        "mean_EC_at_response": float(np.nanmean(cost_at_rt)) if np.any(np.isfinite(cost_at_rt)) else float("nan"),
                        "mean_pcorrect_at_response": float(np.nanmean(p_at_rt)) if np.any(np.isfinite(p_at_rt)) else float("nan"),
                        "mean_pmax_at_response": float(np.nanmean(pmax_at_rt)) if np.any(np.isfinite(pmax_at_rt)) else float("nan"),
                        "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                        "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                        "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                        "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                        "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                        "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                        "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                    }
                    cond_rows.append(cond_row)
                    for i in range(N):
                        trial_rows.append({
                            **cond_meta,
                            "trial_index": int(i),
                            "position": int(pos[i]),
                            "model_rt_ms": float(rt_ms[i]),
                            "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                            "found_flag": bool(found[i]),
                            "decision_token": int(decision_token[i]),
                            "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                            "window_start_token": int(prepared.window_start[i]),
                            "window_end_token": int(prepared.window_end_exclusive[i]),
                            "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                            "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                            "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                            "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                            "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                            "p4_onset_ms": float(p4_onset_ms),
                            "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                            "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                            "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                            "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                            "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                            "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                            "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                            "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                            "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                            "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                            "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                            "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                            "p_correct_at_rt": float(p_at_rt[i]),
                            "p_max_at_rt": float(pmax_at_rt[i]) if np.isfinite(pmax_at_rt[i]) else float("nan"),
                            "expected_cost_at_rt": float(cost_at_rt[i]),
                            "pred_class": int(pred_class_at_rt[i]),
                            "true_class": int(prepared.y_cls[i]),
                            "correct_at_rt": bool(found[i] and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                            "found": bool(found[i]),
                            "timeout_flag": bool(timeout[i]),
                            "forced_deadline": bool(timeout[i]),
                            "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                            "threshold_t": float(thr),
                            "cost_answer_now": float(cost_at_rt[i]),
                            "bayes_error_cost": 1.0,
                            "bayes_time_cost": float(w),
                            "bayes_threshold_start": float("nan"),
                            "bayes_threshold_min": float("nan"),
                            "bayes_urgency_slope": float("nan"),
                            "readout_type": "bayesian_expected_cost_threshold_pcorrect",
                            "kappa": float(thr),
                            "effective_threshold_formula": "EC(t)=1-p_correct(t)+w*t; respond when EC(t)<kappa",
                            "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                            "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                            "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                            "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                            "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                            "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                            "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                            "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                        })
    if include_argmin:
        for w, timeout_ms in w_specs:
            rt_ms = np.full((N,), np.nan, dtype=float)
            found = np.zeros((N,), dtype=bool)
            decision_token = np.full((N,), -1, dtype=int)
            p_at_rt = np.full((N,), np.nan, dtype=float)
            cost_at_rt = np.full((N,), np.nan, dtype=float)
            argmin_at_window_start = np.zeros((N,), dtype=bool)
            argmin_at_window_end = np.zeros((N,), dtype=bool)
            n_window_tokens = np.zeros((N,), dtype=int)
            for i in range(N):
                s = int(effective_start[i]); e = int(end_excl[i])
                wpc = p_corr[i, s:e]
                n_window_tokens[i] = int(wpc.size)
                if wpc.size == 0:
                    continue
                token_idx = np.arange(s, e, dtype=float)
                t_ms = elapsed_for_cost(token_idx, i)
                exp_cost = (1.0 - wpc) + wpc * float(w) * t_ms
                finite = np.isfinite(exp_cost)
                if not np.any(finite):
                    continue
                masked = np.where(finite, exp_cost, np.inf)
                idx = int(np.argmin(masked))
                found[i] = True
                decision_token[i] = s + idx
                rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                p_at_rt[i] = float(wpc[idx])
                cost_at_rt[i] = float(exp_cost[idx])
                argmin_at_window_start[i] = bool(idx == 0)
                argmin_at_window_end[i] = bool(idx == wpc.size - 1)
            cond_meta = {
                "readout_mode": "bayes_cost_argmin",
                "w": float(w),
                "timeout_ms": float(timeout_ms),
                "cost_threshold": float("nan"),
                "p_threshold": float("nan"),
                "k_consec": 1,
                **variant_meta(),
            }
            cond_row = {
                **cond_meta,
                **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                **_decision_validity_stats(
                    decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                    p4_onset_ms=p4_onset_ms,
                    deviant_onset_ms=prepared.deviant_onset_ms,
                    found=found,
                ),
                "argmin_at_window_start_rate": float(np.mean(argmin_at_window_start[found].astype(float))) if np.any(found) else float("nan"),
                "argmin_at_window_end_rate": float(np.mean(argmin_at_window_end[found].astype(float))) if np.any(found) else float("nan"),
                "n_unique_model_rt": int(np.unique(rt_ms[found & np.isfinite(rt_ms)]).size),
                "model_rt_std": float(np.std(rt_ms[found & np.isfinite(rt_ms)])) if np.any(found & np.isfinite(rt_ms)) else float("nan"),
                "mean_min_expected_cost": float(np.nanmean(cost_at_rt)) if np.any(np.isfinite(cost_at_rt)) else float("nan"),
                "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
            }
            cond_rows.append(cond_row)
            for i in range(N):
                trial_rows.append({
                    **cond_meta,
                    "trial_index": int(i),
                    "position": int(pos[i]),
                    "model_rt_ms": float(rt_ms[i]),
                    "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                    "found_flag": bool(found[i]),
                    "decision_token": int(decision_token[i]),
                    "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                    "window_start_token": int(prepared.window_start[i]),
                    "window_end_token": int(prepared.window_end_exclusive[i]),
                    "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                    "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                    "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                    "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                    "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                    "p4_onset_ms": float(p4_onset_ms),
                    "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                    "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                    "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                    "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                    "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                    "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                    "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                    "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                    "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                    "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                    "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                    "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                    "p_correct_at_rt": float(p_at_rt[i]),
                    "p_max_at_rt": float(prepared.probs_max[i, decision_token[i]]) if decision_token[i] >= 0 else float("nan"),
                    "expected_cost_at_rt": float(cost_at_rt[i]),
                    "min_expected_cost": float(cost_at_rt[i]),
                    "elapsed_ms_at_rt": float((decision_token[i] - start[i]) * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                    "argmin_at_window_start": bool(argmin_at_window_start[i]),
                    "argmin_at_window_end": bool(argmin_at_window_end[i]),
                    "argmin_token_index": int(decision_token[i]),
                    "n_window_tokens": int(n_window_tokens[i]),
                    "pred_class": int(prepared.pred_class[i, decision_token[i]]) if decision_token[i] >= 0 else -1,
                    "true_class": int(prepared.y_cls[i]),
                    "correct_at_rt": bool(decision_token[i] >= 0 and int(prepared.pred_class[i, decision_token[i]]) == int(prepared.y_cls[i])),
                    "found": bool(found[i]),
                    "forced_deadline": False,
                    "elapsed_ms": float((decision_token[i] - start[i]) * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                    "threshold_t": float("nan"),
                    "cost_answer_now": float(cost_at_rt[i]),
                    "bayes_error_cost": float("nan"),
                    "bayes_time_cost": float("nan"),
                    "bayes_threshold_start": float("nan"),
                    "bayes_threshold_min": float("nan"),
                    "bayes_urgency_slope": float("nan"),
                    "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                    "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                    "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                    "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                    "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                    "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                    "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                    "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                })
    if include_deadline:
        bayes_k_values = tuple(int(k) for k in (bayes_k_consec_list or k_consec_list))
        for time_cost in bayes_time_cost_list:
            for threshold_start in bayes_threshold_start_list:
                for threshold_min in bayes_threshold_min_list:
                    for urgency_slope in bayes_urgency_slope_list:
                        for k in bayes_k_values:
                            rt_ms = np.full((N,), np.nan, dtype=float)
                            found = np.zeros((N,), dtype=bool)
                            forced_deadline = np.zeros((N,), dtype=bool)
                            decision_token = np.full((N,), -1, dtype=int)
                            p_true_at_rt = np.full((N,), np.nan, dtype=float)
                            p_best_at_rt = np.full((N,), np.nan, dtype=float)
                            threshold_at_rt = np.full((N,), np.nan, dtype=float)
                            cost_at_rt = np.full((N,), np.nan, dtype=float)
                            pred_class_at_rt = np.full((N,), -1, dtype=int)
                            for i in range(N):
                                s = int(effective_start[i]); e = int(end_excl[i])
                                if e <= s:
                                    continue
                                window_p_best = p_max[i, s:e]
                                window_p_true = p_corr[i, s:e]
                                decision_time_ms = np.arange(s, e, dtype=float) * float(prepared.token_ms)
                                elapsed_ms = decision_time_ms - float(prepared.rt_reference_time_ms[i])
                                threshold_curve = np.maximum(
                                    float(threshold_min),
                                    float(threshold_start) - float(urgency_slope) * elapsed_ms,
                                )
                                hits = window_p_best >= threshold_curve
                                idx = _first_k_consecutive(hits, int(k))
                                if idx < 0 and bool(bayes_force_deadline):
                                    idx = len(window_p_best) - 1
                                    forced_deadline[i] = True
                                if idx >= 0:
                                    found[i] = True
                                    decision_token[i] = s + idx
                                    rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                                    p_true_at_rt[i] = float(window_p_true[idx])
                                    p_best_at_rt[i] = float(window_p_best[idx])
                                    threshold_at_rt[i] = float(threshold_curve[idx])
                                    cost_at_rt[i] = float((1.0 - window_p_best[idx]) * float(bayes_error_cost) + elapsed_ms[idx] * float(time_cost))
                                    pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                            cond_meta = {
                                "readout_mode": "bayesian_deadline",
                                "w": float("nan"),
                                "timeout_ms": float("nan"),
                                "cost_threshold": float("nan"),
                                "p_threshold": float("nan"),
                                "k_consec": int(k),
                                "bayes_error_cost": float(bayes_error_cost),
                                "bayes_time_cost": float(time_cost),
                                "bayes_threshold_start": float(threshold_start),
                                "bayes_threshold_min": float(threshold_min),
                                "bayes_urgency_slope": float(urgency_slope),
                                **variant_meta(),
                            }
                            cond_row = {
                                **cond_meta,
                                **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                                **_decision_validity_stats(
                                    decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                                    p4_onset_ms=p4_onset_ms,
                                    deviant_onset_ms=prepared.deviant_onset_ms,
                                    found=found,
                                ),
                                "forced_deadline_rate": float(np.mean(forced_deadline.astype(float))) if forced_deadline.size > 0 else float("nan"),
                                "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                                "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                                "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                                "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                                "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                                "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                                "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                                "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                            }
                            cond_rows.append(cond_row)
                            for i in range(N):
                                trial_rows.append({
                                    **cond_meta,
                                    "trial_index": int(i),
                                    "position": int(pos[i]),
                                    "model_rt_ms": float(rt_ms[i]),
                                    "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                                    "found_flag": bool(found[i]),
                                    "found": bool(found[i]),
                                    "forced_deadline": bool(forced_deadline[i]),
                                    "decision_token": int(decision_token[i]),
                                    "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                                    "window_start_token": int(prepared.window_start[i]),
                                    "window_end_token": int(prepared.window_end_exclusive[i]),
                                    "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                                    "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                                    "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                                    "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                                    "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                                    "p4_onset_ms": float(p4_onset_ms),
                                    "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                                    "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                                    "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                                    "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                                    "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                                    "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                                    "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                                    "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                                    "elapsed_ms": float(rt_ms[i]) if np.isfinite(rt_ms[i]) else float("nan"),
                                    "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                                    "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                                    "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                                    "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                                    "pred_class": int(pred_class_at_rt[i]),
                                    "true_class": int(prepared.y_cls[i]),
                                    "correct_at_rt": bool(found[i] and pred_class_at_rt[i] == prepared.y_cls[i]),
                                    "p_true": float(p_true_at_rt[i]),
                                    "p_best": float(p_best_at_rt[i]),
                                    "p_correct_at_rt": float(p_true_at_rt[i]),
                                    "p_max_at_rt": float(p_best_at_rt[i]),
                                    "threshold_t": float(threshold_at_rt[i]),
                                    "cost_answer_now": float(cost_at_rt[i]),
                                    "expected_cost_at_rt": float(cost_at_rt[i]),
                                    "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                                    "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                                    "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                                    "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                                    "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                                    "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                                    "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                                    "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                                })
    if include_accumulator:
        bayes_k_values = tuple(int(k) for k in (bayes_k_consec_list or k_consec_list))
        eps = 1e-6
        baseline_log_odds = math.log((1.0 / 3.0) / (1.0 - 1.0 / 3.0))
        for evidence_bound in bayes_evidence_bound_list:
            for leak in bayes_leak_list:
                for k in bayes_k_values:
                    rt_ms = np.full((N,), np.nan, dtype=float)
                    found = np.zeros((N,), dtype=bool)
                    forced_deadline = np.zeros((N,), dtype=bool)
                    decision_token = np.full((N,), -1, dtype=int)
                    p_true_at_rt = np.full((N,), np.nan, dtype=float)
                    p_best_at_rt = np.full((N,), np.nan, dtype=float)
                    evidence_at_rt = np.full((N,), np.nan, dtype=float)
                    pred_class_at_rt = np.full((N,), -1, dtype=int)
                    for i in range(N):
                        s = int(effective_start[i]); e = int(end_excl[i])
                        if e <= s:
                            continue
                        window_p_best = np.clip(p_max[i, s:e].astype(float), eps, 1.0 - eps)
                        window_p_true = p_corr[i, s:e]
                        log_odds = np.log(window_p_best / (1.0 - window_p_best))
                        increments = np.maximum(0.0, log_odds - baseline_log_odds)
                        acc = np.zeros_like(increments, dtype=float)
                        running = 0.0
                        for j, inc in enumerate(increments):
                            running = max(0.0, running * (1.0 - float(leak)) + float(inc))
                            acc[j] = running
                        hits = acc >= float(evidence_bound)
                        idx = _first_k_consecutive(hits, int(k))
                        if idx < 0 and bool(bayes_force_deadline):
                            idx = len(window_p_best) - 1
                            forced_deadline[i] = True
                        if idx >= 0:
                            found[i] = True
                            decision_token[i] = s + idx
                            rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                            p_true_at_rt[i] = float(window_p_true[idx])
                            p_best_at_rt[i] = float(window_p_best[idx])
                            evidence_at_rt[i] = float(acc[idx])
                            pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                    cond_meta = {
                        "readout_mode": "bayesian_accumulator",
                        "w": float("nan"),
                        "timeout_ms": float("nan"),
                        "cost_threshold": float("nan"),
                        "p_threshold": float("nan"),
                        "k_consec": int(k),
                        "bayes_error_cost": float("nan"),
                        "bayes_time_cost": float("nan"),
                        "bayes_threshold_start": float("nan"),
                        "bayes_threshold_min": float("nan"),
                        "bayes_urgency_slope": float("nan"),
                        "bayes_evidence_bound": float(evidence_bound),
                        "bayes_leak": float(leak),
                        **variant_meta(),
                    }
                    cond_row = {
                        **cond_meta,
                        **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                        **_decision_validity_stats(
                            decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                            p4_onset_ms=p4_onset_ms,
                            deviant_onset_ms=prepared.deviant_onset_ms,
                            found=found,
                        ),
                        "forced_deadline_rate": float(np.mean(forced_deadline.astype(float))) if forced_deadline.size > 0 else float("nan"),
                        "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                        "mean_evidence_at_rt": float(np.nanmean(evidence_at_rt)) if np.any(np.isfinite(evidence_at_rt)) else float("nan"),
                        "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                        "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                        "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                        "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                        "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                        "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                        "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                    }
                    cond_rows.append(cond_row)
                    for i in range(N):
                        trial_rows.append({
                            **cond_meta,
                            "trial_index": int(i),
                            "position": int(pos[i]),
                            "model_rt_ms": float(rt_ms[i]),
                            "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                            "found_flag": bool(found[i]),
                            "decision_token": int(decision_token[i]),
                            "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                            "window_start_token": int(prepared.window_start[i]),
                            "window_end_token": int(prepared.window_end_exclusive[i]),
                            "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                            "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                            "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                            "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                            "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                            "p4_onset_ms": float(p4_onset_ms),
                            "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                            "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                            "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                            "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                            "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                            "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                            "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                            "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                            "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                            "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                            "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                            "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                            "p_correct_at_rt": float(p_true_at_rt[i]),
                            "p_max_at_rt": float(p_best_at_rt[i]),
                            "expected_cost_at_rt": float("nan"),
                            "evidence_at_rt": float(evidence_at_rt[i]),
                            "pred_class": int(pred_class_at_rt[i]),
                            "true_class": int(prepared.y_cls[i]),
                            "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                            "found": bool(found[i]),
                            "forced_deadline": bool(forced_deadline[i]),
                            "elapsed_ms": float((decision_token[i] - start[i]) * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                            "threshold_t": float(evidence_bound),
                            "cost_answer_now": float("nan"),
                            "bayes_error_cost": float("nan"),
                            "bayes_time_cost": float("nan"),
                            "bayes_threshold_start": float("nan"),
                            "bayes_threshold_min": float("nan"),
                            "bayes_urgency_slope": float("nan"),
                            "bayes_evidence_bound": float(evidence_bound),
                            "bayes_leak": float(leak),
                            "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                            "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                            "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                            "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                            "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                            "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                            "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                            "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                        })
    if include_pcorrect_accumulator:
        bayes_k_values = tuple(int(k) for k in (bayes_k_consec_list or k_consec_list))
        for evidence_bound in bayes_evidence_bound_list:
            for leak in bayes_leak_list:
                for k in bayes_k_values:
                    rt_ms = np.full((N,), np.nan, dtype=float)
                    found = np.zeros((N,), dtype=bool)
                    forced_deadline = np.zeros((N,), dtype=bool)
                    decision_token = np.full((N,), -1, dtype=int)
                    p_true_at_rt = np.full((N,), np.nan, dtype=float)
                    p_best_at_rt = np.full((N,), np.nan, dtype=float)
                    evidence_at_rt = np.full((N,), np.nan, dtype=float)
                    pred_class_at_rt = np.full((N,), -1, dtype=int)
                    for i in range(N):
                        s = int(effective_start[i]); e = int(end_excl[i])
                        if e <= s:
                            continue
                        window_p_true = np.clip(p_corr[i, s:e].astype(float), 0.0, 1.0)
                        acc = np.zeros_like(window_p_true, dtype=float)
                        running = 0.0
                        for j, inc in enumerate(window_p_true):
                            running = max(0.0, running * (1.0 - float(leak)) + float(inc))
                            acc[j] = running
                        hits = acc >= float(evidence_bound)
                        idx = _first_k_consecutive(hits, int(k))
                        if idx < 0 and bool(bayes_force_deadline):
                            idx = len(window_p_true) - 1
                            forced_deadline[i] = True
                        if idx >= 0:
                            found[i] = True
                            decision_token[i] = s + idx
                            rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                            p_true_at_rt[i] = float(p_corr[i, decision_token[i]])
                            p_best_at_rt[i] = float(p_max[i, decision_token[i]])
                            evidence_at_rt[i] = float(acc[idx])
                            pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                    cond_meta = {
                        "readout_mode": "p_correct_accumulator",
                        "w": float("nan"),
                        "timeout_ms": float("nan"),
                        "cost_threshold": float("nan"),
                        "p_threshold": float("nan"),
                        "k_consec": int(k),
                        "bayes_error_cost": float("nan"),
                        "bayes_time_cost": float("nan"),
                        "bayes_threshold_start": float("nan"),
                        "bayes_threshold_min": float("nan"),
                        "bayes_urgency_slope": float("nan"),
                        "bayes_evidence_bound": float(evidence_bound),
                        "bayes_leak": float(leak),
                        **variant_meta(),
                    }
                    cond_rows.append({
                        **cond_meta,
                        **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                        **_decision_validity_stats(
                            decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                            p4_onset_ms=p4_onset_ms,
                            deviant_onset_ms=prepared.deviant_onset_ms,
                            found=found,
                        ),
                        "forced_deadline_rate": float(np.mean(forced_deadline.astype(float))) if forced_deadline.size > 0 else float("nan"),
                        "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                        "mean_evidence_at_rt": float(np.nanmean(evidence_at_rt)) if np.any(np.isfinite(evidence_at_rt)) else float("nan"),
                        "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                        "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                        "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                        "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                        "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                        "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                        "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                    })
                    for i in range(N):
                        trial_rows.append({
                            **cond_meta,
                            "trial_index": int(i),
                            "position": int(pos[i]),
                            "model_rt_ms": float(rt_ms[i]),
                            "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                            "found_flag": bool(found[i]),
                            "decision_token": int(decision_token[i]),
                            "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                            "window_start_token": int(prepared.window_start[i]),
                            "window_end_token": int(prepared.window_end_exclusive[i]),
                            "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                            "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                            "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                            "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                            "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                            "p4_onset_ms": float(p4_onset_ms),
                            "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                            "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                            "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                            "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                            "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                            "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                            "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                            "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                            "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                            "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                            "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                            "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                            "p_correct_at_rt": float(p_true_at_rt[i]),
                            "p_max_at_rt": float(p_best_at_rt[i]),
                            "expected_cost_at_rt": float("nan"),
                            "evidence_at_rt": float(evidence_at_rt[i]),
                            "pred_class": int(pred_class_at_rt[i]),
                            "true_class": int(prepared.y_cls[i]),
                            "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                            "found": bool(found[i]),
                            "forced_deadline": bool(forced_deadline[i]),
                            "elapsed_ms": float((decision_token[i] - start[i]) * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                            "threshold_t": float(evidence_bound),
                            "cost_answer_now": float("nan"),
                            "bayes_error_cost": float("nan"),
                            "bayes_time_cost": float("nan"),
                            "bayes_threshold_start": float("nan"),
                            "bayes_threshold_min": float("nan"),
                            "bayes_urgency_slope": float("nan"),
                            "bayes_evidence_bound": float(evidence_bound),
                            "bayes_leak": float(leak),
                            "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                            "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                            "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                            "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                            "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                            "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                            "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                            "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                        })
    if include_online_cost:
        bayes_k_values = tuple(int(k) for k in (bayes_k_consec_list or k_consec_list))
        for time_cost in bayes_time_cost_list:
            for wait_lag_ms in bayes_wait_lag_ms_list:
                lag_tokens = int(max(1, math.ceil(float(wait_lag_ms) / float(prepared.token_ms))))
                lag_ms = float(lag_tokens) * float(prepared.token_ms)
                for min_p in bayes_min_p_list:
                    for k in bayes_k_values:
                        rt_ms = np.full((N,), np.nan, dtype=float)
                        found = np.zeros((N,), dtype=bool)
                        forced_deadline = np.zeros((N,), dtype=bool)
                        decision_token = np.full((N,), -1, dtype=int)
                        p_true_at_rt = np.full((N,), np.nan, dtype=float)
                        p_best_at_rt = np.full((N,), np.nan, dtype=float)
                        cost_at_rt = np.full((N,), np.nan, dtype=float)
                        cost_delta_at_rt = np.full((N,), np.nan, dtype=float)
                        pred_class_at_rt = np.full((N,), -1, dtype=int)
                        for i in range(N):
                            s = int(effective_start[i]); e = int(end_excl[i])
                            if e <= s:
                                continue
                            window_p_best = p_max[i, s:e].astype(float)
                            window_p_true = p_corr[i, s:e].astype(float)
                            token_idx = np.arange(s, e, dtype=float)
                            elapsed_ms = elapsed_for_cost(token_idx, i)
                            exp_cost = (1.0 - window_p_best) * float(bayes_error_cost) + window_p_best * float(time_cost) * elapsed_ms
                            hits = np.zeros_like(window_p_best, dtype=bool)
                            cost_delta = np.full_like(window_p_best, np.nan, dtype=float)
                            if len(window_p_best) > lag_tokens:
                                cost_delta[lag_tokens:] = exp_cost[:-lag_tokens] - exp_cost[lag_tokens:]
                                hits[lag_tokens:] = (
                                    np.isfinite(exp_cost[lag_tokens:])
                                    & np.isfinite(cost_delta[lag_tokens:])
                                    & (window_p_best[lag_tokens:] >= float(min_p))
                                    & (cost_delta[lag_tokens:] <= 0.0)
                                )
                            idx = _first_k_consecutive(hits, int(k))
                            if idx < 0 and bool(bayes_force_deadline):
                                finite = np.isfinite(exp_cost)
                                if np.any(finite):
                                    idx = len(window_p_best) - 1
                                    forced_deadline[i] = True
                            if idx >= 0:
                                found[i] = True
                                decision_token[i] = s + idx
                                rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                                p_true_at_rt[i] = float(window_p_true[idx])
                                p_best_at_rt[i] = float(window_p_best[idx])
                                cost_at_rt[i] = float(exp_cost[idx])
                                cost_delta_at_rt[i] = float(cost_delta[idx])
                                pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                        cond_meta = {
                            "readout_mode": "bayesian_online_cost",
                            "w": float("nan"),
                            "timeout_ms": float(1.0 / float(time_cost)) if float(time_cost) > 0 else float("nan"),
                            "cost_threshold": float("nan"),
                            "p_threshold": float(min_p),
                            "k_consec": int(k),
                            "bayes_error_cost": float(bayes_error_cost),
                            "bayes_time_cost": float(time_cost),
                            "bayes_threshold_start": float("nan"),
                            "bayes_threshold_min": float("nan"),
                            "bayes_urgency_slope": float("nan"),
                            "bayes_evidence_bound": float("nan"),
                            "bayes_leak": float("nan"),
                            "bayes_wait_lag_ms": float(lag_ms),
                            "bayes_min_p": float(min_p),
                            **variant_meta(),
                        }
                        cond_row = {
                            **cond_meta,
                            **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                            **_decision_validity_stats(
                                decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                                p4_onset_ms=p4_onset_ms,
                                deviant_onset_ms=prepared.deviant_onset_ms,
                                found=found,
                            ),
                            "forced_deadline_rate": float(np.mean(forced_deadline.astype(float))) if forced_deadline.size > 0 else float("nan"),
                            "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                            "mean_p_best_at_rt": float(np.nanmean(p_best_at_rt)) if np.any(np.isfinite(p_best_at_rt)) else float("nan"),
                            "mean_expected_cost_at_rt": float(np.nanmean(cost_at_rt)) if np.any(np.isfinite(cost_at_rt)) else float("nan"),
                            "mean_online_cost_delta_at_rt": float(np.nanmean(cost_delta_at_rt)) if np.any(np.isfinite(cost_delta_at_rt)) else float("nan"),
                            "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                            "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                            "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                            "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                            "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                            "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                            "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                        }
                        cond_rows.append(cond_row)
                        for i in range(N):
                            trial_rows.append({
                                **cond_meta,
                                "trial_index": int(i),
                                "position": int(pos[i]),
                                "model_rt_ms": float(rt_ms[i]),
                                "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                                "found_flag": bool(found[i]),
                                "decision_token": int(decision_token[i]),
                                "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                                "window_start_token": int(prepared.window_start[i]),
                                "window_end_token": int(prepared.window_end_exclusive[i]),
                                "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                                "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                                "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                                "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                                "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                                "p4_onset_ms": float(p4_onset_ms),
                                "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                                "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                                "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                                "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                                "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                                "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                                "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                                "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                                "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                                "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                                "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                                "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                                "p_correct_at_rt": float(p_true_at_rt[i]),
                                "p_max_at_rt": float(p_best_at_rt[i]),
                                "expected_cost_at_rt": float(cost_at_rt[i]),
                                "online_cost_delta_at_rt": float(cost_delta_at_rt[i]),
                                "pred_class": int(pred_class_at_rt[i]),
                                "true_class": int(prepared.y_cls[i]),
                                "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                                "found": bool(found[i]),
                                "forced_deadline": bool(forced_deadline[i]),
                                "elapsed_ms": float((decision_token[i] - start[i]) * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                                "threshold_t": 0.0,
                                "cost_answer_now": float(cost_at_rt[i]),
                                "bayes_error_cost": float(bayes_error_cost),
                                "bayes_time_cost": float(time_cost),
                                "bayes_wait_lag_ms": float(lag_ms),
                                "bayes_min_p": float(min_p),
                                "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                                "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                                "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                                "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                                "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                                "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                                "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                                "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                            })
    if include_stochastic_online:
        for time_cost in bayes_time_cost_list:
            for beta in stochastic_beta_list:
                for b0 in stochastic_b0_list:
                    for seed in stochastic_seed_list:
                        rt_ms = np.full((N,), np.nan, dtype=float)
                        found = np.zeros((N,), dtype=bool)
                        decision_token = np.full((N,), -1, dtype=int)
                        p_true_at_rt = np.full((N,), np.nan, dtype=float)
                        p_best_at_rt = np.full((N,), np.nan, dtype=float)
                        commit_prob_at_rt = np.full((N,), np.nan, dtype=float)
                        expected_cost_at_rt = np.full((N,), np.nan, dtype=float)
                        pred_class_at_rt = np.full((N,), -1, dtype=int)
                        for i in range(N):
                            s = int(effective_start[i]); e = int(end_excl[i])
                            if e <= s:
                                continue
                            window_p_best = p_max[i, s:e].astype(float)
                            window_p_true = p_corr[i, s:e].astype(float)
                            token_idx = np.arange(s, e, dtype=float)
                            elapsed_ms = elapsed_for_cost(token_idx, i)
                            expected_cost = (1.0 - window_p_best) * float(bayes_error_cost) + float(time_cost) * elapsed_ms
                            logits_commit = float(beta) * (window_p_best + float(time_cost) * elapsed_ms - float(b0))
                            logits_commit = np.clip(logits_commit, -60.0, 60.0)
                            commit_prob = 1.0 / (1.0 + np.exp(-logits_commit))
                            rng = np.random.default_rng(int(seed) + (i * 100003))
                            draws = rng.random(commit_prob.shape[0])
                            hits = draws < commit_prob
                            idx = int(np.flatnonzero(hits)[0]) if np.any(hits) else -1
                            if idx >= 0:
                                found[i] = True
                                decision_token[i] = s + idx
                                rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                                p_true_at_rt[i] = float(window_p_true[idx])
                                p_best_at_rt[i] = float(window_p_best[idx])
                                commit_prob_at_rt[i] = float(commit_prob[idx])
                                expected_cost_at_rt[i] = float(expected_cost[idx])
                                pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                        cond_meta = {
                            "readout_mode": "stochastic_online_commit",
                            "w": float("nan"),
                            "timeout_ms": float(1.0 / float(time_cost)) if float(time_cost) > 0 else float("nan"),
                            "cost_threshold": float("nan"),
                            "p_threshold": float("nan"),
                            "k_consec": 1,
                            "bayes_error_cost": float(bayes_error_cost),
                            "bayes_time_cost": float(time_cost),
                            "stochastic_beta": float(beta),
                            "stochastic_b0": float(b0),
                            "stochastic_seed": int(seed),
                            **variant_meta(),
                        }
                        cond_row = {
                            **cond_meta,
                            **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                            **_decision_validity_stats(
                                decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                                p4_onset_ms=p4_onset_ms,
                                deviant_onset_ms=prepared.deviant_onset_ms,
                                found=found,
                            ),
                            "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                            "mean_commit_prob_at_rt": float(np.nanmean(commit_prob_at_rt)) if np.any(np.isfinite(commit_prob_at_rt)) else float("nan"),
                            "mean_expected_cost_at_rt": float(np.nanmean(expected_cost_at_rt)) if np.any(np.isfinite(expected_cost_at_rt)) else float("nan"),
                            "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                            "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                            "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                            "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                            "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                            "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                            "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                        }
                        cond_rows.append(cond_row)
                        for i in range(N):
                            trial_rows.append({
                                **cond_meta,
                                "trial_index": int(i),
                                "position": int(pos[i]),
                                "model_rt_ms": float(rt_ms[i]),
                                "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                                "found_flag": bool(found[i]),
                                "decision_token": int(decision_token[i]),
                                "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                                "window_start_token": int(prepared.window_start[i]),
                                "window_end_token": int(prepared.window_end_exclusive[i]),
                                "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                                "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                                "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                                "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                                "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                                "p4_onset_ms": float(p4_onset_ms),
                                "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                                "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                                "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                                "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                                "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                                "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                                "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                                "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                                "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                                "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                                "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                                "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                                "p_correct_at_rt": float(p_true_at_rt[i]),
                                "p_max_at_rt": float(p_best_at_rt[i]),
                                "commit_prob_at_rt": float(commit_prob_at_rt[i]),
                                "expected_cost_at_rt": float(expected_cost_at_rt[i]),
                                "pred_class": int(pred_class_at_rt[i]),
                                "true_class": int(prepared.y_cls[i]),
                                "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                                "found": bool(found[i]),
                                "forced_deadline": False,
                                "elapsed_ms": float((decision_token[i] - start[i]) * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                                "threshold_t": float("nan"),
                                "cost_answer_now": float(expected_cost_at_rt[i]),
                                "bayes_error_cost": float(bayes_error_cost),
                                "bayes_time_cost": float(time_cost),
                                "stochastic_beta": float(beta),
                                "stochastic_b0": float(b0),
                                "stochastic_seed": int(seed),
                                "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                                "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                                "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                                "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                                "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                                "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                                "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                                "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                            })
    if include_advisor_stochastic:
        for time_cost in bayes_time_cost_list:
            for beta in stochastic_beta_list:
                for b0 in stochastic_b0_list:
                    for seed in stochastic_seed_list:
                        for k in bayes_k_consec_list or k_consec_list:
                            rt_ms = np.full((N,), np.nan, dtype=float)
                            found = np.zeros((N,), dtype=bool)
                            forced_deadline = np.zeros((N,), dtype=bool)
                            decision_token = np.full((N,), -1, dtype=int)
                            p_true_at_rt = np.full((N,), np.nan, dtype=float)
                            p_best_at_rt = np.full((N,), np.nan, dtype=float)
                            commit_prob_at_rt = np.full((N,), np.nan, dtype=float)
                            expected_cost_at_rt = np.full((N,), np.nan, dtype=float)
                            cost_advantage_at_rt = np.full((N,), np.nan, dtype=float)
                            pred_class_at_rt = np.full((N,), -1, dtype=int)
                            for i in range(N):
                                s = int(effective_start[i]); e = int(end_excl[i])
                                if e <= s:
                                    continue
                                window_p_best = p_max[i, s:e].astype(float)
                                window_p_true = p_corr[i, s:e].astype(float)
                                token_idx = np.arange(s, e, dtype=float)
                                elapsed_ms = elapsed_for_cost(token_idx, i)
                                expected_cost = (
                                    (1.0 - window_p_best) * float(bayes_error_cost)
                                    + window_p_best * float(time_cost) * elapsed_ms
                                )
                                cost_advantage = float(bayes_error_cost) - expected_cost
                                logits_commit = float(beta) * cost_advantage + float(b0)
                                logits_commit = np.clip(logits_commit, -60.0, 60.0)
                                commit_prob = 1.0 / (1.0 + np.exp(-logits_commit))
                                rng = np.random.default_rng(int(seed) + (i * 100003))
                                draws = rng.random(commit_prob.shape[0])
                                hits = draws < commit_prob
                                idx = _first_k_consecutive(hits, int(k))
                                if idx < 0 and bool(bayes_force_deadline):
                                    idx = len(window_p_best) - 1
                                    forced_deadline[i] = True
                                if idx >= 0:
                                    found[i] = True
                                    decision_token[i] = s + idx
                                    rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                                    p_true_at_rt[i] = float(window_p_true[idx])
                                    p_best_at_rt[i] = float(window_p_best[idx])
                                    commit_prob_at_rt[i] = float(commit_prob[idx])
                                    expected_cost_at_rt[i] = float(expected_cost[idx])
                                    cost_advantage_at_rt[i] = float(cost_advantage[idx])
                                    pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                            cond_meta = {
                                "readout_mode": "advisor_bayes_stochastic",
                                "w": float(time_cost),
                                "timeout_ms": float(1.0 / float(time_cost)) if float(time_cost) > 0 else float("nan"),
                                "cost_threshold": float("nan"),
                                "p_threshold": float("nan"),
                                "k_consec": int(k),
                                "bayes_error_cost": float(bayes_error_cost),
                                "bayes_time_cost": float(time_cost),
                                "stochastic_beta": float(beta),
                                "stochastic_b0": float(b0),
                                "stochastic_seed": int(seed),
                                **variant_meta(),
                            }
                            cond_row = {
                                **cond_meta,
                                **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                                **_decision_validity_stats(
                                    decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                                    p4_onset_ms=p4_onset_ms,
                                    deviant_onset_ms=prepared.deviant_onset_ms,
                                    found=found,
                                ),
                                "forced_deadline_rate": float(np.mean(forced_deadline.astype(float))) if forced_deadline.size > 0 else float("nan"),
                                "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                                "mean_commit_prob_at_rt": float(np.nanmean(commit_prob_at_rt)) if np.any(np.isfinite(commit_prob_at_rt)) else float("nan"),
                                "mean_expected_cost_at_rt": float(np.nanmean(expected_cost_at_rt)) if np.any(np.isfinite(expected_cost_at_rt)) else float("nan"),
                                "mean_cost_advantage_at_rt": float(np.nanmean(cost_advantage_at_rt)) if np.any(np.isfinite(cost_advantage_at_rt)) else float("nan"),
                                "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                                "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                                "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                                "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                                "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                                "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                                "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                            }
                            cond_rows.append(cond_row)
                            for i in range(N):
                                trial_rows.append({
                                    **cond_meta,
                                    "trial_index": int(i),
                                    "position": int(pos[i]),
                                    "model_rt_ms": float(rt_ms[i]),
                                    "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                                    "found_flag": bool(found[i]),
                                    "decision_token": int(decision_token[i]),
                                    "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                                    "window_start_token": int(prepared.window_start[i]),
                                    "window_end_token": int(prepared.window_end_exclusive[i]),
                                    "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                                    "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                                    "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                                    "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                                    "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                                    "p4_onset_ms": float(p4_onset_ms),
                                    "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                                    "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                                    "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                                    "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                                    "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                                    "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                                    "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                                    "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                                    "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                                    "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                                    "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                                    "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                                    "p_correct_at_rt": float(p_true_at_rt[i]),
                                    "p_max_at_rt": float(p_best_at_rt[i]),
                                    "commit_prob_at_rt": float(commit_prob_at_rt[i]),
                                    "expected_cost_at_rt": float(expected_cost_at_rt[i]),
                                    "online_cost_delta_at_rt": float(cost_advantage_at_rt[i]),
                                    "pred_class": int(pred_class_at_rt[i]),
                                    "true_class": int(prepared.y_cls[i]),
                                    "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                                    "found": bool(found[i]),
                                    "forced_deadline": bool(forced_deadline[i]),
                                    "elapsed_ms": float((decision_token[i] - start[i]) * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                                    "threshold_t": float("nan"),
                                    "cost_answer_now": float(expected_cost_at_rt[i]),
                                    "bayes_error_cost": float(bayes_error_cost),
                                    "bayes_time_cost": float(time_cost),
                                    "stochastic_beta": float(beta),
                                    "stochastic_b0": float(b0),
                                    "stochastic_seed": int(seed),
                                    "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                                    "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                                    "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                                    "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                                    "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                                    "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                                    "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                                    "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                                })
    if include_advisor_dp:
        for time_cost in bayes_time_cost_list:
            for k in bayes_k_consec_list or k_consec_list:
                rt_ms = np.full((N,), np.nan, dtype=float)
                found = np.zeros((N,), dtype=bool)
                forced_deadline = np.zeros((N,), dtype=bool)
                decision_token = np.full((N,), -1, dtype=int)
                p_true_at_rt = np.full((N,), np.nan, dtype=float)
                p_best_at_rt = np.full((N,), np.nan, dtype=float)
                expected_cost_at_rt = np.full((N,), np.nan, dtype=float)
                wait_cost_at_rt = np.full((N,), np.nan, dtype=float)
                cost_advantage_at_rt = np.full((N,), np.nan, dtype=float)
                pred_class_at_rt = np.full((N,), -1, dtype=int)
                for i in range(N):
                    s = int(effective_start[i]); e = int(end_excl[i])
                    if e <= s:
                        continue
                    window_p_best = p_max[i, s:e].astype(float)
                    window_p_true = p_corr[i, s:e].astype(float)
                    token_idx = np.arange(s, e, dtype=int)
                    elapsed_ms = elapsed_for_cost(token_idx.astype(float), i)
                    immediate_cost = ((1.0 - window_p_best) * float(bayes_error_cost)) + (float(time_cost) * elapsed_ms)
                    V = np.full_like(immediate_cost, np.nan, dtype=float)
                    choose_stop = np.zeros_like(immediate_cost, dtype=bool)
                    V[-1] = immediate_cost[-1]
                    choose_stop[-1] = True
                    for j in range(len(immediate_cost) - 2, -1, -1):
                        delta_ms = float(elapsed_ms[j + 1] - elapsed_ms[j])
                        wait_cost = (float(time_cost) * delta_ms) + V[j + 1]
                        if immediate_cost[j] <= wait_cost:
                            V[j] = immediate_cost[j]
                            choose_stop[j] = True
                        else:
                            V[j] = wait_cost
                    idx = _first_k_consecutive(choose_stop, int(k))
                    if idx < 0 and bool(bayes_force_deadline):
                        idx = len(window_p_best) - 1
                        forced_deadline[i] = True
                    if idx >= 0:
                        found[i] = True
                        decision_token[i] = s + idx
                        rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                        p_true_at_rt[i] = float(window_p_true[idx])
                        p_best_at_rt[i] = float(window_p_best[idx])
                        expected_cost_at_rt[i] = float(immediate_cost[idx])
                        if idx < len(immediate_cost) - 1:
                            delta_ms = float(elapsed_ms[idx + 1] - elapsed_ms[idx])
                            wait_cost_at_rt[i] = float((float(time_cost) * delta_ms) + V[idx + 1])
                        cost_advantage_at_rt[i] = float(wait_cost_at_rt[i] - immediate_cost[idx]) if np.isfinite(wait_cost_at_rt[i]) else float("nan")
                        pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                cond_meta = {
                    "readout_mode": "advisor_expected_cost_dp",
                    "w": float(time_cost),
                    "timeout_ms": float(1.0 / float(time_cost)) if float(time_cost) > 0 else float("nan"),
                    "cost_threshold": float("nan"),
                    "p_threshold": float("nan"),
                    "k_consec": int(k),
                    "bayes_error_cost": float(bayes_error_cost),
                    "bayes_time_cost": float(time_cost),
                    **variant_meta(),
                }
                cond_row = {
                    **cond_meta,
                    **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                    **_decision_validity_stats(
                        decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                        p4_onset_ms=p4_onset_ms,
                        deviant_onset_ms=prepared.deviant_onset_ms,
                        found=found,
                    ),
                    "forced_deadline_rate": float(np.mean(forced_deadline.astype(float))) if forced_deadline.size > 0 else float("nan"),
                    "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                    "mean_expected_cost_at_rt": float(np.nanmean(expected_cost_at_rt)) if np.any(np.isfinite(expected_cost_at_rt)) else float("nan"),
                    "mean_wait_cost_at_rt": float(np.nanmean(wait_cost_at_rt)) if np.any(np.isfinite(wait_cost_at_rt)) else float("nan"),
                    "mean_cost_advantage_at_rt": float(np.nanmean(cost_advantage_at_rt)) if np.any(np.isfinite(cost_advantage_at_rt)) else float("nan"),
                    "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                    "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                    "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                    "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                    "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                    "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                    "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                }
                cond_rows.append(cond_row)
                for i in range(N):
                    trial_rows.append({
                        **cond_meta,
                        "trial_index": int(i),
                        "position": int(pos[i]),
                        "model_rt_ms": float(rt_ms[i]),
                        "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                        "found_flag": bool(found[i]),
                        "decision_token": int(decision_token[i]),
                        "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                        "window_start_token": int(prepared.window_start[i]),
                        "window_end_token": int(prepared.window_end_exclusive[i]),
                        "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                        "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                        "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                        "p4_onset_ms": float(p4_onset_ms),
                        "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                        "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                        "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                        "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                        "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                        "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                        "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                        "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                        "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                        "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                        "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                        "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                        "p_correct_at_rt": float(p_true_at_rt[i]),
                        "p_max_at_rt": float(p_best_at_rt[i]),
                        "commit_prob_at_rt": float("nan"),
                        "expected_cost_at_rt": float(expected_cost_at_rt[i]),
                        "wait_cost_at_rt": float(wait_cost_at_rt[i]),
                        "online_cost_delta_at_rt": float(cost_advantage_at_rt[i]),
                        "pred_class": int(pred_class_at_rt[i]),
                        "true_class": int(prepared.y_cls[i]),
                        "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                        "found": bool(found[i]),
                        "forced_deadline": bool(forced_deadline[i]),
                        "elapsed_ms": float((decision_token[i] - start[i]) * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                        "threshold_t": float("nan"),
                        "cost_answer_now": float(expected_cost_at_rt[i]),
                        "cost_wait": float(wait_cost_at_rt[i]),
                        "bayes_error_cost": float(bayes_error_cost),
                        "bayes_time_cost": float(time_cost),
                        "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                        "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                        "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                        "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                        "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                        "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                        "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                        "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                    })
    if include_marginal_wait:
        bayes_k_values = tuple(int(k) for k in (bayes_k_consec_list or k_consec_list))
        for time_cost in bayes_time_cost_list:
            for wait_lag_ms in bayes_wait_lag_ms_list:
                lag_tokens = int(max(1, math.ceil(float(wait_lag_ms) / float(prepared.token_ms))))
                lag_ms = float(lag_tokens) * float(prepared.token_ms)
                wait_cost = float(time_cost) * lag_ms
                for min_p in bayes_min_p_list:
                    for k in bayes_k_values:
                        rt_ms = np.full((N,), np.nan, dtype=float)
                        found = np.zeros((N,), dtype=bool)
                        forced_deadline = np.zeros((N,), dtype=bool)
                        decision_token = np.full((N,), -1, dtype=int)
                        p_true_at_rt = np.full((N,), np.nan, dtype=float)
                        p_best_at_rt = np.full((N,), np.nan, dtype=float)
                        risk_at_rt = np.full((N,), np.nan, dtype=float)
                        benefit_at_rt = np.full((N,), np.nan, dtype=float)
                        pred_class_at_rt = np.full((N,), -1, dtype=int)
                        for i in range(N):
                            s = int(effective_start[i]); e = int(end_excl[i])
                            if e <= s:
                                continue
                            window_p_best = p_max[i, s:e].astype(float)
                            window_p_true = p_corr[i, s:e].astype(float)
                            risk = 1.0 - window_p_best
                            hits = np.zeros_like(window_p_best, dtype=bool)
                            benefit = np.full_like(window_p_best, np.nan, dtype=float)
                            if len(window_p_best) > lag_tokens:
                                benefit[lag_tokens:] = risk[:-lag_tokens] - risk[lag_tokens:]
                                hits[lag_tokens:] = (window_p_best[lag_tokens:] >= float(min_p)) & (benefit[lag_tokens:] <= wait_cost)
                            idx = _first_k_consecutive(hits, int(k))
                            if idx < 0 and bool(bayes_force_deadline):
                                idx = len(window_p_best) - 1
                                forced_deadline[i] = True
                            if idx >= 0:
                                found[i] = True
                                decision_token[i] = s + idx
                                rt_ms[i] = float((decision_token[i] * prepared.token_ms) - prepared.rt_reference_time_ms[i])
                                p_true_at_rt[i] = float(window_p_true[idx])
                                p_best_at_rt[i] = float(window_p_best[idx])
                                risk_at_rt[i] = float(risk[idx])
                                benefit_at_rt[i] = float(benefit[idx])
                                pred_class_at_rt[i] = int(prepared.pred_class[i, decision_token[i]])
                        cond_meta = {
                            "readout_mode": "bayesian_marginal_wait",
                            "w": float("nan"),
                            "timeout_ms": float(1.0 / float(time_cost)) if float(time_cost) > 0 else float("nan"),
                            "cost_threshold": float("nan"),
                            "p_threshold": float(min_p),
                            "k_consec": int(k),
                            "bayes_error_cost": float(bayes_error_cost),
                            "bayes_time_cost": float(time_cost),
                            "bayes_threshold_start": float("nan"),
                            "bayes_threshold_min": float("nan"),
                            "bayes_urgency_slope": float("nan"),
                            "bayes_evidence_bound": float("nan"),
                            "bayes_leak": float("nan"),
                            "bayes_wait_lag_ms": float(lag_ms),
                            "bayes_min_p": float(min_p),
                            **variant_meta(),
                        }
                        cond_row = {
                            **cond_meta,
                            **_condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos),
                            **_decision_validity_stats(
                                decision_time_ms=np.where(decision_token >= 0, decision_token.astype(float) * float(prepared.token_ms), np.nan),
                                p4_onset_ms=p4_onset_ms,
                                deviant_onset_ms=prepared.deviant_onset_ms,
                                found=found,
                            ),
                            "forced_deadline_rate": float(np.mean(forced_deadline.astype(float))) if forced_deadline.size > 0 else float("nan"),
                            "accuracy_at_rt": float(np.mean((pred_class_at_rt[found] == prepared.y_cls[found]).astype(float))) if np.any(found) else float("nan"),
                            "mean_p_best_at_rt": float(np.nanmean(p_best_at_rt)) if np.any(np.isfinite(p_best_at_rt)) else float("nan"),
                            "mean_risk_at_rt": float(np.nanmean(risk_at_rt)) if np.any(np.isfinite(risk_at_rt)) else float("nan"),
                            "mean_benefit_at_rt": float(np.nanmean(benefit_at_rt)) if np.any(np.isfinite(benefit_at_rt)) else float("nan"),
                            "wait_cost": float(wait_cost),
                            "p_correct_at_window_end_P4": float(prepared.by_position_metrics[4]["p_correct_at_window_end"]),
                            "p_correct_at_window_end_P5": float(prepared.by_position_metrics[5]["p_correct_at_window_end"]),
                            "p_correct_at_window_end_P6": float(prepared.by_position_metrics[6]["p_correct_at_window_end"]),
                            "mean_p_correct_in_window_P4": float(prepared.by_position_metrics[4]["mean_p_correct_in_window"]),
                            "mean_p_correct_in_window_P5": float(prepared.by_position_metrics[5]["mean_p_correct_in_window"]),
                            "mean_p_correct_in_window_P6": float(prepared.by_position_metrics[6]["mean_p_correct_in_window"]),
                            "p_correct_ordering_P6_gt_P5_gt_P4": bool(prepared.overall_metrics.get("p_correct_ordering_ok", False)),
                        }
                        cond_rows.append(cond_row)
                        for i in range(N):
                            trial_rows.append({
                                **cond_meta,
                                "trial_index": int(i),
                                "position": int(pos[i]),
                                "model_rt_ms": float(rt_ms[i]),
                                "decision_time_ms": float(decision_token[i] * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                                "found_flag": bool(found[i]),
                                "decision_token": int(decision_token[i]),
                                "decision_token_relative_to_dev_onset": int(decision_token[i] - round(prepared.deviant_onset_ms[i] / prepared.token_ms)) if decision_token[i] >= 0 else -1,
                                "window_start_token": int(prepared.window_start[i]),
                                "window_end_token": int(prepared.window_end_exclusive[i]),
                                "readout_start_time_ms": float(prepared.window_start_time_ms[i]),
                                "readout_end_time_ms": float(prepared.window_end_time_ms[i]),
                                "window_start_time_ms": float(prepared.window_start_time_ms[i]),
                                "window_end_time_ms": float(prepared.window_end_time_ms[i]),
                                "rt_reference_time_ms": float(prepared.rt_reference_time_ms[i]),
                                "p4_onset_ms": float(p4_onset_ms),
                                "deviant_onset_ms": float(prepared.deviant_onset_ms[i]),
                                "deviant_offset_ms": float(prepared.deviant_offset_ms[i]),
                                "previous_tone_onset_ms": float(prepared.previous_tone_onset_ms[i]),
                                "previous_tone_offset_ms": float(prepared.previous_tone_offset_ms[i]),
                                "next_tone_onset_ms": float(prepared.next_tone_onset_ms[i]),
                                "next_tone_offset_ms": float(prepared.next_tone_offset_ms[i]),
                                "decision_minus_p4_onset_ms": float((decision_token[i] * prepared.token_ms) - p4_onset_ms) if decision_token[i] >= 0 else float("nan"),
                                "decision_minus_deviant_onset_ms": float((decision_token[i] * prepared.token_ms) - prepared.deviant_onset_ms[i]) if decision_token[i] >= 0 else float("nan"),
                                "is_negative_rt": bool(np.isfinite(rt_ms[i]) and rt_ms[i] < 0.0),
                                "is_pre_p4_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < p4_onset_ms),
                                "is_pre_deviant_crossing": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) < prepared.deviant_onset_ms[i]),
                                "is_valid_after_p4_onset": bool(decision_token[i] >= 0 and (decision_token[i] * prepared.token_ms) >= p4_onset_ms),
                                "p_correct_at_rt": float(p_true_at_rt[i]),
                                "p_max_at_rt": float(p_best_at_rt[i]),
                                "expected_cost_at_rt": float((1.0 - p_best_at_rt[i]) * float(bayes_error_cost) + float(time_cost) * max(0.0, rt_ms[i])) if np.isfinite(p_best_at_rt[i]) and np.isfinite(rt_ms[i]) else float("nan"),
                                "marginal_wait_benefit_at_rt": float(benefit_at_rt[i]),
                                "marginal_wait_cost": float(wait_cost),
                                "risk_at_rt": float(risk_at_rt[i]),
                                "pred_class": int(pred_class_at_rt[i]),
                                "true_class": int(prepared.y_cls[i]),
                                "correct_at_rt": bool(decision_token[i] >= 0 and int(pred_class_at_rt[i]) == int(prepared.y_cls[i])),
                                "found": bool(found[i]),
                                "forced_deadline": bool(forced_deadline[i]),
                                "elapsed_ms": float((decision_token[i] - start[i]) * prepared.token_ms) if decision_token[i] >= 0 else float("nan"),
                                "threshold_t": float(wait_cost),
                                "cost_answer_now": float((1.0 - p_best_at_rt[i]) * float(bayes_error_cost) + float(time_cost) * max(0.0, rt_ms[i])) if np.isfinite(p_best_at_rt[i]) and np.isfinite(rt_ms[i]) else float("nan"),
                                "bayes_error_cost": float(bayes_error_cost),
                                "bayes_time_cost": float(time_cost),
                                "bayes_wait_lag_ms": float(lag_ms),
                                "bayes_min_p": float(min_p),
                                "p_correct_at_dev_onset": float(prepared.p_correct_at_dev_onset[i]),
                                "p_correct_at_window_end": float(prepared.p_correct_at_window_end[i]),
                                "mean_p_correct_in_window": float(prepared.mean_p_correct_in_window[i]),
                                "auc_p_correct_in_window": float(prepared.auc_p_correct_in_window[i]),
                                "p_max_at_dev_onset": float(prepared.p_max_at_dev_onset[i]),
                                "p_max_at_window_end": float(prepared.p_max_at_window_end[i]),
                                "mean_p_max_in_window": float(prepared.mean_p_max_in_window[i]),
                                "auc_p_max_in_window": float(prepared.auc_p_max_in_window[i]),
                            })
    for row in trial_rows:
        i = int(row.get("trial_index", -1))
        if 0 <= i < N:
            decision_token = int(row.get("decision_token", -1))
            model_rt_raw = float(row.get("model_rt_ms", float("nan")))
            found_flag = bool(row.get("found_flag", row.get("found", False)))
            if (not found_flag) and (not np.isfinite(model_rt_raw)):
                row["model_rt_ms"] = float(NOT_FOUND_IMPUTED_RT_MS)
                row["not_found_imputed_rt_ms"] = float(NOT_FOUND_IMPUTED_RT_MS)
                row["model_rt_imputed_not_found"] = True
            else:
                row["not_found_imputed_rt_ms"] = float("nan")
                row["model_rt_imputed_not_found"] = False
            row["supervision_mode"] = str(prepared.supervision_mode)
            row["p4_token"] = int(round(float(prepared.p4_onset_ms[i]) / float(prepared.token_ms)))
            row["true_deviant_token"] = int(round(float(prepared.deviant_onset_ms[i]) / float(prepared.token_ms)))
            row["model_rt_raw"] = model_rt_raw
            model_rt_for_eval = float(row.get("model_rt_ms", float("nan")))
            row["model_rt_clipped"] = float(max(model_rt_for_eval, 0.0)) if np.isfinite(model_rt_for_eval) else float("nan")
            if decision_token >= 0:
                pred_cls = int(prepared.pred_class[i, decision_token])
                row["predicted_deviant_position"] = int(pred_cls + 4)
                row["decision_confidence"] = float(prepared.probs_max[i, decision_token])
            else:
                row["predicted_deviant_position"] = -1
                row["decision_confidence"] = float("nan")
    for row in cond_rows:
        row["supervision_mode"] = str(prepared.supervision_mode)
    return trial_rows, cond_rows


def aggregate_readout_trial_rows(trial_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not trial_rows:
        return []
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    key_fields = (
        "readout_mode",
        "readout_type",
        "w",
        "timeout_ms",
        "cost_threshold",
        "kappa",
        "p_threshold",
        "k_consec",
        "effective_threshold_formula",
        "bayes_error_cost",
        "bayes_time_cost",
        "bayes_threshold_start",
        "bayes_threshold_min",
        "bayes_urgency_slope",
        "bayes_evidence_bound",
        "bayes_leak",
        "bayes_wait_lag_ms",
        "bayes_min_p",
        "decision_min_elapsed_ms",
    )
    def _stable_group_key_value(v: Any) -> Any:
        # float("nan") != float("nan"), so raw NaNs split identical conditions
        # into one-trial groups. Canonicalize missing numeric metadata first.
        try:
            if isinstance(v, (float, np.floating)) and not np.isfinite(float(v)):
                return None
        except Exception:
            pass
        return v

    for row in trial_rows:
        key = tuple(_stable_group_key_value(row.get(k)) for k in key_fields)
        groups.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for key, rows in groups.items():
        rt_ms = np.asarray([float(r.get("model_rt_ms", float("nan"))) for r in rows], dtype=float)
        found = np.asarray([bool(r.get("found_flag", False)) for r in rows], dtype=bool)
        pos = np.asarray([int(r.get("position", -1)) for r in rows], dtype=int)
        decision_time_ms = np.asarray([float(r.get("decision_time_ms", float("nan"))) for r in rows], dtype=float)
        deviant_onset_ms = np.asarray([float(r.get("deviant_onset_ms", float("nan"))) for r in rows], dtype=float)
        p4_onset_ms_vals = np.asarray([float(r.get("p4_onset_ms", float("nan"))) for r in rows], dtype=float)
        p4_onset_ms = float(np.nanmedian(p4_onset_ms_vals)) if np.any(np.isfinite(p4_onset_ms_vals)) else float("nan")
        base = dict(zip(key_fields, key))
        stats = _condition_rt_stats(rt_ms=rt_ms, found=found, pos=pos)
        validity = _decision_validity_stats(
            decision_time_ms=decision_time_ms,
            p4_onset_ms=p4_onset_ms,
            deviant_onset_ms=deviant_onset_ms,
            found=found,
        )
        forced_deadline = np.asarray([bool(r.get("forced_deadline", False)) for r in rows], dtype=bool)
        timeout_flag = np.asarray([bool(r.get("timeout_flag", False)) for r in rows], dtype=bool)
        ec_at_rt = np.asarray([float(r.get("expected_cost_at_rt", float("nan"))) for r in rows], dtype=float)
        pmax_at_rt = np.asarray([float(r.get("p_max_at_rt", float("nan"))) for r in rows], dtype=float)
        pcorrect_at_rt = np.asarray([float(r.get("p_correct_at_rt", float("nan"))) for r in rows], dtype=float)
        correct_at_rt = np.asarray([bool(r.get("correct_at_rt", False)) for r in rows], dtype=bool)
        for p in [4, 5, 6]:
            rp = [r for r in rows if int(r.get("position", -1)) == p]
            if rp:
                base[f"p_correct_at_window_end_P{p}"] = float(np.nanmean([float(r.get("p_correct_at_window_end", float("nan"))) for r in rp]))
                base[f"mean_p_correct_in_window_P{p}"] = float(np.nanmean([float(r.get("mean_p_correct_in_window", float("nan"))) for r in rp]))
            else:
                base[f"p_correct_at_window_end_P{p}"] = float("nan")
                base[f"mean_p_correct_in_window_P{p}"] = float("nan")
        base["p_correct_ordering_P6_gt_P5_gt_P4"] = bool(
            np.isfinite(base["p_correct_at_window_end_P4"])
            and np.isfinite(base["p_correct_at_window_end_P5"])
            and np.isfinite(base["p_correct_at_window_end_P6"])
            and (base["p_correct_at_window_end_P6"] > base["p_correct_at_window_end_P5"] > base["p_correct_at_window_end_P4"])
        )
        acc_at_rt_by_pos: Dict[int, float] = {}
        for p in [4, 5, 6]:
            mp = pos == p
            acc_at_rt_by_pos[p] = float(np.mean(correct_at_rt[mp].astype(float))) if np.any(mp) else float("nan")
        out.append({
            **base,
            **stats,
            **validity,
            "forced_deadline_rate": float(np.mean(forced_deadline.astype(float))) if forced_deadline.size > 0 else float("nan"),
            "timeout_rate": float(np.mean(timeout_flag.astype(float))) if timeout_flag.size > 0 else float("nan"),
            "n_timeout_trials": int(np.sum(timeout_flag)),
            "acc_at_rt": float(np.mean(correct_at_rt.astype(float))) if correct_at_rt.size > 0 else float("nan"),
            "acc_at_valid_rt": float(np.mean(correct_at_rt[found].astype(float))) if np.any(found) else float("nan"),
            "acc_at_rt_P4": float(acc_at_rt_by_pos[4]),
            "acc_at_rt_P5": float(acc_at_rt_by_pos[5]),
            "acc_at_rt_P6": float(acc_at_rt_by_pos[6]),
            "mean_EC_at_response": float(np.nanmean(ec_at_rt)) if np.any(np.isfinite(ec_at_rt)) else float("nan"),
            "mean_pmax_at_response": float(np.nanmean(pmax_at_rt)) if np.any(np.isfinite(pmax_at_rt)) else float("nan"),
            "mean_pcorrect_at_response": float(np.nanmean(pcorrect_at_rt)) if np.any(np.isfinite(pcorrect_at_rt)) else float("nan"),
        })
    return out


def build_participant_level_rt_eval(trial_df) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    trial_df = trial_df.copy()
    defaults = {
        "readout_type": "",
        "kappa": float("nan"),
        "effective_threshold_formula": "",
        "timeout_flag": False,
    }
    for col_name, default_val in defaults.items():
        if col_name not in trial_df.columns:
            trial_df[col_name] = default_val
    trim_mask = np.isfinite(trial_df["human_rt_ms"].to_numpy(dtype=float))
    trim_mask &= (trial_df["human_rt_ms"].to_numpy(dtype=float) >= -1500.0)
    trim_mask &= (trial_df["human_rt_ms"].to_numpy(dtype=float) <= 1500.0)
    trim_df = trial_df.loc[trim_mask].copy()
    group_cols = ["participant", "checkpoint_path", "epoch", "readout_mode", "readout_type", "w", "timeout_ms", "cost_threshold", "kappa", "p_threshold", "k_consec", "effective_threshold_formula"]
    for keys, g in trim_df.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, keys))
        model_rt = g["model_rt_ms"].to_numpy(dtype=float)
        human_rt = g["human_rt_ms"].to_numpy(dtype=float)
        prior_prob = g["prior_prob"].to_numpy(dtype=float)
        prior_surprisal = g["prior_surprisal"].to_numpy(dtype=float)
        found = g["found_flag"].to_numpy(dtype=bool)
        timeout_flag = g["timeout_flag"].to_numpy(dtype=bool) if "timeout_flag" in g.columns else (~found)
        correct_at_rt = g["correct_at_rt"].to_numpy(dtype=bool) if "correct_at_rt" in g.columns else np.zeros_like(found, dtype=bool)
        imputed_not_found = (
            g["model_rt_imputed_not_found"].to_numpy(dtype=bool)
            if "model_rt_imputed_not_found" in g.columns
            else np.zeros_like(found, dtype=bool)
        )
        # Not-found trials are imputed as 1500 ms for eval/model-human RT summaries.
        # found_rate remains separate, so the imputation does not hide readout failures.
        use = np.isfinite(model_rt) & np.isfinite(human_rt)
        valid_or_imputed = (found | imputed_not_found) & np.isfinite(model_rt)
        a_r = _corr(model_rt[use], human_rt[use])
        b_r_prior = _corr(model_rt[use], prior_prob[use])
        b_r_sur = _corr(model_rt[use], prior_surprisal[use])
        c_r_prior = _corr(human_rt[use], prior_prob[use])
        c_r_sur = _corr(human_rt[use], prior_surprisal[use])
        a_R2 = float(a_r * a_r) if np.isfinite(a_r) else float("nan")
        c_R2_sur = float(c_r_sur * c_r_sur) if np.isfinite(c_r_sur) else float("nan")
        rows.append({
            **d,
            "n_trials": int(len(g)),
            "n_found_trials": int(np.sum(found)),
            "n_eval_rt_trials": int(np.sum(use)),
            "n_not_found_imputed_trials": int(np.sum(imputed_not_found)),
            "a_r": float(a_r),
            "a_R2": float(a_R2),
            "b_r_prior_prob": float(b_r_prior),
            "b_r_surprisal": float(b_r_sur),
            "c_r_prior_prob": float(c_r_prior),
            "c_r_surprisal": float(c_r_sur),
            "c_R2_surprisal": float(c_R2_sur),
            "standalone_delta_R2": float(a_R2 - c_R2_sur) if np.isfinite(a_R2) and np.isfinite(c_R2_sur) else float("nan"),
            "found_rate": float(np.mean(found.astype(float))) if found.size > 0 else float("nan"),
            "timeout_rate": float(np.mean(timeout_flag.astype(float))) if timeout_flag.size > 0 else float("nan"),
            "n_timeout_trials": int(np.sum(timeout_flag)),
            "miss_rate": float(np.mean((~found).astype(float))) if found.size > 0 else float("nan"),
            "acc_at_rt": float(np.mean(correct_at_rt.astype(float))) if correct_at_rt.size > 0 else float("nan"),
            "acc_at_valid_rt": float(np.mean(correct_at_rt[found].astype(float))) if np.any(found) else float("nan"),
            "mean_rt_ms": float(np.mean(model_rt[use])) if int(np.sum(use)) > 0 else float("nan"),
            "median_rt_ms": float(np.median(model_rt[use])) if int(np.sum(use)) > 0 else float("nan"),
            "mean_rt_ms_valid_or_imputed": float(np.mean(model_rt[valid_or_imputed])) if int(np.sum(valid_or_imputed)) > 0 else float("nan"),
            "median_rt_ms_valid_or_imputed": float(np.median(model_rt[valid_or_imputed])) if int(np.sum(valid_or_imputed)) > 0 else float("nan"),
            "floor_rate_5ms": float(np.mean(model_rt[use] <= 5.0)) if int(np.sum(use)) > 0 else float("nan"),
            "floor_rate_10ms": float(np.mean(model_rt[use] <= 10.0)) if int(np.sum(use)) > 0 else float("nan"),
            "floor_rate_20ms": float(np.mean(model_rt[use] <= 20.0)) if int(np.sum(use)) > 0 else float("nan"),
            "not_found_imputed_rt_ms": float(NOT_FOUND_IMPUTED_RT_MS),
            "warning_negative_a_r": bool(np.isfinite(a_r) and a_r < 0.0),
        })
    return rows


def _fisher_mean(rs: Sequence[float]) -> float:
    vals = np.asarray([r for r in rs if np.isfinite(r) and abs(r) < 1.0], dtype=float)
    if vals.size == 0:
        return float("nan")
    z = np.arctanh(vals)
    return float(np.tanh(np.mean(z)))


def build_group_level_rt_eval(participant_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not participant_rows:
        return []
    import pandas as pd

    df = pd.DataFrame(participant_rows)
    defaults = {
        "readout_type": "",
        "kappa": float("nan"),
        "effective_threshold_formula": "",
        "timeout_rate": float("nan"),
        "acc_at_valid_rt": float("nan"),
        "median_rt_ms": float("nan"),
    }
    for col_name, default_val in defaults.items():
        if col_name not in df.columns:
            df[col_name] = default_val
    group_cols = ["checkpoint_path", "epoch", "readout_mode", "readout_type", "w", "timeout_ms", "cost_threshold", "kappa", "p_threshold", "k_consec", "effective_threshold_formula"]
    rows: List[Dict[str, Any]] = []
    for keys, g in df.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, keys))
        better = g["a_R2"] > g["c_R2_surprisal"]
        rows.append({
            **d,
            "n_participants": int(g["participant"].nunique()),
            "mean_a_r_fisher": float(_fisher_mean(g["a_r"].tolist())),
            "mean_a_R2": float(g["a_R2"].mean()),
            "mean_c_R2_surprisal": float(g["c_R2_surprisal"].mean()),
            "mean_standalone_delta_R2": float(g["standalone_delta_R2"].mean()),
            "n_participants_RNN_R2_greater_prior_R2": int(np.sum(better.fillna(False).to_numpy(dtype=bool))),
            "proportion_RNN_R2_greater_prior_R2": float(np.mean(better.fillna(False).to_numpy(dtype=float))),
            "mean_b_r_surprisal": float(g["b_r_surprisal"].mean()),
            "mean_found_rate": float(g["found_rate"].mean()),
            "mean_timeout_rate": float(g["timeout_rate"].mean()) if "timeout_rate" in g.columns else float("nan"),
            "mean_miss_rate": float(g["miss_rate"].mean()),
            "mean_acc_at_rt": float(g["acc_at_rt"].mean()) if "acc_at_rt" in g.columns else float("nan"),
            "mean_acc_at_valid_rt": float(g["acc_at_valid_rt"].mean()) if "acc_at_valid_rt" in g.columns else float("nan"),
            "mean_median_rt_ms": float(g["median_rt_ms"].mean()) if "median_rt_ms" in g.columns else float("nan"),
            "mean_floor_rate_5ms": float(g["floor_rate_5ms"].mean()) if "floor_rate_5ms" in g.columns else float("nan"),
        })
    return rows
