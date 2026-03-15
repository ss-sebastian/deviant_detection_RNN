#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a sweep of trained PredictiveGRU checkpoints on HUMAN trials,
rendering stimuli WITH THE SAME noise hyperparameters as each checkpoint.

Changes you requested (IMPORTANT):
1) RT search starts at DEVIANT ONSET (not deviant end).
   - We define dev_start token as the first token of the deviant tone.

2) If the model already satisfies the RT criterion at the first checked token,
   RT is reported as 1 token (not 0).
   - i.e., rt_tokens = (first_detect_token - t0) + 1

This reduces the "RT==0 collapse" problem and matches your desired convention.

Inputs:
- sweep_root: folder containing runs like i=...__sig_other=...__rtp=... etc
- stimuli_dir: directory containing input_blocks.pt (B,10,8) and labels_blocks.pt (B,10) with positions {4,5,6}
- human_csv: per-trial RT rows (subject_id, isi_ms, rt_ms, optional position)

Alignment:
- Within each subject, human rows are mapped to (block_idx, trial_idx) by row order:
  row0 -> block0 trial0, row1 -> block0 trial1, ...

Outputs:
- out_dir/all_runs_summary.csv
- out_dir/all_trialtype_merged.csv
- per-run:
    out_dir/<run_name>/per_stim_trial.csv
    out_dir/<run_name>/per_human_trial_merged.csv
    out_dir/<run_name>/per_trialtype.csv
    out_dir/<run_name>/run_meta.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas required") from e

try:
    from sklearn.metrics import f1_score, roc_auc_score
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

from model import PredictiveGRU, ModelConfig


# =========================
# ERB binning + render
# =========================
def hz_to_erb_rate_np(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * f_hz)

def make_erb_edges(f_min_hz: float, f_max_hz: float, n_bins: int) -> np.ndarray:
    erb_min = float(hz_to_erb_rate_np(np.array([f_min_hz], dtype=np.float32))[0])
    erb_max = float(hz_to_erb_rate_np(np.array([f_max_hz], dtype=np.float32))[0])
    return np.linspace(erb_min, erb_max, int(n_bins) + 1, dtype=np.float32)

def freq_to_bin_erb(f_hz: float, edges_erb: np.ndarray) -> int:
    erb = float(hz_to_erb_rate_np(np.array([f_hz], dtype=np.float32))[0])
    j = int(np.searchsorted(edges_erb, erb, side="right") - 1)
    j = max(0, min(j, int(edges_erb.shape[0] - 2)))
    return j

def render_trial_onehot(
    freqs_8: np.ndarray,               # (8,)
    edges_erb: np.ndarray,
    n_bins: int,
    tone_T: int,
    isi_T: int,
    add_eos: bool,
    sigma_other_noise: float,
    p_other_noise: float,
    sigma_silence_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    eos_dim = 1 if add_eos else 0
    D = n_bins + eos_dim
    trial_T_tokens = 8 * tone_T + 7 * isi_T
    X = np.zeros((trial_T_tokens, D), dtype=np.float32)

    t = 0
    for i in range(8):
        bin_i = freq_to_bin_erb(float(freqs_8[i]), edges_erb)

        for _ in range(tone_T):
            X[t, bin_i] = 1.0

            # "other-bin" noise on tone tokens (matches training logic)
            if sigma_other_noise > 0 and p_other_noise > 0:
                if float(rng.random()) < float(p_other_noise):
                    noise = rng.normal(0.0, float(sigma_other_noise), size=(n_bins,)).astype(np.float32)
                    noise[bin_i] = 0.0
                    X[t, :n_bins] += noise

            t += 1

        # silence gaps between tones
        if i < 7:
            if sigma_silence_noise > 0:
                for _ in range(isi_T):
                    X[t, :n_bins] = rng.normal(0.0, float(sigma_silence_noise), size=(n_bins,)).astype(np.float32)
                    t += 1
            else:
                t += isi_T

    if add_eos:
        X[-1, n_bins] = 1.0
    return X


# =========================
# Labels + RT
# =========================
def labels_to_class_index(y_456: torch.Tensor) -> torch.Tensor:
    return (y_456 - 4).long()

def deviant_start_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    """
    Deviation position y_pos_456 in {4,5,6} -> dev_idx in {3,4,5} (0-based tone index).
    Each tone start is separated by (tone_T + isi_T).
    Deviant START token = dev_idx * (tone_T + isi_T)
    """
    dev_idx = (y_pos_456 - 1).long()  # {4,5,6}->{3,4,5}
    step = int(tone_T + isi_T)
    return dev_idx * step

def compute_rt_from_logits(
    logits: torch.Tensor,   # (B,10,Tt,3)
    y_cls: torch.Tensor,    # (B,10) in {0,1,2}
    dev_start: torch.Tensor,# (B,10) start token index for deviant (0..Tt-1)
    p_thresh: float,
    k_consec: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RT search begins at t0 = dev_start (deviant onset).
    If first detection occurs at t0 (or earliest possible), RT = 1 token, not 0:
      rt_tokens = (first_t - t0) + 1
    """
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)  # (B,10,Tt)
    y = y_cls.unsqueeze(-1).expand_as(pred)
    correct = (pred == y)

    py = probs.gather(
        dim=-1,
        index=y_cls.view(*y_cls.shape, 1, 1).expand(*pred.shape, 1)
    ).squeeze(-1)
    ok = correct & (py >= float(p_thresh))

    B, N, Tt = pred.shape
    rt = torch.full((B, N), -1, dtype=torch.long)
    found = torch.zeros((B, N), dtype=torch.bool)
    K = int(max(1, k_consec))

    for b in range(B):
        for tr in range(N):
            t0 = int(dev_start[b, tr].item())
            if t0 >= Tt:
                continue
            run = 0
            for t in range(t0, Tt):
                if bool(ok[b, tr, t].item()):
                    run += 1
                    if run >= K:
                        first_t = t - K + 1
                        rt[b, tr] = (first_t - t0) + 1  # <-- MIN RT = 1 token
                        found[b, tr] = True
                        break
                else:
                    run = 0
    return rt, found


# =========================
# Metrics helpers
# =========================
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
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(n_classes))))
    except Exception:
        return float("nan")


# =========================
# Params parsing (priority: ckpt.sweep > run_dir_name > ckpt.args)
# =========================
_RUN_RE = re.compile(
    r"sig_other=(?P<sig_other>[-0-9.]+).*?"
    r"p_other=(?P<p_other>[-0-9.]+).*?"
    r"sig_sil=(?P<sig_sil>[-0-9.]+).*?"
    r"rtp=(?P<rtp>[-0-9.]+).*?"
    r"rtk=(?P<rtk>[-0-9.]+)"
)

def parse_params_from_run_name(run_name: str) -> Optional[Dict[str, Any]]:
    m = _RUN_RE.search(run_name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "sigma_other_noise": float(d["sig_other"]),
        "p_other_noise": float(d["p_other"]),
        "sigma_silence_noise": float(d["sig_sil"]),
        "rt_p_thresh": float(d["rtp"]),
        "rt_k_consec": int(float(d["rtk"])),
    }

def read_params_from_ckpt_or_name(ckpt: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    sweep = ckpt.get("sweep", None)
    if isinstance(sweep, dict):
        need = ["sigma_other_noise", "p_other_noise", "sigma_silence_noise", "rt_p_thresh", "rt_k_consec"]
        if all(k in sweep for k in need):
            return {
                "sigma_other_noise": float(sweep["sigma_other_noise"]),
                "p_other_noise": float(sweep["p_other_noise"]),
                "sigma_silence_noise": float(sweep["sigma_silence_noise"]),
                "rt_p_thresh": float(sweep["rt_p_thresh"]),
                "rt_k_consec": int(sweep["rt_k_consec"]),
                "source": "ckpt.sweep",
            }

    p = parse_params_from_run_name(run_dir.name)
    if p is not None:
        p["source"] = "run_dir_name"
        return p

    args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    return {
        "sigma_other_noise": float(args.get("sigma_other_noise", 0.0)),
        "p_other_noise": float(args.get("p_other_noise", 1.0)),
        "sigma_silence_noise": float(args.get("sigma_silence_noise", 0.0)),
        "rt_p_thresh": float(args.get("rt_p_thresh", 0.7)),
        "rt_k_consec": int(args.get("rt_k_consec", 3)),
        "source": "ckpt.args_fallback",
    }

def get_ckpt_train_args(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    a = ckpt.get("args", {})
    return a if isinstance(a, dict) else {}


# =========================
# Model forward (chunked)
# =========================
@torch.no_grad()
def forward_logits_all(model: PredictiveGRU, x: torch.Tensor, chunk_len: int) -> torch.Tensor:
    """
    x: (B,T,D) on device
    return logits_all on CPU: (B,T,3)
    """
    model.eval()
    h = None
    logits_chunks = []
    T = x.shape[1]
    for s in range(0, T, int(chunk_len)):
        e = min(s + int(chunk_len), T)
        x_in = x[:, s:e, :]
        h_seq, h = model.forward_chunk(x_in, h0=h)
        logits = model.classify_tokens(h_seq)
        logits_chunks.append(logits.detach().to("cpu"))
        h = h.detach()
    return torch.cat(logits_chunks, dim=1)


# =========================
# Trialtype key
# =========================
def make_trialtype_key(freqs_8: np.ndarray, position: int, isi_ms: int) -> str:
    seq = ",".join([f"{float(x):.6g}" for x in freqs_8.tolist()])
    return f"isi={int(isi_ms)}|pos={int(position)}|seq={seq}"


# =========================
# Align human rows -> (block_idx, trial_idx) within subject
# =========================
def align_human_within_subject(
    human_df: "pd.DataFrame",
    isi_filter: Optional[int],
) -> "pd.DataFrame":
    df = human_df.copy()

    if isi_filter is not None and "isi_ms" in df.columns:
        df["isi_ms"] = pd.to_numeric(df["isi_ms"], errors="coerce")
        df = df[df["isi_ms"] == int(isi_filter)].copy()

    if "rt_ms" not in df.columns:
        raise RuntimeError("human_csv must contain rt_ms")

    if "subject_id" not in df.columns:
        df["subject_id"] = "subj_0"

    df = df.reset_index(drop=True)

    out_rows = []
    for sid, g in df.groupby("subject_id", sort=True):
        g = g.copy().reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        idx = np.arange(n)
        g["block_idx"] = (idx // 10).astype(int)
        g["trial_idx"] = (idx % 10).astype(int)
        out_rows.append(g)

    return pd.concat(out_rows, ignore_index=True)


# =========================
# Compute model outputs on stimuli (freqs -> onehot w/ noise params)
# =========================
@torch.no_grad()
def compute_model_on_stimuli(
    model: PredictiveGRU,
    X_blocks: np.ndarray,     # (B,10,8) freqs
    Y_blocks: np.ndarray,     # (B,10) positions {4,5,6}
    isi_ms: int,
    tone_ms: int,
    token_ms: int,
    f_min_hz: float,
    f_max_hz: float,
    n_bins: int,
    add_eos: bool,
    sigma_other_noise: float,
    p_other_noise: float,
    sigma_silence_noise: float,
    rt_p_thresh: float,
    rt_k_consec: int,
    chunk_len: int,
    batch_blocks: int,
    device: torch.device,
    seed: int,
) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    if tone_ms % token_ms != 0:
        raise ValueError("tone_ms must be divisible by token_ms")
    if isi_ms % token_ms != 0:
        raise ValueError("isi_ms must be divisible by token_ms")

    tone_T = tone_ms // token_ms
    isi_T = isi_ms // token_ms
    trial_T_tokens = 8 * tone_T + 7 * isi_T
    T_block = 10 * trial_T_tokens

    edges_erb = make_erb_edges(float(f_min_hz), float(f_max_hz), int(n_bins))
    D = int(n_bins) + (1 if add_eos else 0)

    B_total = int(X_blocks.shape[0])

    rows = []
    y_true_all, y_pred_all, y_prob_all = [], [], []
    rt_found_total, rt_miss_total = 0, 0

    for i0 in range(0, B_total, int(batch_blocks)):
        b1 = min(B_total, i0 + int(batch_blocks))
        B = b1 - i0

        X = np.zeros((B, T_block, D), dtype=np.float32)
        Y = Y_blocks[i0:b1, :].astype(np.int64)  # (B,10)

        for bi in range(B):
            block_idx = i0 + bi
            freqs10 = X_blocks[block_idx]  # (10,8)
            for tr in range(10):
                # deterministic per-trial RNG (reproducible)
                seed_trial = int(seed) + int(block_idx) * 1000 + int(tr) * 17
                rng = np.random.default_rng(seed_trial)

                Xt = render_trial_onehot(
                    freqs_8=freqs10[tr].astype(np.float32),
                    edges_erb=edges_erb,
                    n_bins=int(n_bins),
                    tone_T=int(tone_T),
                    isi_T=int(isi_T),
                    add_eos=bool(add_eos),
                    sigma_other_noise=float(sigma_other_noise),
                    p_other_noise=float(p_other_noise),
                    sigma_silence_noise=float(sigma_silence_noise),
                    rng=rng,
                )
                X[bi, tr*trial_T_tokens:(tr+1)*trial_T_tokens, :] = Xt

        x = torch.from_numpy(X).to(device)
        y_pos_456 = torch.from_numpy(Y).long()      # CPU
        y_cls = labels_to_class_index(y_pos_456)    # CPU

        logits_all = forward_logits_all(model, x, chunk_len=int(chunk_len))  # CPU (B,T,3)
        logits_trial = logits_all.view(B, 10, trial_T_tokens, 3)
        logits_end = logits_trial[:, :, -1, :]  # (B,10,3)

        probs = torch.softmax(logits_end, dim=-1).numpy().reshape(-1, 3)
        pred = probs.argmax(axis=1)
        true = y_cls.numpy().reshape(-1)

        y_true_all.append(true)
        y_pred_all.append(pred)
        y_prob_all.append(probs)

        # ---- RT: start at deviant onset, min 1 token ----
        dev_start = deviant_start_token_in_trial(y_pos_456=y_pos_456, tone_T=int(tone_T), isi_T=int(isi_T))
        rt_tokens, found = compute_rt_from_logits(
            logits=logits_trial,
            y_cls=y_cls,
            dev_start=dev_start,
            p_thresh=float(rt_p_thresh),
            k_consec=int(rt_k_consec),
        )
        rt_ms = rt_tokens.float().numpy() * float(token_ms)
        found_np = found.numpy().astype(bool)

        rt_found_total += int(found_np.sum())
        rt_miss_total += int((~found_np).sum())

        for bi in range(B):
            block_idx = i0 + bi
            for tr in range(10):
                k = bi * 10 + tr
                rows.append({
                    "block_idx": int(block_idx),
                    "trial_idx": int(tr),
                    "isi_ms": int(isi_ms),
                    "position_stim": int(Y[bi, tr]),
                    "true_cls": int(true[k]),
                    "pred_cls": int(pred[k]),
                    "p0": float(probs[k, 0]),
                    "p1": float(probs[k, 1]),
                    "p2": float(probs[k, 2]),
                    "model_rt_ms": float(rt_ms[bi, tr]),
                    "found": int(found_np[bi, tr]),
                    "freqs_8": X_blocks[block_idx, tr, :].tolist(),
                    "trial_type": make_trialtype_key(X_blocks[block_idx, tr, :], int(Y[bi, tr]), int(isi_ms)),
                })

    df = pd.DataFrame(rows)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.zeros((0, 3), dtype=np.float32)

    acc = float((y_true == y_pred).mean()) if y_true.size else float("nan")
    f1 = safe_f1_macro(y_true, y_pred) if y_true.size else float("nan")
    auc = safe_auc_ovr(y_true, y_prob, n_classes=3) if y_true.size else float("nan")

    metrics = {
        "acc": acc,
        "f1_macro": f1,
        "auc_ovr": auc,
        "rt_found": int(rt_found_total),
        "rt_miss": int(rt_miss_total),
    }
    return df, metrics


import numpy as np
import pandas as pd

def corr_pear_spear_kendall(a: np.ndarray, b: np.ndarray):
    """
    Returns:
      pearson_r, spearman_rho, kendall_tau_b
    Notes:
      - Spearman computed via average ranks (tie-aware).
      - Kendall tau-b is robust under many ties.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return (float("nan"), float("nan"), float("nan"))

    aa = a[m]
    bb = b[m]

    # Pearson on raw values
    pear = float(np.corrcoef(aa, bb)[0, 1])

    # Spearman with tie-aware average ranks
    ra = pd.Series(aa).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(bb).rank(method="average").to_numpy(dtype=float)
    spear = float(np.corrcoef(ra, rb)[0, 1])

    # Kendall's tau-b (tie-aware). Try scipy if installed; else manual fallback = NaN.
    tau_b = float("nan")
    try:
        from scipy.stats import kendalltau
        tau_b = float(kendalltau(aa, bb, variant="b").correlation)
    except Exception:
        tau_b = float("nan")

    return pear, spear, tau_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, required=True)
    ap.add_argument("--human_csv", type=str, required=True)
    ap.add_argument("--stimuli_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--isi_filter", type=int, default=700)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    # speed knobs
    ap.add_argument("--batch_blocks", type=int, default=64)
    ap.add_argument("--ckpt_prefer_best_isi", action="store_true")

    # override-only if you must; otherwise use ckpt.args
    ap.add_argument("--tone_ms", type=int, default=None)
    ap.add_argument("--token_ms", type=int, default=None)
    ap.add_argument("--f_min_hz", type=float, default=None)
    ap.add_argument("--f_max_hz", type=float, default=None)
    ap.add_argument("--n_bins", type=int, default=None)
    ap.add_argument("--chunk_len", type=int, default=None)
    ap.add_argument("--add_eos", action="store_true")
    ap.add_argument("--no_add_eos", action="store_true")

    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    stim_dir = Path(args.stimuli_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xb = stim_dir / "input_blocks.pt"
    yb = stim_dir / "labels_blocks.pt"
    if not xb.exists() or not yb.exists():
        raise RuntimeError("stimuli_dir must contain input_blocks.pt and labels_blocks.pt")

    X_blocks = torch.load(xb, map_location="cpu").numpy().astype(np.float32)  # (B,10,8)
    Y_blocks = torch.load(yb, map_location="cpu").numpy().astype(np.int64)    # (B,10)
    if X_blocks.ndim != 3 or X_blocks.shape[1:] != (10, 8):
        raise RuntimeError(f"input_blocks.pt must be (B,10,8). got {X_blocks.shape}")
    if Y_blocks.ndim != 2 or Y_blocks.shape[1:] != (10,):
        raise RuntimeError(f"labels_blocks.pt must be (B,10). got {Y_blocks.shape}")

    df_h = pd.read_csv(Path(args.human_csv))
    df_h = align_human_within_subject(df_h, isi_filter=int(args.isi_filter))

    # add_eos override logic
    override_add_eos = None
    if args.add_eos and args.no_add_eos:
        raise RuntimeError("Cannot set both --add_eos and --no_add_eos")
    if args.add_eos:
        override_add_eos = True
    if args.no_add_eos:
        override_add_eos = False

    run_dirs = sorted([p for p in sweep_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    all_meta: List[Dict[str, Any]] = []
    all_tt_rows: List["pd.DataFrame"] = []

    dev = torch.device(args.device)

    for run_dir in run_dirs:
        isi = int(args.isi_filter)
        candidates = []
        if args.ckpt_prefer_best_isi:
            candidates.append(run_dir / f"best_isi{isi}.pt")
        candidates += [run_dir / f"best_isi{isi}.pt", run_dir / "best.pt", run_dir / "last.pt"]

        ckpt_path = None
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            continue

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        params = read_params_from_ckpt_or_name(ckpt, run_dir)
        train_args = get_ckpt_train_args(ckpt)
        cfg_dict = ckpt.get("cfg", None)
        if cfg_dict is None:
            continue

        # infer rendering params from ckpt.args unless overridden
        tone_ms  = int(args.tone_ms)  if args.tone_ms  is not None else int(train_args.get("tone_ms", 50))
        token_ms = int(args.token_ms) if args.token_ms is not None else int(train_args.get("token_ms", 10))
        f_min_hz = float(args.f_min_hz) if args.f_min_hz is not None else float(train_args.get("f_min_hz", 1300.0))
        f_max_hz = float(args.f_max_hz) if args.f_max_hz is not None else float(train_args.get("f_max_hz", 1700.0))
        n_bins   = int(args.n_bins) if args.n_bins is not None else int(train_args.get("n_bins", 128))
        add_eos  = bool(override_add_eos) if override_add_eos is not None else bool(train_args.get("add_eos", False))
        chunk_len = int(args.chunk_len) if args.chunk_len is not None else int(train_args.get("chunk_len", 600))

        cfg = ModelConfig(
            input_dim=int(cfg_dict["input_dim"]),
            hidden_dim=int(cfg_dict["hidden_dim"]),
            num_layers=int(cfg_dict["num_layers"]),
            dropout=float(cfg_dict["dropout"]),
            layer_norm=bool(cfg_dict.get("layer_norm", False)),
        )

        expected_input_dim = int(n_bins) + (1 if add_eos else 0)
        if expected_input_dim != int(cfg.input_dim):
            raise RuntimeError(
                f"input_dim mismatch for {run_dir.name}: cfg.input_dim={cfg.input_dim} but n_bins/add_eos -> {expected_input_dim}"
            )

        model = PredictiveGRU(cfg).to(dev)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        model_trials, metrics = compute_model_on_stimuli(
            model=model,
            X_blocks=X_blocks,
            Y_blocks=Y_blocks,
            isi_ms=int(args.isi_filter),
            tone_ms=int(tone_ms),
            token_ms=int(token_ms),
            f_min_hz=float(f_min_hz),
            f_max_hz=float(f_max_hz),
            n_bins=int(n_bins),
            add_eos=bool(add_eos),
            sigma_other_noise=float(params["sigma_other_noise"]),
            p_other_noise=float(params["p_other_noise"]),
            sigma_silence_noise=float(params["sigma_silence_noise"]),
            rt_p_thresh=float(params["rt_p_thresh"]),
            rt_k_consec=int(params["rt_k_consec"]),
            chunk_len=int(chunk_len),
            batch_blocks=int(args.batch_blocks),
            device=dev,
            seed=int(args.seed),
        )

        # merge human
        merged = df_h.merge(
            model_trials[[
                "block_idx", "trial_idx", "trial_type", "position_stim",
                "true_cls", "pred_cls", "p0", "p1", "p2", "model_rt_ms", "found"
            ]],
            on=["block_idx", "trial_idx"],
            how="left",
            validate="many_to_one"
        )
        merged["rt_ms"] = pd.to_numeric(merged["rt_ms"], errors="coerce")
        merged = merged[merged["rt_ms"].notna()].copy()

        # trialtype aggregation
        g = merged.groupby("trial_type", sort=True)
        per_tt = g.agg(
            n=("trial_type", "size"),
            human_rt_ms=("rt_ms", "mean"),
            model_rt_ms=("model_rt_ms", "mean"),
            found_rate=("found", "mean"),
            acc=("pred_cls", lambda x: float(np.mean(x.to_numpy() == merged.loc[x.index, "true_cls"].to_numpy()))),
        ).reset_index()

        pear, spear, tau_b = corr_pear_spear_kendall(
            per_tt["human_rt_ms"].to_numpy(float),
            per_tt["model_rt_ms"].to_numpy(float),
        )

        run_out = out_dir / run_dir.name
        run_out.mkdir(parents=True, exist_ok=True)
        model_trials.to_csv(run_out / "per_stim_trial.csv", index=False)
        merged.to_csv(run_out / "per_human_trial_merged.csv", index=False)
        per_tt.to_csv(run_out / "per_trialtype.csv", index=False)

        meta = {
            "run_dir": str(run_dir),
            "ckpt": str(ckpt_path),
            "params_used": params,

            "isi_ms": int(args.isi_filter),
            "tone_ms": int(tone_ms),
            "token_ms": int(token_ms),
            "f_min_hz": float(f_min_hz),
            "f_max_hz": float(f_max_hz),
            "n_bins": int(n_bins),
            "add_eos": bool(add_eos),
            "chunk_len": int(chunk_len),
            "batch_blocks": int(args.batch_blocks),

            "acc": float(metrics["acc"]),
            "f1_macro": float(metrics["f1_macro"]),
            "auc_ovr": float(metrics["auc_ovr"]),
            "rt_found": int(metrics["rt_found"]),
            "rt_miss": int(metrics["rt_miss"]),

            "pearson_trialtype_rt": float(pear),
            "spearman_trialtype_rt": float(spear),
            "kendall_tau_b_trialtype_rt": float(tau_b),
            "spearman_rank_method": "average",
            "n_trialtypes": int(per_tt.shape[0]),
            "n_human_trials_used": int(merged.shape[0]),
            "n_stim_trials_total": int(model_trials.shape[0]),
        }
        (run_out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        all_meta.append(meta)

        per_tt["model_id"] = run_dir.name
        per_tt["sigma_other_noise"] = float(params["sigma_other_noise"])
        per_tt["p_other_noise"] = float(params["p_other_noise"])
        per_tt["sigma_silence_noise"] = float(params["sigma_silence_noise"])
        per_tt["rt_p_thresh"] = float(params["rt_p_thresh"])
        per_tt["rt_k_consec"] = int(params["rt_k_consec"])
        all_tt_rows.append(per_tt)

    if len(all_meta) == 0:
        raise RuntimeError("No runs evaluated.")

    summary = pd.DataFrame(all_meta)
    summary.to_csv(out_dir / "all_runs_summary.csv", index=False)

    if len(all_tt_rows) > 0:
        merged_tt = pd.concat(all_tt_rows, ignore_index=True)
        merged_tt.to_csv(out_dir / "all_trialtype_merged.csv", index=False)

    print((out_dir / "all_runs_summary.csv").resolve())
    if (out_dir / "all_trialtype_merged.csv").exists():
        print((out_dir / "all_trialtype_merged.csv").resolve())


if __name__ == "__main__":
    main()