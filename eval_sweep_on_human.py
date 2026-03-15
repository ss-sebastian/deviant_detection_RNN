#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a sweep of trained PredictiveGRU checkpoints on HUMAN trials,
using the SAME stimuli sequences as training (input_blocks.pt).

human_csv is assumed to have per-trial rows with at least:
  subject_id, isi_ms, position, rt_ms
(no tone frequency columns needed)

We align human rows to stimuli trials by row order:
  row 0 -> stimuli block 0 trial 0
  row 1 -> stimuli block 0 trial 1
  ...
This is the standard if you logged human trials in presentation order.

Outputs:
- out_dir/all_runs_summary.csv
- out_dir/all_trialtype_merged.csv
- per-run:
    out_dir/<run_name>/per_trial.csv
    out_dir/<run_name>/per_trialtype.csv
    out_dir/<run_name>/run_meta.json
"""

from __future__ import annotations

import argparse
import copy
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

# Optional sklearn
try:
    from sklearn.metrics import f1_score, roc_auc_score
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

from model import PredictiveGRU, ModelConfig


# -------------------------
# ERB binning + render
# -------------------------
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
            if sigma_other_noise > 0 and p_other_noise > 0:
                if float(rng.random()) < float(p_other_noise):
                    noise = rng.normal(0.0, float(sigma_other_noise), size=(n_bins,)).astype(np.float32)
                    noise[bin_i] = 0.0
                    X[t, :n_bins] += noise
            t += 1

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


# -------------------------
# RT + labels
# -------------------------
def labels_to_class_index(y_456: torch.Tensor) -> torch.Tensor:
    return (y_456 - 4).long()

def deviant_end_token_in_trial(y_pos_456: torch.Tensor, tone_T: int, isi_T: int) -> torch.Tensor:
    dev_idx = (y_pos_456 - 1).long()  # {4,5,6}->{3,4,5}
    step = int(tone_T + isi_T)
    return dev_idx * step + int(tone_T) - 1

def compute_rt_from_logits(
    logits: torch.Tensor,   # (B,10,Tt,3)
    y_cls: torch.Tensor,    # (B,10) in {0,1,2}
    dev_end: torch.Tensor,  # (B,10)
    p_thresh: float,
    k_consec: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    y_cls = y_cls.long()
    mn = int(y_cls.min().item())
    mx = int(y_cls.max().item())
    if mn < 0 or mx >= 3:
        raise ValueError(f"invalid y_cls range [{mn},{mx}] (expected 0..2).")

    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)              # (B,10,Tt)
    y = y_cls.unsqueeze(-1).expand_as(pred)
    correct = (pred == y)

    py = probs.gather(dim=-1, index=y_cls.view(*y_cls.shape, 1, 1).expand(*pred.shape, 1)).squeeze(-1)
    ok = correct & (py >= float(p_thresh))

    B, N, Tt = pred.shape
    rt = torch.full((B, N), -1, dtype=torch.long)
    found = torch.zeros((B, N), dtype=torch.bool)
    K = int(max(1, k_consec))

    for b in range(B):
        for tr in range(N):
            t0 = int(dev_end[b, tr].item()) + 1
            if t0 >= Tt:
                continue
            run = 0
            for t in range(t0, Tt):
                if bool(ok[b, tr, t].item()):
                    run += 1
                    if run >= K:
                        first_t = t - K + 1
                        rt[b, tr] = first_t - t0
                        found[b, tr] = True
                        break
                else:
                    run = 0
    return rt, found


# -------------------------
# Metrics
# -------------------------
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


# -------------------------
# Parse run params (ckpt sweep or folder name)
# -------------------------
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
        out = {}
        for k in ["sigma_other_noise", "p_other_noise", "sigma_silence_noise", "rt_p_thresh", "rt_k_consec"]:
            if k in sweep:
                out[k] = sweep[k]
        if len(out) == 5:
            out["source"] = "ckpt.sweep"
            return out

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


# -------------------------
# Model forward (chunked)
# -------------------------
@torch.no_grad()
def forward_logits_all(model: PredictiveGRU, x: torch.Tensor, chunk_len: int) -> torch.Tensor:
    model.eval()
    B, T, D = x.shape
    h = None
    logits_chunks = []
    for s in range(0, T, int(chunk_len)):
        e = min(s + int(chunk_len), T)
        x_in = x[:, s:e, :]
        h_seq, h = model.forward_chunk(x_in, h0=h)
        logits = model.classify_tokens(h_seq)
        logits_chunks.append(logits.cpu())
        h = h.detach()
    return torch.cat(logits_chunks, dim=1)  # (B,T,3)


# -------------------------
# Alignment: human rows <-> stimuli trials
# -------------------------
def align_human_to_stimuli(
    human_df: "pd.DataFrame",
    X_blocks: np.ndarray,   # (B,10,8)
    Y_blocks: Optional[np.ndarray],  # (B,10) {4,5,6}
    isi_filter: Optional[int],
) -> "pd.DataFrame":
    df = human_df.copy()
    if isi_filter is not None and "isi_ms" in df.columns:
        df["isi_ms"] = pd.to_numeric(df["isi_ms"], errors="coerce")
        df = df[df["isi_ms"] == int(isi_filter)].copy()

    if "position" not in df.columns or "rt_ms" not in df.columns:
        raise RuntimeError("human_csv must contain: position, rt_ms")

    df = df.reset_index(drop=True)

    # total trials available in stimuli
    B = X_blocks.shape[0]
    total_trials = B * 10

    if len(df) > total_trials:
        # Human has more rows than stimuli: truncate human to match stimuli
        df = df.iloc[:total_trials].copy()

    if len(df) < total_trials:
        # Stimuli has more trials than human: truncate stimuli later
        pass

    # attach block_idx / trial_idx by row order
    idx = np.arange(len(df))
    df["block_idx"] = (idx // 10).astype(int)
    df["trial_idx"] = (idx % 10).astype(int)

    # attach freqs from stimuli
    freqs = []
    pos_stim = []
    for i in range(len(df)):
        b = int(df.loc[i, "block_idx"])
        t = int(df.loc[i, "trial_idx"])
        freqs.append(X_blocks[b, t, :].tolist())
        if Y_blocks is not None:
            pos_stim.append(int(Y_blocks[b, t]))
        else:
            pos_stim.append(-1)

    df["freqs_8"] = freqs
    if Y_blocks is not None:
        df["position_stim"] = pos_stim
        # mismatch rate diagnostic
        df["pos_match"] = (df["position"].astype(int) == df["position_stim"].astype(int)).astype(int)

    return df


def make_trialtype_key(freqs_8: List[float], position: int, isi_ms: int) -> str:
    seq = ",".join([f"{float(x):.6g}" for x in freqs_8])
    return f"isi={int(isi_ms)}|pos={int(position)}|seq={seq}"


# -------------------------
# Evaluate one run
# -------------------------
def evaluate_one_run(
    run_dir: Path,
    ckpt_path: Path,
    human_df_aligned: "pd.DataFrame",
    out_dir: Path,
    isi_ms: int,
    tone_ms: int,
    token_ms: int,
    f_min_hz: float,
    f_max_hz: float,
    n_bins: int,
    add_eos: bool,
    chunk_len: int,
    batch_blocks: int,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    params = read_params_from_ckpt_or_name(ckpt, run_dir)

    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError(f"Checkpoint missing cfg: {ckpt_path}")

    cfg = ModelConfig(
        input_dim=int(cfg_dict["input_dim"]),
        hidden_dim=int(cfg_dict["hidden_dim"]),
        num_layers=int(cfg_dict["num_layers"]),
        dropout=float(cfg_dict["dropout"]),
        layer_norm=bool(cfg_dict.get("layer_norm", False)),
    )

    dev = torch.device(device)
    model = PredictiveGRU(cfg).to(dev)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    if tone_ms % token_ms != 0:
        raise ValueError("tone_ms must be divisible by token_ms")
    if isi_ms % token_ms != 0:
        raise ValueError("isi_ms must be divisible by token_ms")

    tone_T = tone_ms // token_ms
    isi_T = isi_ms // token_ms
    trial_T_tokens = 8 * tone_T + 7 * isi_T
    T_block = 10 * trial_T_tokens

    edges_erb = make_erb_edges(float(f_min_hz), float(f_max_hz), int(n_bins))
    eos_dim = 1 if add_eos else 0
    D = int(n_bins) + eos_dim

    df = human_df_aligned.copy()
    df["trial_type"] = df.apply(lambda r: make_trialtype_key(r["freqs_8"], int(r["position"]), int(r["isi_ms"])), axis=1)

    # group into complete blocks
    blocks = []
    for b, g in df.groupby("block_idx", sort=True):
        g = g.sort_values("trial_idx")
        if len(g) != 10:
            continue
        freqs10 = np.stack([np.array(x, dtype=np.float32) for x in g["freqs_8"].tolist()], axis=0)  # (10,8)
        y_pos10 = g["position"].to_numpy(np.int64)
        hrt10 = g["rt_ms"].to_numpy(np.float32)
        ttypes = g["trial_type"].tolist()
        sid = g["subject_id"].tolist() if "subject_id" in g.columns else [None]*10
        blocks.append((int(b), freqs10, y_pos10, hrt10, ttypes, sid))

    if len(blocks) == 0:
        raise RuntimeError("No complete blocks of 10 trials after alignment.")

    per_trial_rows: List[Dict[str, Any]] = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    rt_found_total = 0
    rt_miss_total = 0

    for i0 in range(0, len(blocks), int(batch_blocks)):
        batch = blocks[i0:i0 + int(batch_blocks)]
        B = len(batch)

        X = np.zeros((B, T_block, D), dtype=np.float32)
        Y = np.zeros((B, 10), dtype=np.int64)
        HRT = np.zeros((B, 10), dtype=np.float32)
        BT = [[None]*10 for _ in range(B)]
        BIDX = np.zeros((B,), dtype=np.int64)
        SID = [[None]*10 for _ in range(B)]

        for bi, (block_idx, freqs10, y_pos10, hrt10, ttypes, sids) in enumerate(batch):
            BIDX[bi] = block_idx
            Y[bi, :] = y_pos10
            HRT[bi, :] = hrt10
            for tr in range(10):
                BT[bi][tr] = ttypes[tr]
                SID[bi][tr] = sids[tr]
                rng = np.random.default_rng(int(seed) + int(block_idx) * 1000 + tr * 17)
                Xt = render_trial_onehot(
                    freqs_8=freqs10[tr],
                    edges_erb=edges_erb,
                    n_bins=int(n_bins),
                    tone_T=int(tone_T),
                    isi_T=int(isi_T),
                    add_eos=bool(add_eos),
                    sigma_other_noise=float(params["sigma_other_noise"]),
                    p_other_noise=float(params["p_other_noise"]),
                    sigma_silence_noise=float(params["sigma_silence_noise"]),
                    rng=rng,
                )
                X[bi, tr*trial_T_tokens:(tr+1)*trial_T_tokens, :] = Xt

        x = torch.from_numpy(X).to(dev)
        y_pos_456 = torch.from_numpy(Y).long()  # cpu
        y_cls = labels_to_class_index(y_pos_456)  # cpu
        y_cls_np = y_cls.numpy().reshape(-1)

        logits_all = forward_logits_all(model, x, chunk_len=int(chunk_len))   # (B,T,3) cpu
        logits_trial = logits_all.view(B, 10, trial_T_tokens, 3)
        logits_end = logits_trial[:, :, -1, :]  # (B,10,3)

        probs = torch.softmax(logits_end, dim=-1).numpy().reshape(-1, 3)
        pred = probs.argmax(axis=1)

        y_true_all.append(y_cls_np)
        y_pred_all.append(pred)
        y_prob_all.append(probs)

        dev_end = deviant_end_token_in_trial(y_pos_456=y_pos_456, tone_T=int(tone_T), isi_T=int(isi_T))
        rt_tokens, found = compute_rt_from_logits(
            logits=logits_trial,
            y_cls=y_cls,
            dev_end=dev_end,
            p_thresh=float(params["rt_p_thresh"]),
            k_consec=int(params["rt_k_consec"]),
        )
        rt_ms = rt_tokens.float().numpy() * float(token_ms)
        found_np = found.numpy().astype(bool)

        rt_found_total += int(found_np.sum())
        rt_miss_total += int((~found_np).sum())

        for bi in range(B):
            for tr in range(10):
                k = bi*10 + tr
                per_trial_rows.append({
                    "model_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "ckpt": str(ckpt_path),
                    "isi_ms": int(isi_ms),
                    "block_idx": int(BIDX[bi]),
                    "trial_idx": int(tr),
                    "subject_id": SID[bi][tr],
                    "position": int(Y[bi, tr]),
                    "human_rt_ms": float(HRT[bi, tr]),
                    "model_rt_ms": float(rt_ms[bi, tr]),
                    "found": int(found_np[bi, tr]),
                    "trial_type": str(BT[bi][tr]),
                    "true_cls": int(y_cls_np[k]),
                    "pred_cls": int(pred[k]),
                    "p0": float(probs[k, 0]),
                    "p1": float(probs[k, 1]),
                    "p2": float(probs[k, 2]),
                })

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.zeros((0, 3), dtype=np.float32)

    acc = float((y_true == y_pred).mean()) if y_true.size else float("nan")
    f1 = safe_f1_macro(y_true, y_pred) if y_true.size else float("nan")
    auc = safe_auc_ovr(y_true, y_prob, n_classes=3) if y_true.size else float("nan")

    per_trial = pd.DataFrame(per_trial_rows)

    # per trial-type aggregation
    g = per_trial.groupby("trial_type", sort=True)
    per_tt = g.agg(
        n=("trial_type", "size"),
        human_rt_ms=("human_rt_ms", "mean"),
        model_rt_ms=("model_rt_ms", "mean"),
        found_rate=("found", "mean"),
        acc=("pred_cls", lambda x: float(np.mean(x.to_numpy() == per_trial.loc[x.index, "true_cls"].to_numpy()))),
    ).reset_index()

    # correlations across trial types
    def corr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return (float("nan"), float("nan"))
        aa, bb = a[m], b[m]
        pear = float(np.corrcoef(aa, bb)[0, 1])
        ra = aa.argsort().argsort().astype(float)
        rb = bb.argsort().argsort().astype(float)
        spear = float(np.corrcoef(ra, rb)[0, 1])
        return pear, spear

    pear, spear = corr(per_tt["human_rt_ms"].to_numpy(float), per_tt["model_rt_ms"].to_numpy(float))

    run_out = out_dir / run_dir.name
    run_out.mkdir(parents=True, exist_ok=True)
    per_trial.to_csv(run_out / "per_trial.csv", index=False)
    per_tt.to_csv(run_out / "per_trialtype.csv", index=False)

    meta = {
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "params_used": params,
        "isi_ms": int(isi_ms),
        "tone_ms": int(tone_ms),
        "token_ms": int(token_ms),
        "trial_T_tokens": int(trial_T_tokens),
        "n_bins": int(n_bins),
        "add_eos": bool(add_eos),
        "acc": acc,
        "f1_macro": f1,
        "auc_ovr": auc,
        "rt_found": int(rt_found_total),
        "rt_miss": int(rt_miss_total),
        "pearson_trialtype_rt": pear,
        "spearman_trialtype_rt": spear,
        "n_trialtypes": int(per_tt.shape[0]),
        "n_trials": int(per_trial.shape[0]),
    }
    (run_out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, required=True)
    ap.add_argument("--human_csv", type=str, required=True)
    ap.add_argument("--stimuli_dir", type=str, required=True,
                    help="Directory containing input_blocks.pt (and optionally labels_blocks.pt) used for stimulus sequences.")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--isi_filter", type=int, default=700)
    ap.add_argument("--tone_ms", type=int, default=50)
    ap.add_argument("--token_ms", type=int, default=10)

    ap.add_argument("--f_min_hz", type=float, default=1300.0)
    ap.add_argument("--f_max_hz", type=float, default=1700.0)
    ap.add_argument("--n_bins", type=int, default=128)
    ap.add_argument("--add_eos", action="store_true")

    ap.add_argument("--chunk_len", type=int, default=600)
    ap.add_argument("--batch_blocks", type=int, default=64,
                    help="Batch size in BLOCKS (each block contains 10 trials).")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ckpt_prefer_best_isi", action="store_true")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    if not sweep_root.exists():
        raise RuntimeError(f"sweep_root not found: {sweep_root.resolve()}")

    human_csv = Path(args.human_csv)
    if not human_csv.exists():
        raise RuntimeError(f"human_csv not found: {human_csv}")

    stim_dir = Path(args.stimuli_dir)
    xb = stim_dir / "input_blocks.pt"
    if not xb.exists():
        raise RuntimeError(f"input_blocks.pt not found in stimuli_dir: {stim_dir.resolve()}")

    yb = stim_dir / "labels_blocks.pt"
    Y_blocks = None
    if yb.exists():
        Y_blocks = torch.load(yb, map_location="cpu").numpy()

    X_blocks = torch.load(xb, map_location="cpu").numpy().astype(np.float32)  # (B,10,8)
    if X_blocks.ndim != 3 or X_blocks.shape[1:] != (10, 8):
        raise RuntimeError(f"input_blocks.pt must be (B,10,8). got {X_blocks.shape}")

    df_h = pd.read_csv(human_csv)
    df_aligned = align_human_to_stimuli(
        human_df=df_h,
        X_blocks=X_blocks,
        Y_blocks=Y_blocks,
        isi_filter=int(args.isi_filter),
    )

    if "pos_match" in df_aligned.columns:
        mismatch = 1.0 - float(df_aligned["pos_match"].mean())
        print(f"[align] position mismatch rate vs labels_blocks.pt: {mismatch:.4f} (0 is perfect)")
        if mismatch > 0.01:
            print("[align] WARNING: mismatch > 1%. Your human row order may not match stimuli order.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in sweep_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    all_meta = []
    all_tt_rows = []

    for run_dir in run_dirs:
        isi = int(args.isi_filter)

        candidates = []
        if args.ckpt_prefer_best_isi:
            candidates.append(run_dir / f"best_isi{isi}.pt")
        candidates += [run_dir / "best.pt", run_dir / "last.pt", run_dir / f"best_isi{isi}.pt"]

        ckpt_path = None
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            print(f"[skip] no checkpoint in: {run_dir.name}")
            continue

        print(f"[eval] {run_dir.name} ckpt={ckpt_path.name}")

        meta = evaluate_one_run(
            run_dir=run_dir,
            ckpt_path=ckpt_path,
            human_df_aligned=df_aligned,
            out_dir=out_dir,
            isi_ms=int(args.isi_filter),
            tone_ms=int(args.tone_ms),
            token_ms=int(args.token_ms),
            f_min_hz=float(args.f_min_hz),
            f_max_hz=float(args.f_max_hz),
            n_bins=int(args.n_bins),
            add_eos=bool(args.add_eos),
            chunk_len=int(args.chunk_len),
            batch_blocks=int(args.batch_blocks),
            device=str(args.device),
            seed=int(args.seed),
        )
        all_meta.append(meta)

        per_tt_path = out_dir / run_dir.name / "per_trialtype.csv"
        per_tt = pd.read_csv(per_tt_path)
        per_tt["model_id"] = run_dir.name
        per_tt["sigma_other_noise"] = float(meta["params_used"]["sigma_other_noise"])
        per_tt["p_other_noise"] = float(meta["params_used"]["p_other_noise"])
        per_tt["sigma_silence_noise"] = float(meta["params_used"]["sigma_silence_noise"])
        per_tt["rt_p_thresh"] = float(meta["params_used"]["rt_p_thresh"])
        per_tt["rt_k_consec"] = int(meta["params_used"]["rt_k_consec"])
        all_tt_rows.append(per_tt)

    if len(all_meta) == 0:
        raise RuntimeError("No runs evaluated.")

    summary = pd.DataFrame(all_meta)
    summary.to_csv(out_dir / "all_runs_summary.csv", index=False)

    if len(all_tt_rows) > 0:
        merged = pd.concat(all_tt_rows, ignore_index=True)
        merged.to_csv(out_dir / "all_trialtype_merged.csv", index=False)

    print("[done] saved:")
    print(" -", (out_dir / "all_runs_summary.csv").resolve())
    if (out_dir / "all_trialtype_merged.csv").exists():
        print(" -", (out_dir / "all_trialtype_merged.csv").resolve())


if __name__ == "__main__":
    main()