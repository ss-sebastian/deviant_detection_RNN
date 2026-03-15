#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_online.py

Evaluate a trained PredictiveGRU checkpoint on:
  A) Hz blocks: input_blocks.pt (B,10,8) + labels_blocks.pt (B,10)
  B) Single block: input_tensor.pt (1,10,8) + labels_tensor.pt (1,10)
  C) Already tokenized sequences: input_tokens.pt (B,T,D) + labels_*.pt (flexible)
     (optional; kept for compatibility)

Main goal (your current use case):
  - Take Hz blocks (B,10,8)
  - Render into 10ms token one-hot (B, T_tokens, 128)
  - Run model, collect logits + hidden states
  - Compute end-of-trial metrics + RT-from-logits
  - Save summary + per-item outputs

Expected label format:
  - y in {4,5,6}, meaning deviant position within trial (1-indexed)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch

# Your model definitions (must match train_online.py)
from model import PredictiveGRU, ModelConfig


# -------------------------
# Basics
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
# Label helpers
# -------------------------
def labels_to_class_index(y_456: torch.Tensor) -> torch.Tensor:
    """
    y_456: (...,) values in {4,5,6} (1-indexed deviant position)
    map to {0,1,2}
    """
    return (y_456 - 4).long()


def normalize_labels_to_456(Y: torch.Tensor) -> torch.Tensor:
    """
    Accept {4,5,6} or {0,1,2}. Return {4,5,6}.
    """
    uniq = set(torch.unique(Y).tolist())
    if uniq.issubset({4, 5, 6}):
        return Y.long()
    if uniq.issubset({0, 1, 2}):
        return (Y + 4).long()
    raise ValueError(f"Unsupported label set: {sorted(list(uniq))}, expected subset of {{4,5,6}} or {{0,1,2}}")


# -------------------------
# IO: load data
# -------------------------
def load_blocks_or_single(in_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Supports:
      - input_blocks.pt (B,10,8), labels_blocks.pt (B,10)
      - input_tensor.pt (1,10,8), labels_tensor.pt (1,10)
      - generic fallback: input_tokens.pt (B,T,D) + labels_tokens.pt
    Returns X, Y, layout
    """
    xb = in_dir / "input_blocks.pt"
    yb = in_dir / "labels_blocks.pt"
    xs = in_dir / "input_tensor.pt"
    ys = in_dir / "labels_tensor.pt"

    xt = in_dir / "input_tokens.pt"
    yt = in_dir / "labels_tokens.pt"

    if xb.exists() and yb.exists():
        X = torch.load(xb, map_location="cpu")
        Y = torch.load(yb, map_location="cpu")
        return X, Y, "blocks"

    if xs.exists() and ys.exists():
        X = torch.load(xs, map_location="cpu")
        Y = torch.load(ys, map_location="cpu")
        return X, Y, "single"

    if xt.exists() and yt.exists():
        X = torch.load(xt, map_location="cpu")
        Y = torch.load(yt, map_location="cpu")
        return X, Y, "tokens"

    raise FileNotFoundError(
        "Need one of:\n"
        "  - input_blocks.pt + labels_blocks.pt\n"
        "  - input_tensor.pt + labels_tensor.pt\n"
        "  - input_tokens.pt + labels_tokens.pt\n"
        f"in --data_dir={in_dir}"
    )


# -------------------------
# ERB mapping (must match train_online.py)
# -------------------------
def hz_to_erb_rate_np(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * f_hz)


def make_erb_edges(f_min_hz: float, f_max_hz: float, n_bins: int) -> np.ndarray:
    erb_min = float(hz_to_erb_rate_np(np.array([f_min_hz], dtype=np.float32))[0])
    erb_max = float(hz_to_erb_rate_np(np.array([f_max_hz], dtype=np.float32))[0])
    edges = np.linspace(erb_min, erb_max, int(n_bins) + 1, dtype=np.float32)
    return edges


def freq_to_bin_erb(f_hz: float, edges_erb: np.ndarray) -> int:
    erb = float(hz_to_erb_rate_np(np.array([f_hz], dtype=np.float32))[0])
    j = int(np.searchsorted(edges_erb, erb, side="right") - 1)
    j = max(0, min(j, int(edges_erb.shape[0] - 2)))
    return j


def render_blocks_hz_to_token_onehot(
    X_hz: torch.Tensor,   # (B,10,8) float Hz
    tone_ms: int,
    isi_ms: int,
    token_ms: int,
    f_min_hz: float,
    f_max_hz: float,
    n_bins: int,
    add_eos: bool = False,
    # --- noise (match train_online.py) ---
    sigma_other_noise: float = 0.0,
    p_other_noise: float = 0.0,
    sigma_silence_noise: float = 0.0,
    seed: int = 42,
) -> torch.Tensor:
    """
    Convert Hz blocks (B,10,8) into token one-hot sequence (B, T_tokens, D),
    with optional noise injection matching OnlineRenderDataset in train_online.py.

    Noise behavior (same spirit as train_online.py):
      - During tone tokens: set correct bin=1.0. With prob p_other_noise, add Gaussian
        noise N(0, sigma_other_noise) to other bins (target bin noise forced to 0).
      - During ISI tokens: if sigma_silence_noise>0, fill all bins with Gaussian noise
        N(0, sigma_silence_noise); else keep zeros.
      - EOS: if add_eos, last token of each trial has eos dim = 1.
    """
    if X_hz.ndim != 3 or tuple(X_hz.shape[1:]) != (10, 8):
        raise ValueError(f"Expected X_hz (B,10,8), got {tuple(X_hz.shape)}")

    if token_ms <= 0:
        raise ValueError("token_ms must be > 0")
    if (tone_ms % token_ms) != 0:
        raise ValueError(f"tone_ms={tone_ms} not divisible by token_ms={token_ms}")
    if (isi_ms % token_ms) != 0:
        raise ValueError(f"isi_ms={isi_ms} not divisible by token_ms={token_ms}")

    tone_T = tone_ms // token_ms
    isi_T = isi_ms // token_ms
    trial_T_tokens = 8 * tone_T + 7 * isi_T
    T_tokens = 10 * trial_T_tokens

    edges_erb = make_erb_edges(float(f_min_hz), float(f_max_hz), int(n_bins))
    eos_dim = 1 if add_eos else 0
    D = int(n_bins) + eos_dim

    X_hz = X_hz.detach().cpu().float()
    B = int(X_hz.shape[0])

    out = torch.zeros((B, T_tokens, D), dtype=torch.float32)

    # numpy RNG for speed + reproducibility (and matches your train style)
    base_rng = np.random.default_rng(int(seed))

    for b in range(B):
        t_abs = 0
        for tr in range(10):
            freqs8 = X_hz[b, tr]  # (8,)

            # make per-(b,tr) rng to avoid order dependence
            rng = np.random.default_rng(int(seed) + b * 1000 + tr * 17)

            for i in range(8):
                bin_i = freq_to_bin_erb(float(freqs8[i].item()), edges_erb)

                # tone tokens
                for _ in range(tone_T):
                    out[b, t_abs, bin_i] = 1.0

                    if (sigma_other_noise > 0.0) and (p_other_noise > 0.0):
                        if float(rng.random()) < float(p_other_noise):
                            noise = rng.normal(
                                0.0, float(sigma_other_noise), size=(int(n_bins),)
                            ).astype(np.float32)
                            noise[bin_i] = 0.0  # do not corrupt the true bin
                            out[b, t_abs, :int(n_bins)] += torch.from_numpy(noise)

                    t_abs += 1

                # ISI tokens after tones 1..7
                if i < 7:
                    if (sigma_silence_noise > 0.0) and (isi_T > 0):
                        for _ in range(isi_T):
                            noise = rng.normal(
                                0.0, float(sigma_silence_noise), size=(int(n_bins),)
                            ).astype(np.float32)
                            out[b, t_abs, :int(n_bins)] = torch.from_numpy(noise)
                            t_abs += 1
                    else:
                        t_abs += isi_T  # keep zeros

            # EOS marker at last token of trial (same as train)
            if add_eos:
                out[b, t_abs - 1, int(n_bins)] = 1.0

        if t_abs != T_tokens:
            raise RuntimeError(f"Token fill mismatch: ended at {t_abs}, expected {T_tokens}")

    return out



# -------------------------
# Trial indexing + RT
# -------------------------
def infer_end_indices_from_T(T: int, trials_per_block: int = 10) -> torch.Tensor:
    if T % trials_per_block != 0:
        raise ValueError(f"Cannot infer trial length: T={T} not divisible by {trials_per_block}")
    trial_T = T // trials_per_block
    return torch.tensor([(i + 1) * trial_T - 1 for i in range(trials_per_block)], dtype=torch.long)


def deviant_end_token_in_trial(
    y_pos_456: torch.Tensor,   # (B,10) values in {4,5,6}
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    """
    dev_pos 4/5/6 -> dev_idx 3/4/5 (0-indexed within 8 tones)
    step = tone_T + isi_T
    dev_end = dev_idx*step + (tone_T - 1)
    """
    dev_idx = (y_pos_456 - 1).long()  # {4,5,6}->{3,4,5}
    step = int(tone_T + isi_T)
    dev_end = dev_idx * step + int(tone_T) - 1
    return dev_end


def compute_rt_from_logits(
    logits: torch.Tensor,          # (B,10,trial_T_tokens,3)
    y_cls: torch.Tensor,           # (B,10) in {0,1,2}
    dev_end: torch.Tensor,         # (B,10) dev_end token idx
    p_thresh: float,
    k_consec: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RT definition:
      start counting at t0 = dev_end + 1
      find earliest time where:
        - prediction is correct AND p(correct_class) >= p_thresh
        - holds for k_consec consecutive tokens
      RT tokens = first_t - t0
    If not found => rt=-1, found=False
    """
    B, N, Tt, C = logits.shape
    assert N == 10 and C == 3

    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)

    y = y_cls.unsqueeze(-1).expand(B, N, Tt)
    correct = (pred == y)

    py = probs.gather(
        dim=-1,
        index=y_cls.view(B, N, 1, 1).expand(B, N, Tt, 1)
    ).squeeze(-1)

    confident = py >= float(p_thresh)
    ok = correct & confident

    rt = torch.full((B, N), -1, dtype=torch.long, device=logits.device)
    found = torch.zeros((B, N), dtype=torch.bool, device=logits.device)

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
# Metrics (no sklearn dependency required, but uses sklearn if available for AUC)
# -------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def macro_auc_ovr(y_true: np.ndarray, prob: np.ndarray, n_classes: int = 3) -> float:
    """
    One-vs-rest macro AUC. Uses sklearn if present; else returns NaN.
    prob: (N,3)
    """
    try:
        from sklearn.metrics import roc_auc_score
        y_true = y_true.astype(np.int64)
        return float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


# -------------------------
# Forward runner: chunked, collect logits + hidden
# -------------------------
@torch.no_grad()
def run_model_collect(
    model: PredictiveGRU,
    x: torch.Tensor,              # (B,T,D)
    chunk_len: int,
    return_hidden: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      logits_all: (B,T,3)
      hidden_all: (B,T,H) if return_hidden else None
    """
    B, T, D = x.shape
    h0 = None
    logits_chunks = []
    hidden_chunks = [] if return_hidden else None

    for s in range(0, T, int(chunk_len)):
        e = min(s + int(chunk_len), T)
        x_in = x[:, s:e, :]
        h_seq, h0 = model.forward_chunk(x_in, h0=h0)    # (B,L,H)
        logits = model.classify_tokens(h_seq)           # (B,L,3)

        logits_chunks.append(logits.detach())
        if return_hidden:
            hidden_chunks.append(h_seq.detach())

        h0 = h0.detach()

    logits_all = torch.cat(logits_chunks, dim=1)  # (B,T,3)
    hidden_all = torch.cat(hidden_chunks, dim=1) if return_hidden else None

    return logits_all, hidden_all


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--chunk_len", type=int, default=512)

    # timing (needed for rendering + dev_end + RT-to-ms)
    p.add_argument("--tone_ms", type=int, default=50)
    p.add_argument("--isi_ms", type=int, default=0)
    p.add_argument("--token_ms", type=int, default=10)

    # ERB one-hot space (MUST match training)
    p.add_argument("--f_min_hz", type=float, default=1300.0)
    p.add_argument("--f_max_hz", type=float, default=1700.0)
    p.add_argument("--n_bins", type=int, default=128)
    p.add_argument("--add_eos", action="store_true")

    # RT criterion
    p.add_argument("--rt_p_thresh", type=float, default=0.7)
    p.add_argument("--rt_k_consec", type=int, default=3)

    # saving
    p.add_argument("--save_hidden", action="store_true", help="save hidden states (can be large)")
    p.add_argument("--max_save_samples", type=int, default=128, help="cap samples for hidden saving")
    p.add_argument("--hidden_fp16", action="store_true", help="save hidden as float16 to reduce size")
    
    # noise (match train_online.py renderer)
    p.add_argument("--sigma_other_noise", type=float, default=0.0)
    p.add_argument("--p_other_noise", type=float, default=0.0)
    p.add_argument("--sigma_silence_noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)


    args = p.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[device] using: {device}")

    # -------- load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "cfg" in ckpt:
        cfg_dict = ckpt["cfg"]
        cfg = ModelConfig(**cfg_dict) if isinstance(cfg_dict, dict) else ModelConfig(**asdict(cfg_dict))
    elif "model_cfg" in ckpt:
        cfg = ModelConfig(**ckpt["model_cfg"])
    else:
        raise ValueError("Checkpoint must contain 'cfg' (saved by train_online.py).")

    model = PredictiveGRU(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    print(f"[ckpt] loaded: {Path(args.ckpt).resolve()}")
    print(f"[model] input_dim={cfg.input_dim} hidden_dim={cfg.hidden_dim} num_layers={cfg.num_layers}")

    # -------- load data
    X, Y, layout = load_blocks_or_single(data_dir)

    # Make sure labels are torch tensors
    if not torch.is_tensor(X):
        raise ValueError(f"X must be torch.Tensor, got {type(X)}")
    if not torch.is_tensor(Y):
        raise ValueError(f"Y must be torch.Tensor, got {type(Y)}")

    # Normalize labels to {4,5,6}
    Y = normalize_labels_to_456(Y)

    # If Hz blocks, render to tokens
    if X.ndim == 3 and tuple(X.shape[1:]) == (10, 8):
        print("[data] Detected Hz blocks (B,10,8). Rendering to 10ms token one-hot...")
        X_tok = render_blocks_hz_to_token_onehot(
            X_hz=X,
            tone_ms=int(args.tone_ms),
            isi_ms=int(args.isi_ms),
            token_ms=int(args.token_ms),
            f_min_hz=float(args.f_min_hz),
            f_max_hz=float(args.f_max_hz),
            n_bins=int(args.n_bins),
            add_eos=bool(args.add_eos),
            sigma_other_noise=float(args.sigma_other_noise),
            p_other_noise=float(args.p_other_noise),
            sigma_silence_noise=float(args.sigma_silence_noise),
            seed=int(args.seed),
        )

        layout = "tokenized_from_hz"
    else:
        # Already tokenized: expect (B,T,D)
        if X.ndim != 3:
            raise ValueError(f"Unsupported X shape: {tuple(X.shape)}. Expected (B,10,8) or (B,T,D).")
        X_tok = X.float()

    if X_tok.ndim != 3:
        raise ValueError(f"Tokenized X must be (B,T,D). Got {tuple(X_tok.shape)}")

    B, T, D = X_tok.shape

    # Y handling: want shape (B,10) for blocks
    if Y.ndim == 2 and Y.shape[0] == B and Y.shape[1] == 10:
        Yb = Y
    elif Y.ndim == 1 and Y.numel() == B:
        # interpret as one label per sample -> expand to 10
        Yb = Y.view(B, 1).expand(B, 10).contiguous()
    elif Y.ndim == 1 and Y.numel() == B * 10:
        Yb = Y.view(B, 10)
    else:
        raise ValueError(f"Unsupported Y shape {tuple(Y.shape)} for B={B}. Expected (B,10) or (B,) or (B*10,)")

    Yb = Yb.long()

    # sanity timing divisibility
    if args.tone_ms % args.token_ms != 0:
        raise ValueError(f"tone_ms={args.tone_ms} not divisible by token_ms={args.token_ms}")
    if args.isi_ms % args.token_ms != 0:
        raise ValueError(f"isi_ms={args.isi_ms} not divisible by token_ms={args.token_ms}")

    tone_T = args.tone_ms // args.token_ms
    isi_T = args.isi_ms // args.token_ms

    # Must be exactly 10 trials per block for RT logic
    if T % 10 != 0:
        raise ValueError(f"T must be divisible by 10 to form 10 trials per block. Got T={T}")
    trial_T_tokens = T // 10

    print(
        f"[data] layout={layout} X_tok={tuple(X_tok.shape)} Y={tuple(Yb.shape)} "
        f"trial_T_tokens={trial_T_tokens} tone_T={tone_T} isi_T={isi_T} "
        f"uniqY={sorted(torch.unique(Yb).tolist())}"
    )

    # dimension check against model
    if int(D) != int(cfg.input_dim):
        raise ValueError(f"Input dim mismatch: X has D={D}, but model expects input_dim={cfg.input_dim}")

    # -------- run model
    Xd = X_tok.to(device, non_blocking=True)

    logits_all, hidden_all = run_model_collect(
        model=model,
        x=Xd,
        chunk_len=int(args.chunk_len),
        return_hidden=bool(args.save_hidden),
    )

    # reshape logits to (B,10,trial_T_tokens,3)
    logits_trial = logits_all.view(B, 10, int(trial_T_tokens), 3)

    # y in class indices (B,10)
    y_cls = labels_to_class_index(Yb.to(device))

    # end-of-trial prediction: last token of each trial
    end_idx = infer_end_indices_from_T(int(T), trials_per_block=10).to(device)  # (10,)
    end_logits = logits_all.index_select(dim=1, index=end_idx)   # (B,10,3)
    end_prob = torch.softmax(end_logits, dim=-1)                # (B,10,3)
    end_pred = end_prob.argmax(dim=-1)                          # (B,10)

    # RT
    dev_end = deviant_end_token_in_trial(Yb.to(device), tone_T=int(tone_T), isi_T=int(isi_T))  # (B,10)
    rt_tokens, found = compute_rt_from_logits(
        logits=logits_trial,
        y_cls=y_cls,
        dev_end=dev_end,
        p_thresh=float(args.rt_p_thresh),
        k_consec=int(args.rt_k_consec),
    )

    # -------- aggregate metrics (flatten B*10 trials)
    y_true = y_cls.detach().cpu().numpy().reshape(-1)      # (B*10,)
    y_hat = end_pred.detach().cpu().numpy().reshape(-1)    # (B*10,)
    prob = end_prob.detach().cpu().numpy().reshape(-1, 3)  # (B*10,3)

    acc = float((y_true == y_hat).mean())
    f1 = macro_f1(y_true, y_hat, n_classes=3)
    auc = macro_auc_ovr(y_true, prob, n_classes=3)

    rt_tok_np = rt_tokens.detach().cpu().numpy().reshape(-1)
    found_np = found.detach().cpu().numpy().reshape(-1).astype(bool)
    rt_ms_np = rt_tok_np.astype(np.float32) * float(args.token_ms)

    # Per-label RT (only for found)
    per_label_rt: Dict[str, Any] = {}
    for c in range(3):
        m = (y_true == c) & (found_np) & (rt_tok_np >= 0)
        if np.any(m):
            vals = rt_ms_np[m]
            per_label_rt[str(c)] = {
                "n": int(vals.size),
                "mean_ms": float(np.mean(vals)),
                "median_ms": float(np.median(vals)),
                "p25_ms": float(np.percentile(vals, 25)),
                "p75_ms": float(np.percentile(vals, 75)),
            }
        else:
            per_label_rt[str(c)] = {"n": 0, "mean_ms": float("nan"), "median_ms": float("nan")}

    summary = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "layout": layout,
        "B": int(B),
        "T": int(T),
        "D": int(D),
        "trial_T_tokens": int(trial_T_tokens),
        "tone_ms": int(args.tone_ms),
        "isi_ms": int(args.isi_ms),
        "token_ms": int(args.token_ms),
        "erb": {
            "f_min_hz": float(args.f_min_hz),
            "f_max_hz": float(args.f_max_hz),
            "n_bins": int(args.n_bins),
            "add_eos": bool(args.add_eos),
        },
        "rt_p_thresh": float(args.rt_p_thresh),
        "rt_k_consec": int(args.rt_k_consec),
        "accuracy": acc,
        "macro_f1": f1,
        "macro_auc_ovr": auc,
        "rt_found": int(found_np.sum()),
        "rt_miss": int((~found_np).sum()),
        "per_label_rt_ms": per_label_rt,
    }

    print("[summary]")
    for k in ["accuracy", "macro_f1", "macro_auc_ovr", "rt_found", "rt_miss"]:
        print(f"  {k}: {summary[k]}")

    # -------- save outputs
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # per-item table (B*10 rows)
    out = {
        "y_true": y_true.astype(np.int64),
        "y_pred": y_hat.astype(np.int64),
        "p0": prob[:, 0].astype(np.float32),
        "p1": prob[:, 1].astype(np.float32),
        "p2": prob[:, 2].astype(np.float32),
        "rt_tokens": rt_tok_np.astype(np.int64),
        "rt_ms": rt_ms_np.astype(np.float32),
        "rt_found": found_np.astype(np.bool_),
    }
    np.savez_compressed(save_dir / "per_item.npz", **out)

    # hidden states (optional)
    if args.save_hidden:
        if hidden_all is None:
            print("[warn] --save_hidden set but hidden_all is None (unexpected).")
        else:
            n_save = int(min(B, max(0, args.max_save_samples)))
            if n_save > 0:
                h_save = hidden_all[:n_save].detach().cpu()  # (n_save,T,H)
                if args.hidden_fp16:
                    h_save = h_save.to(torch.float16)
                torch.save(
                    {
                        "hidden": h_save,
                        "B_saved": n_save,
                        "T": int(T),
                        "H": int(h_save.shape[-1]),
                        "note": "hidden states for first n_save samples; increase --max_save_samples if needed",
                    },
                    save_dir / "hidden_states.pt",
                )
                print(f"[save] hidden_states.pt (samples={n_save}, T={T}, fp16={bool(args.hidden_fp16)})")
            else:
                print("[save] --save_hidden set but max_save_samples<=0; skip saving hidden")

    print("[save] wrote:")
    print("  - summary.json")
    print("  - per_item.npz")
    if args.save_hidden:
        print("  - hidden_states.pt (capped)")


if __name__ == "__main__":
    main()
