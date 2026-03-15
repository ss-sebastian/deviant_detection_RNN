import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import re

# =========================
# 0) PATHS (CHANGE THESE)
# =========================
CKPT_PATH   = Path("ckpt_base_warmstart_resume_sweep_5x5_to700/i=6__sig_other=0.01__p_other=1__sig_sil=0__rtp=0.5__rtk=1/best_isi700.pt")
GM_DATA_DIR = Path("gm_data_erb_1300_1700")  # the directory with input_blocks.pt/labels_blocks.pt
HUMAN_CSV   = Path("/Users/seb/Desktop/bcbl/msc_thesis/logfiles/human_trial.csv")  # must have: subject_id, isi_ms, position, rt_ms

DEVICE = torch.device("cpu")  # set "mps" if you want, but cpu is safer for debugging

# =========================
# 1) IMPORT YOUR MODEL/DATASET CODE
#    (assumes this notebook is in the same folder as train_online.py + model.py)
# =========================
from model import PredictiveGRU, ModelConfig
from train_online import (
    OnlineRenderDataset,
    build_loaders_for_isi,
    split_indices,
    evaluate,
)

# =========================
# 2) LOAD CKPT + AUTO-PARAMS
# =========================
ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
cfg_dict = ckpt.get("cfg", None)
args_dict = ckpt.get("args", {})

print("Loaded ckpt:", CKPT_PATH)
print("ckpt keys:", list(ckpt.keys()))
print("ckpt stage/isi:", ckpt.get("stage", None), ckpt.get("isi_ms", None))
print("cfg:", cfg_dict)

# Pull defaults from ckpt args if available; fallback to common defaults
tone_ms  = int(args_dict.get("tone_ms", 50))
token_ms = int(args_dict.get("token_ms", 10))
ramp_ms  = int(args_dict.get("ramp_ms", 5))
n_bins   = int(args_dict.get("n_bins", 128))
f_min_hz = float(args_dict.get("f_min_hz", 1300.0))
f_max_hz = float(args_dict.get("f_max_hz", 1700.0))
add_eos  = bool(args_dict.get("add_eos", False))

chunk_len = int(args_dict.get("chunk_len", 600))
lambda_token = float(args_dict.get("lambda_token", 0.5))
token_loss_mode = str(args_dict.get("token_loss_mode", "exp"))
token_tau = float(args_dict.get("token_tau", 50.0))
token_w_min = float(args_dict.get("token_w_min", 0.05))
tok_window_ms = int(args_dict.get("tok_window_ms", 200))
tok_start_offset_ms = int(args_dict.get("tok_start_offset_ms", 0))

# IMPORTANT: the evaluation RT criterion is part of the sweep; many people accidentally override it at eval-time.
rt_p_thresh = float(args_dict.get("rt_p_thresh", 0.6))
rt_k_consec = int(args_dict.get("rt_k_consec", 2))

# Noise used in dataset rendering (must match the run if you want comparable behavior)
sigma_other_noise = float(args_dict.get("sigma_other_noise", 0.0))
p_other_noise = float(args_dict.get("p_other_noise", 1.0))
sigma_silence_noise = float(args_dict.get("sigma_silence_noise", 0.0))

print("\n=== Params inferred from checkpoint ===")
print(dict(
    tone_ms=tone_ms, token_ms=token_ms, ramp_ms=ramp_ms,
    n_bins=n_bins, f_min_hz=f_min_hz, f_max_hz=f_max_hz, add_eos=add_eos,
    chunk_len=chunk_len, lambda_token=lambda_token,
    tok_window_ms=tok_window_ms, tok_start_offset_ms=tok_start_offset_ms,
    token_loss_mode=token_loss_mode, token_tau=token_tau, token_w_min=token_w_min,
    rt_p_thresh=rt_p_thresh, rt_k_consec=rt_k_consec,
    sigma_other_noise=sigma_other_noise, p_other_noise=p_other_noise, sigma_silence_noise=sigma_silence_noise,
))

# =========================
# 3) REBUILD MODEL + LOAD WEIGHTS
# =========================
cfg = ModelConfig(
    input_dim=int(cfg_dict["input_dim"]),
    hidden_dim=int(cfg_dict["hidden_dim"]),
    num_layers=int(cfg_dict["num_layers"]),
    dropout=float(cfg_dict["dropout"]),
    layer_norm=bool(cfg_dict.get("layer_norm", False)),
)
model = PredictiveGRU(cfg).to(DEVICE)
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()

# =========================
# 4) SANITY CHECK A: EVAL ON GM_DATA @ isi=700
# =========================
# Build same split logic as training
base_ds = OnlineRenderDataset(
    data_dir=GM_DATA_DIR,
    seed=int(args_dict.get("seed", 42)),
    tone_ms=tone_ms,
    isi_ms=int(args_dict.get("isi_schedule", [0])[0]) if isinstance(args_dict.get("isi_schedule", [0]), list) else 0,
    ramp_ms=ramp_ms,
    token_ms=token_ms,
    f_min_hz=f_min_hz,
    f_max_hz=f_max_hz,
    n_bins=n_bins,
    add_eos=add_eos,
    sigma_other_noise=sigma_other_noise,
    p_other_noise=p_other_noise,
    sigma_silence_noise=sigma_silence_noise,
    quiet=True,
)

# respect max_blocks used during training if present
max_blocks = int(args_dict.get("max_blocks", 0) or 0)
if max_blocks > 0:
    m = min(max_blocks, int(base_ds.B))
    base_ds.X = base_ds.X[:m]
    base_ds.Y = base_ds.Y[:m]
    base_ds.B = m

train_idx, val_idx = split_indices(len(base_ds), float(args_dict.get("val_split", 0.1)), int(args_dict.get("seed", 42)))

# Now build loaders specifically at isi=700
isi_eval = 700
ds, _, val_loader = build_loaders_for_isi(
    data_dir=GM_DATA_DIR,
    seed=int(args_dict.get("seed", 42)),
    tone_ms=tone_ms,
    isi_ms=isi_eval,
    ramp_ms=ramp_ms,
    token_ms=token_ms,
    f_min_hz=f_min_hz,
    f_max_hz=f_max_hz,
    n_bins=n_bins,
    add_eos=add_eos,
    sigma_other_noise=sigma_other_noise,
    p_other_noise=p_other_noise,
    sigma_silence_noise=sigma_silence_noise,
    train_idx=train_idx,
    val_idx=val_idx,
    batch_size=128,
    num_workers=0,
    device=DEVICE,
    assert_labels=False,
)

va = evaluate(
    model=model,
    loader=val_loader,
    device=DEVICE,
    chunk_len=chunk_len,
    lambda_token=lambda_token,
    trial_T_tokens=int(ds.trial_T_tokens),
    tone_T=int(ds.tone_T),
    isi_T=int(ds.isi_T),
    token_loss_mode=token_loss_mode,
    token_tau=token_tau,
    token_w_min=token_w_min,
    token_ms=int(ds.token_ms),
    tok_window_ms=tok_window_ms,
    tok_start_offset_ms=tok_start_offset_ms,
    rt_p_thresh=rt_p_thresh,
    rt_k_consec=rt_k_consec,
    debug_labels=False,
    epoch_global=int(ckpt.get("epoch_global", 0)),
)

print("\n=== Sanity A: eval on GM_DATA val split @ isi=700 ===")
print(va)

# If this acc is low (~0.33), the "high train acc" you saw earlier might have been at a different ISI or on train not val,
# or the checkpoint isn't the one you think it is.

# =========================
# 5) SANITY CHECK B: HUMAN trialtype RT correlation
#    (Requires that your human trials correspond to the same (std,dev,pos) conditions as GM stimuli OR at least share "position" factor)
# =========================
dfh = pd.read_csv(HUMAN_CSV)
need_cols = {"position", "rt_ms"}
if not need_cols.issubset(dfh.columns):
    raise RuntimeError(f"human_csv missing required columns: {need_cols - set(dfh.columns)}")

# Filter to isi=700 if present
if "isi_ms" in dfh.columns:
    dfh = dfh[dfh["isi_ms"] == isi_eval].copy()

dfh["position"] = pd.to_numeric(dfh["position"], errors="coerce")
dfh["rt_ms"] = pd.to_numeric(dfh["rt_ms"], errors="coerce")
dfh = dfh[dfh["position"].isin([4,5,6]) & dfh["rt_ms"].notna()].copy()

# For a minimal sanity check when you don't have frequencies:
# compare mean RT by position (4/5/6) between model and human.
# We'll get model RT-by-position from the GM_DATA eval run by re-running and storing per-trial RTs quickly.

# --- quick function to extract per-trial RTs from model on val_loader (using same RT criterion) ---
from train_online import _run_block_through_tbptt, infer_end_indices_from_T, deviant_end_token_in_trial, compute_rt_from_logits, labels_to_class_index

@torch.no_grad()
def collect_model_trial_rts(val_loader, ds):
    rows = []
    for x, y_pos_456 in val_loader:
        x = x.to(DEVICE)
        y_pos_456 = y_pos_456.long().to(DEVICE)
        B, T, D = x.shape
        end_idx = infer_end_indices_from_T(int(T), trials_per_block=10).to(DEVICE)

        _, _, logits_all = _run_block_through_tbptt(
            model=model,
            x=x,
            y_pos_456=y_pos_456,
            end_idx=end_idx,
            chunk_len=chunk_len,
            token_ce=nn.CrossEntropyLoss(reduction="mean"),
            trial_T_tokens=int(ds.trial_T_tokens),
            tone_T=int(ds.tone_T),
            isi_T=int(ds.isi_T),
            token_ms=int(ds.token_ms),
            tok_window_ms=tok_window_ms,
            tok_start_offset_ms=tok_start_offset_ms,
            token_loss_mode=token_loss_mode,
            token_tau=token_tau,
            token_w_min=token_w_min,
            return_full_logits=True,
        )
        if logits_all is None:
            continue

        logits_trial = logits_all.view(B, 10, int(ds.trial_T_tokens), 3).detach().cpu()
        y_cpu = y_pos_456.detach().cpu().long()
        y_cls_cpu = (y_cpu - 4).long()
        dev_end_cpu = deviant_end_token_in_trial(y_pos_456=y_cpu, tone_T=int(ds.tone_T), isi_T=int(ds.isi_T))

        rt_tokens, found = compute_rt_from_logits(
            logits=logits_trial,
            y_cls=y_cls_cpu,
            dev_end=dev_end_cpu,
            p_thresh=float(rt_p_thresh),
            k_consec=int(rt_k_consec),
        )
        rt_ms = rt_tokens.float() * float(ds.token_ms)

        for b in range(B):
            for tr in range(10):
                rows.append({
                    "position": int(y_cpu[b, tr].item()),
                    "rt_ms": float(rt_ms[b, tr].item()),
                    "found": int(found[b, tr].item()),
                })
    return pd.DataFrame(rows)

dfm = collect_model_trial_rts(val_loader, ds)
# keep only found trials
dfm_found = dfm[dfm["found"] == 1].copy()

human_pos_mean = dfh.groupby("position")["rt_ms"].mean()
model_pos_mean = dfm_found.groupby("position")["rt_ms"].mean()

print("\n=== Sanity B: RT by deviant position (human vs model) ===")
print("Human mean RT (ms) by position:\n", human_pos_mean)
print("Model  mean RT (ms) by position (found only):\n", model_pos_mean)

# correlation across positions (only 3 points; not a real statistic, but catches gross bugs)
common_pos = sorted(set(human_pos_mean.index) & set(model_pos_mean.index))
if len(common_pos) >= 2:
    x = human_pos_mean.loc[common_pos].to_numpy()
    y = model_pos_mean.loc[common_pos].to_numpy()
    r = np.corrcoef(x, y)[0,1]
    print(f"\nPos-level Pearson r (VERY rough, n={len(common_pos)}): {r:.3f}")
else:
    print("\nNot enough overlapping positions with found RTs to correlate.")