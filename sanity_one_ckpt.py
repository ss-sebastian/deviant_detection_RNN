import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# =========================
# 0) PATHS (CHANGE THESE)
# =========================
CKPT_PATH   = Path("ckpt_base_warmstart_resume_sweep_5x5_to700/i=6__sig_other=0.01__p_other=1__sig_sil=0__rtp=0.5__rtk=1/best_isi700.pt")
GM_DATA_DIR = Path("gm_data_erb_1300_1700")  # must contain input_blocks.pt + labels_blocks.pt
HUMAN_CSV   = Path("/Users/seb/Desktop/bcbl/msc_thesis/logfiles/human_trial.csv")  # subject_id, isi_ms(optional), position, rt_ms

DEVICE = torch.device("cpu")  # cpu is safest for debugging

# =========================
# 1) IMPORT YOUR MODEL/DATASET CODE
# =========================
from model import PredictiveGRU, ModelConfig
from train_online import (
    OnlineRenderDataset,
    build_loaders_for_isi,
    split_indices,
    evaluate,
    _run_block_through_tbptt,
    infer_end_indices_from_T,
    deviant_end_token_in_trial,
    compute_rt_from_logits,
)

# =========================
# helpers
# =========================
def _parse_isi_schedule(x):
    """
    ckpt["args"]["isi_schedule"] might be:
      - list[int]
      - string like "0,50,300,700"
    """
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str):
        parts = [p.strip() for p in x.replace(" ", "").split(",") if p.strip() != ""]
        return [int(float(p)) for p in parts]
    return []

def _pick_params_from_ckpt(ckpt):
    """
    Priority:
      1) ckpt["sweep"] (if exists): this stores the *true run condition*.
      2) ckpt["args"] fallback.
    """
    args_dict = ckpt.get("args", {}) or {}
    sweep_dict = ckpt.get("sweep", {}) or {}

    # Some of your ckpts store sweep params under sweep itself,
    # but also store args.*; we trust sweep first when present.
    def get(k, default=None):
        if k in sweep_dict:
            return sweep_dict[k]
        return args_dict.get(k, default)

    # core timing/space
    tone_ms  = int(get("tone_ms", 50))
    token_ms = int(get("token_ms", 10))
    ramp_ms  = int(get("ramp_ms", 5))
    n_bins   = int(get("n_bins", 128))
    f_min_hz = float(get("f_min_hz", 1300.0))
    f_max_hz = float(get("f_max_hz", 1700.0))
    add_eos  = bool(get("add_eos", False))

    # tbptt/loss params
    chunk_len = int(get("chunk_len", 600))
    lambda_token = float(get("lambda_token", 0.5))
    token_loss_mode = str(get("token_loss_mode", "exp"))
    token_tau = float(get("token_tau", 50.0))
    token_w_min = float(get("token_w_min", 0.05))
    tok_window_ms = int(get("tok_window_ms", 200))
    tok_start_offset_ms = int(get("tok_start_offset_ms", 0))

    # RT criterion (IMPORTANT: should match run condition)
    rt_p_thresh = float(get("rt_p_thresh", 0.6))
    rt_k_consec = int(get("rt_k_consec", 2))

    # Noise in rendering (IMPORTANT: should match run condition)
    sigma_other_noise = float(get("sigma_other_noise", 0.0))
    p_other_noise = float(get("p_other_noise", 1.0))
    sigma_silence_noise = float(get("sigma_silence_noise", 0.0))

    # split controls
    seed = int(args_dict.get("seed", 42))
    val_split = float(args_dict.get("val_split", 0.1))
    max_blocks = int(args_dict.get("max_blocks", 0) or 0)
    isi_schedule = _parse_isi_schedule(args_dict.get("isi_schedule", []))

    return dict(
        seed=seed, val_split=val_split, max_blocks=max_blocks, isi_schedule=isi_schedule,
        tone_ms=tone_ms, token_ms=token_ms, ramp_ms=ramp_ms,
        n_bins=n_bins, f_min_hz=f_min_hz, f_max_hz=f_max_hz, add_eos=add_eos,
        chunk_len=chunk_len, lambda_token=lambda_token,
        token_loss_mode=token_loss_mode, token_tau=token_tau, token_w_min=token_w_min,
        tok_window_ms=tok_window_ms, tok_start_offset_ms=tok_start_offset_ms,
        rt_p_thresh=rt_p_thresh, rt_k_consec=rt_k_consec,
        sigma_other_noise=sigma_other_noise, p_other_noise=p_other_noise, sigma_silence_noise=sigma_silence_noise,
        params_source=("sweep" if ("sweep" in ckpt and ckpt["sweep"]) else "args")
    )

@torch.no_grad()
def collect_model_trial_rts_on_loader(model, loader, ds, *,
                                      chunk_len, token_loss_mode, token_tau, token_w_min,
                                      tok_window_ms, tok_start_offset_ms,
                                      rt_p_thresh, rt_k_consec,
                                      device):
    rows = []
    model.eval()
    for x, y_pos_456 in loader:
        x = x.to(device)
        y_pos_456 = y_pos_456.long().to(device)
        B, T, D = x.shape
        end_idx = infer_end_indices_from_T(int(T), trials_per_block=10).to(device)

        # full logits for RT
        _, _, logits_all = _run_block_through_tbptt(
            model=model,
            x=x,
            y_pos_456=y_pos_456,
            end_idx=end_idx,
            chunk_len=int(chunk_len),
            token_ce=nn.CrossEntropyLoss(reduction="mean"),
            trial_T_tokens=int(ds.trial_T_tokens),
            tone_T=int(ds.tone_T),
            isi_T=int(ds.isi_T),
            token_ms=int(ds.token_ms),
            tok_window_ms=int(tok_window_ms),
            tok_start_offset_ms=int(tok_start_offset_ms),
            token_loss_mode=str(token_loss_mode),
            token_tau=float(token_tau),
            token_w_min=float(token_w_min),
            return_full_logits=True,
        )
        if logits_all is None:
            continue

        logits_trial = logits_all.view(B, 10, int(ds.trial_T_tokens), 3).detach().cpu()
        y_cpu = y_pos_456.detach().cpu().long()
        y_cls_cpu = (y_cpu - 4).long()
        dev_end_cpu = deviant_end_token_in_trial(
            y_pos_456=y_cpu, tone_T=int(ds.tone_T), isi_T=int(ds.isi_T)
        )

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

# =========================
# 2) LOAD CKPT + PARAMS
# =========================
ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
cfg_dict = ckpt.get("cfg", None)
if cfg_dict is None:
    raise RuntimeError("ckpt missing cfg")

params = _pick_params_from_ckpt(ckpt)

print("Loaded ckpt:", CKPT_PATH)
print("ckpt keys:", list(ckpt.keys()))
print("ckpt stage/isi:", ckpt.get("stage", None), ckpt.get("isi_ms", None))
print("cfg:", cfg_dict)
print("\n=== Params used (priority: sweep > args) ===")
print(json.dumps(params, indent=2))

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
# 4) Build split (same as training) using GM_DATA blocks
# =========================
# NOTE: for split indices we only need B. isi_ms here doesn't affect B.
base_ds = OnlineRenderDataset(
    data_dir=GM_DATA_DIR,
    seed=int(params["seed"]),
    tone_ms=int(params["tone_ms"]),
    isi_ms=0,  # arbitrary for split; B doesn't depend on it
    ramp_ms=int(params["ramp_ms"]),
    token_ms=int(params["token_ms"]),
    f_min_hz=float(params["f_min_hz"]),
    f_max_hz=float(params["f_max_hz"]),
    n_bins=int(params["n_bins"]),
    add_eos=bool(params["add_eos"]),
    sigma_other_noise=float(params["sigma_other_noise"]),
    p_other_noise=float(params["p_other_noise"]),
    sigma_silence_noise=float(params["sigma_silence_noise"]),
    quiet=True,
)

# respect max_blocks from training
if int(params["max_blocks"]) > 0:
    m = min(int(params["max_blocks"]), int(base_ds.B))
    base_ds.X = base_ds.X[:m]
    base_ds.Y = base_ds.Y[:m]
    base_ds.B = m

train_idx, val_idx = split_indices(
    n=len(base_ds),
    val_split=float(params["val_split"]),
    seed=int(params["seed"])
)

# =========================
# 5) SANITY A: eval on GM_DATA val split @ isi=700
# =========================
ISI_EVAL = 700

ds, _, val_loader = build_loaders_for_isi(
    data_dir=GM_DATA_DIR,
    seed=int(params["seed"]),
    tone_ms=int(params["tone_ms"]),
    isi_ms=int(ISI_EVAL),
    ramp_ms=int(params["ramp_ms"]),
    token_ms=int(params["token_ms"]),
    f_min_hz=float(params["f_min_hz"]),
    f_max_hz=float(params["f_max_hz"]),
    n_bins=int(params["n_bins"]),
    add_eos=bool(params["add_eos"]),
    sigma_other_noise=float(params["sigma_other_noise"]),
    p_other_noise=float(params["p_other_noise"]),
    sigma_silence_noise=float(params["sigma_silence_noise"]),
    train_idx=train_idx,
    val_idx=val_idx,
    batch_size=128,
    num_workers=0,
    device=DEVICE,
    assert_labels=False,
)

# ---- HARD ASSERTS: catch the usual "chance-level because mismatch" bugs ----
assert int(ds.isi_ms) == int(ISI_EVAL), f"Wrong ISI in dataset: {ds.isi_ms} != {ISI_EVAL}"
assert int(ds.input_dim) == int(cfg.input_dim), f"input_dim mismatch: ds={ds.input_dim} ckpt={cfg.input_dim}"
assert int(ds.n_bins) == int(params["n_bins"]), f"n_bins mismatch: ds={ds.n_bins} params={params['n_bins']}"
assert bool(ds.add_eos) == bool(params["add_eos"]), f"add_eos mismatch: ds={ds.add_eos} params={params['add_eos']}"

# quick batch stats: detect "all zeros" / NaNs
xb, yb = next(iter(val_loader))
print("\n[batch stats] x mean/std/min/max:",
      xb.float().mean().item(), xb.float().std().item(), xb.float().min().item(), xb.float().max().item())
print("[batch stats] y unique:", torch.unique(yb).tolist())

va = evaluate(
    model=model,
    loader=val_loader,
    device=DEVICE,
    chunk_len=int(params["chunk_len"]),
    lambda_token=float(params["lambda_token"]),
    trial_T_tokens=int(ds.trial_T_tokens),
    tone_T=int(ds.tone_T),
    isi_T=int(ds.isi_T),
    token_loss_mode=str(params["token_loss_mode"]),
    token_tau=float(params["token_tau"]),
    token_w_min=float(params["token_w_min"]),
    token_ms=int(ds.token_ms),
    tok_window_ms=int(params["tok_window_ms"]),
    tok_start_offset_ms=int(params["tok_start_offset_ms"]),
    rt_p_thresh=float(params["rt_p_thresh"]),
    rt_k_consec=int(params["rt_k_consec"]),
    debug_labels=False,
    epoch_global=int(ckpt.get("epoch_global", 0)),
)

print("\n=== Sanity A: eval on GM_DATA val split @ isi=700 ===")
print(va)

# =========================
# 6) SANITY B: compare RT by deviant position (human vs model)
# =========================
dfh = pd.read_csv(HUMAN_CSV)
need = {"position", "rt_ms"}
if not need.issubset(dfh.columns):
    raise RuntimeError(f"human_csv missing required columns: {need - set(dfh.columns)}")

# filter isi if present
if "isi_ms" in dfh.columns:
    dfh = dfh[dfh["isi_ms"] == int(ISI_EVAL)].copy()

dfh["position"] = pd.to_numeric(dfh["position"], errors="coerce")
dfh["rt_ms"] = pd.to_numeric(dfh["rt_ms"], errors="coerce")
dfh = dfh[dfh["position"].isin([4,5,6]) & dfh["rt_ms"].notna()].copy()

# model trial RTs on the SAME val_loader (GM_DATA stimuli)
dfm = collect_model_trial_rts_on_loader(
    model=model,
    loader=val_loader,
    ds=ds,
    chunk_len=int(params["chunk_len"]),
    token_loss_mode=str(params["token_loss_mode"]),
    token_tau=float(params["token_tau"]),
    token_w_min=float(params["token_w_min"]),
    tok_window_ms=int(params["tok_window_ms"]),
    tok_start_offset_ms=int(params["tok_start_offset_ms"]),
    rt_p_thresh=float(params["rt_p_thresh"]),
    rt_k_consec=int(params["rt_k_consec"]),
    device=DEVICE
)
dfm_found = dfm[dfm["found"] == 1].copy()

human_pos_mean = dfh.groupby("position")["rt_ms"].mean()
model_pos_mean = dfm_found.groupby("position")["rt_ms"].mean()

print("\n=== Sanity B: RT by deviant position (human vs model) ===")
print("Human mean RT (ms) by position:\n", human_pos_mean)
print("Model mean RT (ms) by position (found only):\n", model_pos_mean)

common_pos = sorted(set(human_pos_mean.index) & set(model_pos_mean.index))
if len(common_pos) >= 2:
    x = human_pos_mean.loc[common_pos].to_numpy()
    y = model_pos_mean.loc[common_pos].to_numpy()
    r = np.corrcoef(x, y)[0,1]
    print(f"\nPos-level Pearson r (rough, n={len(common_pos)}): {r:.3f}")
else:
    print("\nNot enough overlapping positions with found RTs to correlate.")

print("\nDONE.")