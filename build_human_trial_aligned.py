#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build an aligned human_trial.csv from many BIDS-like events TSV files.

Input:
- log_dir: contains sub-XX-runYY.tsv (tab-separated), with columns:
    onset, duration, trial_type, answer, response_time
  trial_type examples:
    TRloc6-std1455Hz-dev1600Hz-len8-st0
    TRloc6-std1455Hz-dev1600Hz-len8-std
    TRloc6-std1455Hz-dev1600Hz-len8-dev
  response_time is assumed in SECONDS (as in your screenshot); we convert to ms.

- gm_data_dir: contains input_blocks.pt (B,10,8) and labels_blocks.pt (B,10) with values {4,5,6}

We:
1) Parse per-trial responses from TSVs (one row per TR*, using the "-dev" row as the response anchor).
2) Build gm signatures (isi_ms, position, std_hz, dev_hz) for each gm trial.
3) Match human trials to gm trials by signature using a queue (handles duplicates).
4) Output aligned CSV with block_idx/trial_idx/freqs_8 so eval can be correct.

This avoids relying on row order (which your mismatch warning shows is wrong).
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch


TR_RE = re.compile(
    r"TRloc(?P<loc>[456])"
    r"-std(?P<std>\d+(?:\.\d+)?)Hz"
    r"-dev(?P<dev>\d+(?:\.\d+)?)Hz"
    r"-len8-(?P<tag>st0|std|dev)$"
)

def parse_one_tsv(tsv_path: Path, isi_ms: int) -> pd.DataFrame:
    """
    Extract deviant trials from one events TSV.
    Returns rows: subject_id, run_id, isi_ms, position, rt_ms, std_hz, dev_hz
    """
    df = pd.read_csv(tsv_path, sep="\t")
    need = {"trial_type"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"{tsv_path.name} missing columns: {need - set(df.columns)}")

    subject_id = tsv_path.name.split("-")[0]  # "sub-01"
    run_id = None
    m = re.search(r"(run\d+)", tsv_path.stem)
    if m:
        run_id = m.group(1)
    else:
        run_id = tsv_path.stem

    rows = []
    for _, r in df.iterrows():
        tt = str(r.get("trial_type", ""))
        m = TR_RE.match(tt)
        if not m:
            continue
        if m.group("tag") != "dev":
            continue  # use dev row as the "trial" record

        pos = int(m.group("loc"))
        std_hz = float(m.group("std"))
        dev_hz = float(m.group("dev"))

        rt = r.get("response_time", np.nan)
        # response_time sometimes empty; also sometimes negative in your screenshot
        rt = pd.to_numeric(rt, errors="coerce")
        rt_ms = float(rt) * 1000.0 if np.isfinite(rt) else np.nan

        rows.append(
            dict(
                subject_id=subject_id,
                run_id=run_id,
                isi_ms=int(isi_ms),
                position=int(pos),
                rt_ms=rt_ms,
                std_hz=std_hz,
                dev_hz=dev_hz,
                source_tsv=str(tsv_path),
            )
        )

    out = pd.DataFrame(rows)
    return out


def gm_trial_signature(freqs_8: np.ndarray, y_pos: int) -> Tuple[int, int, float, float]:
    """
    Given one gm trial (8 tones) and deviant position y_pos in {4,5,6},
    compute (pos, std_hz, dev_hz).

    Assumption: gm_stimuli uses a single std freq and a single dev freq per trial.
    We infer std as the mode of the sequence, and dev as the tone at the deviant position.
    This is robust even if the first tones are not always std.
    """
    pos = int(y_pos)
    dev_idx0 = pos - 1  # 1-indexed -> 0-indexed
    dev_hz = float(freqs_8[dev_idx0])

    # std is the most frequent value in the 8-tone sequence
    # (works because std repeats many times)
    vals, counts = np.unique(freqs_8.astype(np.float64), return_counts=True)
    std_hz = float(vals[np.argmax(counts)])

    return pos, std_hz, dev_hz


def build_gm_queues(
    gm_data_dir: Path,
    isi_ms: int,
) -> Dict[Tuple[int, int, float, float], deque]:
    """
    Build signature -> queue of gm trials.
    Each item in queue: (block_idx, trial_idx, freqs_8(list))
    Signature: (isi_ms, position, std_hz, dev_hz)
    """
    xb = gm_data_dir / "input_blocks.pt"
    yb = gm_data_dir / "labels_blocks.pt"
    if not xb.exists() or not yb.exists():
        raise RuntimeError("Need input_blocks.pt and labels_blocks.pt in gm_data_dir")

    X = torch.load(xb, map_location="cpu").numpy().astype(np.float32)  # (B,10,8)
    Y = torch.load(yb, map_location="cpu").numpy().astype(np.int64)    # (B,10)

    if X.ndim != 3 or X.shape[1:] != (10, 8):
        raise RuntimeError(f"input_blocks.pt expected (B,10,8), got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] != 10:
        raise RuntimeError(f"labels_blocks.pt expected (B,10), got {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise RuntimeError("Block count mismatch between input_blocks and labels_blocks")

    queues: Dict[Tuple[int, int, float, float], deque] = defaultdict(deque)

    B = X.shape[0]
    for b in range(B):
        for t in range(10):
            freqs_8 = X[b, t, :]
            pos = int(Y[b, t])
            pos2, std_hz, dev_hz = gm_trial_signature(freqs_8, pos)
            sig = (int(isi_ms), int(pos2), float(std_hz), float(dev_hz))
            queues[sig].append((int(b), int(t), freqs_8.tolist()))

    return queues


def match_human_to_gm(
    human_df: pd.DataFrame,
    gm_queues: Dict[Tuple[int, int, float, float], deque],
    isi_ms: int,
    tol_hz: float = 1e-6,
) -> pd.DataFrame:
    """
    Match each human trial to a gm trial by signature (isi, pos, std_hz, dev_hz).
    Uses tolerance by rounding to avoid float mismatch.
    """
    df = human_df.copy()

    # normalize floats for stable matching
    def norm(x: float) -> float:
        # match gm generation is usually exact (5 Hz step), so rounding is safe
        return float(np.round(float(x) / tol_hz) * tol_hz) if tol_hz > 0 else float(x)

    # Build a secondary dict with normalized keys (because gm keys are floats too)
    gm_norm: Dict[Tuple[int, int, float, float], deque] = defaultdict(deque)
    for (isi, pos, std, dev), q in gm_queues.items():
        k = (int(isi), int(pos), norm(std), norm(dev))
        gm_norm[k] = q  # queues already deques; shared reference is fine here

    block_idx = []
    trial_idx = []
    freqs_8_col = []
    matched = []

    miss = 0
    for _, r in df.iterrows():
        sig = (int(isi_ms), int(r["position"]), norm(r["std_hz"]), norm(r["dev_hz"]))
        q = gm_norm.get(sig, None)
        if q is None or len(q) == 0:
            miss += 1
            block_idx.append(-1)
            trial_idx.append(-1)
            freqs_8_col.append(None)
            matched.append(0)
            continue
        b, t, freqs8 = q.popleft()
        block_idx.append(b)
        trial_idx.append(t)
        freqs_8_col.append(freqs8)
        matched.append(1)

    df["block_idx"] = block_idx
    df["trial_idx"] = trial_idx
    df["freqs_8"] = freqs_8_col
    df["matched_to_gm"] = matched

    print(f"[match] matched={sum(matched)} / {len(df)}  miss={miss}")
    if miss > 0:
        print("[match] If miss is large, check: isi_ms correct? tol_hz? std/dev parsing? gm_data corresponds to experiment?")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True, help="Directory with sub-*-run*.tsv")
    ap.add_argument("--gm_data_dir", type=str, required=True, help="Directory with input_blocks.pt + labels_blocks.pt")
    ap.add_argument("--out_csv", type=str, required=True, help="Output aligned CSV path")
    ap.add_argument("--isi_ms", type=int, required=True, help="ISI in ms for these logs (e.g., 700)")
    ap.add_argument("--tol_hz", type=float, default=1e-6, help="Tolerance for matching std/dev Hz")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    gm_dir = Path(args.gm_data_dir)
    out_csv = Path(args.out_csv)

    tsvs = sorted(log_dir.glob("sub-*-run*.tsv"))
    if len(tsvs) == 0:
        raise RuntimeError(f"No TSVs found in {log_dir.resolve()}")

    # 1) parse all TSVs
    all_rows = []
    for p in tsvs:
        df = parse_one_tsv(p, isi_ms=int(args.isi_ms))
        if len(df) > 0:
            all_rows.append(df)
    if len(all_rows) == 0:
        raise RuntimeError("Parsed 0 dev trials from all TSVs. Check TR_RE or file content.")

    human = pd.concat(all_rows, ignore_index=True)

    # basic cleaning
    human["position"] = pd.to_numeric(human["position"], errors="coerce")
    human["rt_ms"] = pd.to_numeric(human["rt_ms"], errors="coerce")
    human = human[human["position"].isin([4, 5, 6])].copy()

    print(f"[human] rows={len(human)} subjects={human['subject_id'].nunique()} runs={human['run_id'].nunique()}")

    # 2) build gm queues
    gm_q = build_gm_queues(gm_dir, isi_ms=int(args.isi_ms))
    print(f"[gm] unique signatures={len(gm_q)} total_trials={sum(len(q) for q in gm_q.values())}")

    # 3) match
    aligned = match_human_to_gm(human, gm_q, isi_ms=int(args.isi_ms), tol_hz=float(args.tol_hz))

    # 4) diagnostics
    if (aligned["matched_to_gm"] == 0).mean() > 0.01:
        print("[warn] >1% unmatched. You probably mixed ISIs or gm_data does not correspond to these logs.")

    # 5) save
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(out_csv, index=False)
    print("[done] saved:", out_csv.resolve())


if __name__ == "__main__":
    main()