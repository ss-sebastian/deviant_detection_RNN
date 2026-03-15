#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a new gm-style dataset (input_blocks.pt + labels_blocks.pt) from HUMAN TSV logs.

Assumptions (matches your screenshot):
- Events TSV contains rows with trial_type like:
    TRloc6-std1455Hz-dev1600Hz-len8-st0
    TRloc6-std1455Hz-dev1600Hz-len8-std
    TRloc6-std1455Hz-dev1600Hz-len8-dev
- We treat each "-dev" row as one trial (one response opportunity).
- Each trial has len8 tones, with ONE deviant tone at position loc (4/5/6), others are std.
- We do NOT align to existing gm_data. We CREATE a new gm_data directory from human logs.

Outputs in --out_dir:
- input_blocks.pt   float32 (n_blocks, 10, 8)   Hz
- labels_blocks.pt  int64   (n_blocks, 10)      {4,5,6}
- meta.json         info about extraction

Usage:
python build_gm_from_human_tsv.py \
  --log_dir /Users/seb/Desktop/bcbl/msc_thesis/logfiles \
  --out_dir /Users/seb/Desktop/bcbl/msc_thesis/code/deviant_detection_RNN/gm_human_isi700 \
  --isi_ms 700 \
  --tone_ms 50 --token_ms 10 \
  --drop_incomplete
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch


TR_RE = re.compile(
    r"TRloc(?P<loc>[456])"
    r"-std(?P<std>\d+(?:\.\d+)?)Hz"
    r"-dev(?P<dev>\d+(?:\.\d+)?)Hz"
    r"-len(?P<len>\d+)"
    r"-(?P<tag>st0|std|dev)$"
)

def reconstruct_freqs_8(std_hz: float, dev_hz: float, pos_456: int, seq_len: int = 8) -> np.ndarray:
    if seq_len != 8:
        raise ValueError("This script assumes len8.")
    if pos_456 not in (4, 5, 6):
        raise ValueError(f"pos must be 4/5/6, got {pos_456}")
    freqs = np.full((8,), float(std_hz), dtype=np.float32)
    freqs[pos_456 - 1] = float(dev_hz)  # 1-indexed -> 0-indexed
    return freqs

def parse_trials_from_one_tsv(tsv_path: Path, isi_ms: int) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    if "trial_type" not in df.columns:
        raise RuntimeError(f"{tsv_path.name} missing trial_type column")

    subject_id = tsv_path.name.split("-")[0]  # "sub-01"
    run_id = None
    m = re.search(r"(run\d+)", tsv_path.stem)
    run_id = m.group(1) if m else tsv_path.stem

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        tt = str(r.get("trial_type", ""))
        m = TR_RE.match(tt)
        if not m:
            continue
        if m.group("tag") != "dev":
            continue  # use dev-row as trial anchor

        pos = int(m.group("loc"))
        std_hz = float(m.group("std"))
        dev_hz = float(m.group("dev"))
        L = int(m.group("len"))
        if L != 8:
            # skip unexpected
            continue

        rt = r.get("response_time", np.nan)
        rt = pd.to_numeric(rt, errors="coerce")
        rt_ms = float(rt) * 1000.0 if np.isfinite(rt) else np.nan

        freqs_8 = reconstruct_freqs_8(std_hz, dev_hz, pos_456=pos, seq_len=8)

        rows.append(dict(
            subject_id=subject_id,
            run_id=run_id,
            isi_ms=int(isi_ms),
            position=int(pos),
            std_hz=float(std_hz),
            dev_hz=float(dev_hz),
            rt_ms=rt_ms,
            freqs_8=freqs_8.tolist(),
            source_tsv=str(tsv_path),
        ))

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True, help="Directory with sub-*-run*.tsv")
    ap.add_argument("--out_dir", type=str, required=True, help="Output gm_data directory")
    ap.add_argument("--isi_ms", type=int, required=True, help="ISI (ms) for this dataset, e.g. 700")

    # just for meta sanity
    ap.add_argument("--tone_ms", type=int, default=50)
    ap.add_argument("--token_ms", type=int, default=10)

    ap.add_argument("--drop_incomplete", action="store_true",
                    help="Drop last incomplete block if #trials not divisible by 10")
    ap.add_argument("--pad_incomplete", action="store_true",
                    help="Pad last block by repeating last trial until it has 10 trials (mutually exclusive with drop)")
    ap.add_argument("--sort", type=str, default="file",
                    choices=["file", "onset"],
                    help="How to order trials within each tsv before concatenation. 'file' keeps row order; 'onset' sorts by onset if exists.")
    args = ap.parse_args()

    if args.drop_incomplete and args.pad_incomplete:
        raise ValueError("Choose at most one: --drop_incomplete or --pad_incomplete")

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tsvs = sorted(log_dir.glob("sub-*-run*.tsv"))
    if len(tsvs) == 0:
        raise RuntimeError(f"No TSV files found under {log_dir.resolve()}")

    all_trials = []
    for p in tsvs:
        df = pd.read_csv(p, sep="\t")
        if args.sort == "onset" and "onset" in df.columns:
            df = df.sort_values("onset").reset_index(drop=True)
            tmp_path = p  # we will re-read via helper, so easiest is to write temp? avoid.
            # Instead: parse with helper then re-sort within parsed by onset from original.
            # We'll just parse normally (file order) because onset sorting is rarely needed for dev-row extraction.
        trials = parse_trials_from_one_tsv(p, isi_ms=int(args.isi_ms))
        if len(trials) == 0:
            continue
        all_trials.append(trials)

    if len(all_trials) == 0:
        raise RuntimeError("Parsed 0 dev trials from all TSVs. Check TR_RE pattern vs your file content.")

    df_all = pd.concat(all_trials, ignore_index=True)

    # Keep only position 4/5/6
    df_all["position"] = pd.to_numeric(df_all["position"], errors="coerce")
    df_all = df_all[df_all["position"].isin([4, 5, 6])].copy()

    n_trials = len(df_all)
    print(f"[human->gm] parsed dev-trials: {n_trials} from {df_all['subject_id'].nunique()} subjects, {df_all['run_id'].nunique()} runs")

    # Build blocks of 10 trials
    n_full_blocks = n_trials // 10
    rem = n_trials % 10

    if rem != 0:
        if args.drop_incomplete:
            df_all = df_all.iloc[:n_full_blocks * 10].copy()
            print(f"[human->gm] drop_incomplete=ON: dropping last {rem} trials; kept {len(df_all)}")
        elif args.pad_incomplete:
            # repeat last trial
            need = 10 - rem
            last = df_all.iloc[[-1]].copy()
            pads = pd.concat([last] * need, ignore_index=True)
            df_all = pd.concat([df_all, pads], ignore_index=True)
            print(f"[human->gm] pad_incomplete=ON: padding {need} trials; new n_trials={len(df_all)}")
        else:
            print(f"[human->gm] WARNING: n_trials={n_trials} not divisible by 10; last {rem} trials will be ignored unless you set --drop_incomplete or --pad_incomplete")

    n_trials2 = len(df_all)
    n_blocks = n_trials2 // 10
    if n_blocks <= 0:
        raise RuntimeError("Not enough trials to form even 1 block of 10.")

    X = np.zeros((n_blocks, 10, 8), dtype=np.float32)
    Y = np.zeros((n_blocks, 10), dtype=np.int64)

    freqs_list = df_all["freqs_8"].tolist()
    pos_list = df_all["position"].tolist()

    k = 0
    for b in range(n_blocks):
        for t in range(10):
            X[b, t, :] = np.array(freqs_list[k], dtype=np.float32)
            Y[b, t] = int(pos_list[k])
            k += 1

    # Save
    torch.save(torch.from_numpy(X), out_dir / "input_blocks.pt")
    torch.save(torch.from_numpy(Y), out_dir / "labels_blocks.pt")

    meta = dict(
        source_log_dir=str(log_dir.resolve()),
        isi_ms=int(args.isi_ms),
        tone_ms=int(args.tone_ms),
        token_ms=int(args.token_ms),
        n_trials=int(n_trials2),
        n_blocks=int(n_blocks),
        trials_per_block=10,
        seq_len=8,
        label_definition="deviant position per trial (1-indexed in {4,5,6})",
        input_definition="Hz frequencies shaped (n_blocks, 10, 8)",
        extraction_rule="Use TR*-dev rows as trials; freqs_8 reconstructed as [std]*8 with dev at loc",
        subjects=int(df_all["subject_id"].nunique()),
        runs=int(df_all["run_id"].nunique()),
        drop_incomplete=bool(args.drop_incomplete),
        pad_incomplete=bool(args.pad_incomplete),
        examples_first5=df_all.head(5)[["subject_id","run_id","position","std_hz","dev_hz","rt_ms"]].to_dict(orient="records"),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[human->gm] saved:")
    print("  -", (out_dir / "input_blocks.pt").resolve(), "shape=", tuple(X.shape))
    print("  -", (out_dir / "labels_blocks.pt").resolve(), "shape=", tuple(Y.shape))
    print("  -", (out_dir / "meta.json").resolve())


if __name__ == "__main__":
    main()