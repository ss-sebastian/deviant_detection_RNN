# run_pipeline.py
# One-command pipeline runner for Colab/local:
#   A) generate GM blocks (gm_stimuli.py)
#   B) transform to ms-feature-C OR gammatone+smooth
#   C) train GRU with joint cls + predictive loss (train.py)
#
# Assumptions (by design, to keep this script stable):
# - gm_stimuli.py supports: --n_blocks ... --export_n N (or --export_all) --save_dir ...
# - transform scripts read from a directory that contains input_blocks.pt / labels_blocks.pt
# - train.py supports: --data_dir ... --input_pt ... --label_pt ... --save_dir ...
#
# Tip: if your transform scripts use different flag names, use the `--xform_extra ...`
#      pass-through to append arbitrary extra flags.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run(cmd: List[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()

    # --- outputs ---
    p.add_argument("--out_root", type=str, required=True, help="Root output folder (will create subfolders).")
    p.add_argument("--tag", type=str, default="run1", help="A name tag for this run (affects folder names).")

    # --- GM generation ---
    p.add_argument("--gm_script", type=str, default="gm_stimuli.py")
    p.add_argument("--n_blocks", type=int, default=10000, help="How many blocks to internally sample through.")
    p.add_argument("--export_n", type=int, default=1000, help="How many blocks to export to disk.")
    p.add_argument("--export_all", action="store_true", help="Export all blocks (ignores export_n).")

    p.add_argument("--f_min", type=float, required=True)
    p.add_argument("--f_max", type=float, required=True)
    p.add_argument("--f_step", type=float, required=True)
    p.add_argument("--min_diff", type=float, default=None, help="Default=f_step if omitted.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--exclude_freqs", type=float, nargs="*", default=[1455.0, 1500.0, 1600.0])
    p.add_argument("--no_seen_pairs", action="store_true")

    # --- Transform selection ---
    p.add_argument("--mode", choices=["ms", "gt"], required=True, help="Transform mode: ms feature-C, or gammatone (gt).")
    p.add_argument("--xform_feature_script", type=str, default="transform_stimuli_feature_C.py")
    p.add_argument("--xform_gt_script", type=str, default="transform_stimuli_gammatone_smooth.py")

    # Common transform params you probably want
    p.add_argument("--tone_ms", type=int, default=50)
    p.add_argument("--isi_ms", type=int, default=700)

    # GT-specific convenience flags (only used when mode=gt)
    p.add_argument("--smooth_ms", type=int, default=7)
    p.add_argument("--log_compress", action="store_true")

    # Append arbitrary extra flags to the transform command (for when your scripts differ)
    p.add_argument("--xform_extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra flags appended after the transform command, e.g. --xform_extra --foo 1 --bar baz")

    # --- Training ---
    p.add_argument("--train_script", type=str, default="train.py")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--chunk_len", type=int, default=1000)
    p.add_argument("--lambda_pred", type=float, default=0.1)
    p.add_argument("--layer_norm", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--device", type=str, default=None, help='e.g. "cuda" or "cpu". If omitted, train.py decides.')

    # misc
    p.add_argument("--dry_run", action="store_true", help="Print commands only; do not execute.")
    args = p.parse_args()

    py = sys.executable
    out_root = Path(args.out_root).expanduser().resolve()
    tag = args.tag

    # Layout
    gm_dir = out_root / "data" / f"raw_gm_{tag}"
    xform_dir = out_root / "data" / (f"ms_{tag}" if args.mode == "ms" else f"gt_{tag}")
    ckpt_dir = out_root / "checkpoints" / (f"ms_{tag}" if args.mode == "ms" else f"gt_{tag}")

    ensure_dir(gm_dir)
    ensure_dir(xform_dir)
    ensure_dir(ckpt_dir)

    # ---------- A) Generate blocks ----------
    gm_cmd = [
        py, args.gm_script,
        "--n_blocks", str(args.n_blocks),
        "--f_min", str(args.f_min),
        "--f_max", str(args.f_max),
        "--f_step", str(args.f_step),
        "--seed", str(args.seed),
        "--save_dir", str(gm_dir),
        "--exclude_freqs", *[str(v) for v in args.exclude_freqs],
    ]
    if args.min_diff is not None:
        gm_cmd += ["--min_diff", str(args.min_diff)]
    if args.no_seen_pairs:
        gm_cmd += ["--no_seen_pairs"]

    if args.export_all:
        gm_cmd += ["--export_all"]
    else:
        gm_cmd += ["--export_n", str(args.export_n)]

    run(gm_cmd, dry_run=args.dry_run)

    # ---------- B) Transform ----------
    # We copy/link the generated gm outputs into xform_dir to keep each stage self-contained.
    # (Colab sometimes has weird relative paths; this reduces pain.)
    if not args.dry_run:
        # minimal copying
        for name in ["input_blocks.pt", "labels_blocks.pt", "meta.json"]:
            src = gm_dir / name
            if src.exists():
                (xform_dir / name).write_bytes(src.read_bytes())

    if args.mode == "ms":
        # You control what transform_stimuli_feature_C.py expects.
        # We pass in_dir (directory) + tone/isi defaults; you can override via --xform_extra.
        xform_cmd = [
            py, args.xform_feature_script,
            "--in_dir", str(xform_dir),
            "--tone_ms", str(args.tone_ms),
            "--isi_ms", str(args.isi_ms),
        ] + args.xform_extra
        run(xform_cmd, dry_run=args.dry_run)

        input_pt = "ms_input_tensor.pt"
        label_pt = "ms_labels_tensor.pt"

    else:
        # gammatone transform convenience flags
        xform_cmd = [
            py, args.xform_gt_script,
            "--in_dir", str(xform_dir),
            "--tone_ms", str(args.tone_ms),
            "--isi_ms", str(args.isi_ms),
            "--smooth_ms", str(args.smooth_ms),
            "--save_name", "gt_input_tensor.pt",
        ]
        if args.log_compress:
            xform_cmd.append("--log_compress")
        xform_cmd += args.xform_extra
        run(xform_cmd, dry_run=args.dry_run)

        input_pt = "gt_input_tensor.pt"
        label_pt = "gt_labels_tensor.pt"

    # ---------- C) Train ----------
    train_cmd = [
        py, args.train_script,
        "--data_dir", str(xform_dir),
        "--save_dir", str(ckpt_dir),
        "--input_pt", input_pt,
        "--label_pt", label_pt,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--hidden_dim", str(args.hidden_dim),
        "--chunk_len", str(args.chunk_len),
        "--lambda_pred", str(args.lambda_pred),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
    ]
    if args.layer_norm:
        train_cmd.append("--layer_norm")
    if args.device is not None:
        train_cmd += ["--device", args.device]

    run(train_cmd, dry_run=args.dry_run)

    # ---------- Save a top-level run manifest ----------
    manifest = {
        "out_root": str(out_root),
        "tag": tag,
        "mode": args.mode,
        "gm_dir": str(gm_dir),
        "xform_dir": str(xform_dir),
        "ckpt_dir": str(ckpt_dir),
        "gm_args": {
            "n_blocks": args.n_blocks,
            "export_all": args.export_all,
            "export_n": args.export_n,
            "f_min": args.f_min,
            "f_max": args.f_max,
            "f_step": args.f_step,
            "min_diff": args.min_diff,
            "seed": args.seed,
            "exclude_freqs": args.exclude_freqs,
            "no_seen_pairs": args.no_seen_pairs,
        },
        "xform_args": {
            "tone_ms": args.tone_ms,
            "isi_ms": args.isi_ms,
            "smooth_ms": args.smooth_ms,
            "log_compress": args.log_compress,
            "xform_extra": args.xform_extra,
        },
        "train_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "chunk_len": args.chunk_len,
            "lambda_pred": args.lambda_pred,
            "layer_norm": args.layer_norm,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "device": args.device,
        },
    }

    if not args.dry_run:
        (out_root / f"manifest_{tag}_{args.mode}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print("GM:", gm_dir)
    print("XFORM:", xform_dir)
    print("CKPT:", ckpt_dir)
    if not args.dry_run:
        print("Manifest:", out_root / f"manifest_{tag}_{args.mode}.json")


if __name__ == "__main__":
    main()
