#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path.home() / "Downloads" / "autodl_sync" / "merged_participant_level_with_train_loss.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "crossvalidated_behavioral_checkpoint_analysis" / "merged_train_loss"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Leave-one-participant-out checkpoint selection on merged participant-level "
            "RT eval tables, carrying through train/val losses from metrics.csv."
        )
    )
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--min_found_rate", type=float, default=0.60)
    p.add_argument("--min_acc_at_rt", type=float, default=0.70)
    p.add_argument("--require_positive_a_r", action="store_true", default=True)
    p.add_argument("--allow_nonpositive_a_r", dest="require_positive_a_r", action="store_false")
    p.add_argument("--require_positive_delta", action="store_true", default=True)
    p.add_argument("--allow_nonpositive_delta", dest="require_positive_delta", action="store_false")
    return p.parse_args()


def _parse_noise_from_tag(noise_tag: Any) -> float:
    if pd.isna(noise_tag):
        return float("nan")
    s = str(noise_tag).strip()
    if not s:
        return float("nan")
    if s == "000":
        return 0.0
    try:
        return float(int(s)) / 100.0
    except Exception:
        return float("nan")


def _extract_epoch_from_model_name(model_name: str) -> float:
    m = re.search(r"epoch[_-]?0*([0-9]+)", str(model_name))
    if not m:
        return float("nan")
    return float(int(m.group(1)))


def standardize_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "participant" not in out.columns:
        raise ValueError("Missing required column: participant")
    if "epoch" not in out.columns:
        if "model_name" in out.columns:
            out["epoch"] = out["model_name"].map(_extract_epoch_from_model_name)
        else:
            raise ValueError("Missing required column: epoch")
    if "run_name" not in out.columns:
        raise ValueError("Missing required column: run_name")
    if "standalone_delta_R2" in out.columns and "delta_R2" not in out.columns:
        out["delta_R2"] = pd.to_numeric(out["standalone_delta_R2"], errors="coerce")
    if "delta_R2" not in out.columns:
        raise ValueError("Missing required delta_R2/standalone_delta_R2 column")
    if "c_R2_surprisal" in out.columns and "c_R2" not in out.columns:
        out["c_R2"] = pd.to_numeric(out["c_R2_surprisal"], errors="coerce")

    for col in [
        "epoch",
        "a_r",
        "a_R2",
        "c_R2",
        "delta_R2",
        "found_rate",
        "acc_at_rt",
        "train_total_loss",
        "train_token_loss",
        "train_acc",
        "train_window_acc",
        "train_window_acc_P4",
        "train_window_acc_P5",
        "train_window_acc_P6",
        "val_total_loss",
        "val_token_loss",
        "val_acc",
        "val_window_acc",
        "val_window_acc_P4",
        "val_window_acc_P5",
        "val_window_acc_P6",
        "p_threshold",
        "k_consec",
        "rf",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "noise" not in out.columns:
        if "noise_tag" in out.columns:
            out["noise"] = out["noise_tag"].map(_parse_noise_from_tag)
        else:
            out["noise"] = float("nan")

    out["participant"] = out["participant"].astype(str)
    out["run_name"] = out["run_name"].astype(str)
    out["readout_mode"] = out.get("readout_mode", "NA").astype(str)
    out["checkpoint_path"] = out.get("checkpoint_path", "").astype(str)

    out["candidate_key"] = out.apply(
        lambda r: " | ".join(
            [
                f"run={r['run_name']}",
                f"epoch={int(r['epoch'])}" if pd.notna(r["epoch"]) else "epoch=NA",
                f"readout={r['readout_mode']}",
                f"p={float(r['p_threshold']):.3f}" if pd.notna(r.get("p_threshold", np.nan)) else "p=NA",
                f"k={int(r['k_consec'])}" if pd.notna(r.get("k_consec", np.nan)) else "k=NA",
            ]
        ),
        axis=1,
    )

    essential = [
        "participant",
        "run_name",
        "epoch",
        "candidate_key",
        "a_r",
        "a_R2",
        "delta_R2",
        "found_rate",
        "acc_at_rt",
    ]
    out = out.dropna(subset=essential).copy()
    return out


def evaluate_candidate_group(
    g: pd.DataFrame,
    min_found_rate: float,
    min_acc_at_rt: float,
    require_positive_a_r: bool,
    require_positive_delta: bool,
) -> dict[str, Any]:
    valid_mask = np.ones(len(g), dtype=bool)
    if require_positive_a_r:
        valid_mask &= pd.to_numeric(g["a_r"], errors="coerce").to_numpy() > 0
    if require_positive_delta:
        valid_mask &= pd.to_numeric(g["delta_R2"], errors="coerce").to_numpy() > 0
    valid_mask &= pd.to_numeric(g["found_rate"], errors="coerce").to_numpy() > float(min_found_rate)
    valid_mask &= pd.to_numeric(g["acc_at_rt"], errors="coerce").to_numpy() > float(min_acc_at_rt)

    valid = g.loc[valid_mask].copy()
    if not valid.empty:
        return {
            "score": float(valid["delta_R2"].median()),
            "validation_mean_delta_R2": float(valid["delta_R2"].mean()),
            "validation_median_delta_R2": float(valid["delta_R2"].median()),
            "validation_n_valid": int(len(valid)),
            "validation_n_positive": int((valid["delta_R2"] > 0).sum()),
            "fallback_used": False,
        }

    return {
        "score": float(g["delta_R2"].median()),
        "validation_mean_delta_R2": float(g["delta_R2"].mean()),
        "validation_median_delta_R2": float(g["delta_R2"].median()),
        "validation_n_valid": 0,
        "validation_n_positive": int((g["delta_R2"] > 0).sum()),
        "fallback_used": True,
    }


def build_cv_selection_table(
    df: pd.DataFrame,
    min_found_rate: float,
    min_acc_at_rt: float,
    require_positive_a_r: bool,
    require_positive_delta: bool,
) -> pd.DataFrame:
    participants = sorted(df["participant"].unique().tolist())
    rows: list[dict[str, Any]] = []

    for test_participant in participants:
        validation = df[df["participant"] != test_participant].copy()
        heldout = df[df["participant"] == test_participant].copy()
        if heldout.empty:
            continue
        if validation["participant"].nunique() != len(participants) - 1:
            raise AssertionError(
                f"Expected {len(participants) - 1} validation participants for {test_participant}, "
                f"got {validation['participant'].nunique()}"
            )

        cand_rows: list[dict[str, Any]] = []
        for cand_key, g in validation.groupby("candidate_key", sort=False):
            stats = evaluate_candidate_group(
                g=g,
                min_found_rate=min_found_rate,
                min_acc_at_rt=min_acc_at_rt,
                require_positive_a_r=require_positive_a_r,
                require_positive_delta=require_positive_delta,
            )
            meta = g.iloc[0]
            cand_rows.append(
                {
                    "candidate_key": cand_key,
                    "run_name": str(meta["run_name"]),
                    "epoch": int(meta["epoch"]) if pd.notna(meta["epoch"]) else -1,
                    "rf": float(meta["rf"]) if "rf" in g.columns and pd.notna(meta["rf"]) else float("nan"),
                    "noise": float(meta["noise"]) if "noise" in g.columns and pd.notna(meta["noise"]) else float("nan"),
                    "readout_mode": str(meta["readout_mode"]),
                    "p_threshold": float(meta["p_threshold"]) if "p_threshold" in g.columns and pd.notna(meta["p_threshold"]) else float("nan"),
                    "k_consec": int(meta["k_consec"]) if "k_consec" in g.columns and pd.notna(meta["k_consec"]) else -1,
                    **stats,
                }
            )

        candidate_df = pd.DataFrame(cand_rows)
        if candidate_df.empty:
            continue
        selected = candidate_df.sort_values(
            ["score", "validation_mean_delta_R2", "validation_n_positive"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]

        heldout_match = heldout[heldout["candidate_key"] == selected["candidate_key"]].copy()
        if heldout_match.empty:
            raise ValueError(f"No held-out row found for selected candidate: {selected['candidate_key']}")
        heldout_row = heldout_match.iloc[0]

        rows.append(
            {
                "test_participant": test_participant,
                "selected_run_name": str(selected["run_name"]),
                "selected_epoch": int(selected["epoch"]),
                "selected_rf": float(selected["rf"]),
                "selected_noise": float(selected["noise"]),
                "selected_readout_mode": str(selected["readout_mode"]),
                "selected_p_threshold": float(selected["p_threshold"]) if pd.notna(selected["p_threshold"]) else float("nan"),
                "selected_k_consec": int(selected["k_consec"]) if pd.notna(selected["k_consec"]) and int(selected["k_consec"]) >= 0 else pd.NA,
                "selected_candidate_key": str(selected["candidate_key"]),
                "validation_score": float(selected["score"]),
                "validation_median_delta_R2": float(selected["validation_median_delta_R2"]),
                "validation_mean_delta_R2": float(selected["validation_mean_delta_R2"]),
                "validation_n_valid": int(selected["validation_n_valid"]),
                "validation_n_positive": int(selected["validation_n_positive"]),
                "fallback_used": bool(selected["fallback_used"]),
                "heldout_a_r": float(heldout_row["a_r"]),
                "heldout_a_R2": float(heldout_row["a_R2"]),
                "heldout_c_R2": float(heldout_row["c_R2"]) if "c_R2" in heldout_row.index and pd.notna(heldout_row["c_R2"]) else float("nan"),
                "heldout_delta_R2": float(heldout_row["delta_R2"]),
                "heldout_found_rate": float(heldout_row["found_rate"]),
                "heldout_acc_at_rt": float(heldout_row["acc_at_rt"]),
                "heldout_mean_rt_ms": float(heldout_row["mean_rt_ms"]) if "mean_rt_ms" in heldout_row.index and pd.notna(heldout_row["mean_rt_ms"]) else float("nan"),
                "train_total_loss": float(heldout_row["train_total_loss"]) if "train_total_loss" in heldout_row.index and pd.notna(heldout_row["train_total_loss"]) else float("nan"),
                "train_token_loss": float(heldout_row["train_token_loss"]) if "train_token_loss" in heldout_row.index and pd.notna(heldout_row["train_token_loss"]) else float("nan"),
                "train_acc": float(heldout_row["train_acc"]) if "train_acc" in heldout_row.index and pd.notna(heldout_row["train_acc"]) else float("nan"),
                "train_window_acc": float(heldout_row["train_window_acc"]) if "train_window_acc" in heldout_row.index and pd.notna(heldout_row["train_window_acc"]) else float("nan"),
                "train_window_acc_P4": float(heldout_row["train_window_acc_P4"]) if "train_window_acc_P4" in heldout_row.index and pd.notna(heldout_row["train_window_acc_P4"]) else float("nan"),
                "train_window_acc_P5": float(heldout_row["train_window_acc_P5"]) if "train_window_acc_P5" in heldout_row.index and pd.notna(heldout_row["train_window_acc_P5"]) else float("nan"),
                "train_window_acc_P6": float(heldout_row["train_window_acc_P6"]) if "train_window_acc_P6" in heldout_row.index and pd.notna(heldout_row["train_window_acc_P6"]) else float("nan"),
                "val_total_loss": float(heldout_row["val_total_loss"]) if "val_total_loss" in heldout_row.index and pd.notna(heldout_row["val_total_loss"]) else float("nan"),
                "val_window_acc": float(heldout_row["val_window_acc"]) if "val_window_acc" in heldout_row.index and pd.notna(heldout_row["val_window_acc"]) else float("nan"),
                "source_checkpoint_path": str(heldout_row["checkpoint_path"]) if "checkpoint_path" in heldout_row.index else "",
            }
        )

    return pd.DataFrame(rows)


def build_candidate_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["run_name", "epoch", "readout_mode"]
    extra = []
    if "p_threshold" in df.columns:
        extra.append("p_threshold")
    if "k_consec" in df.columns:
        extra.append("k_consec")
    group_cols += extra
    agg_spec = {
        "rf": ("rf", "first"),
        "noise": ("noise", "first"),
        "median_delta_R2": ("delta_R2", "median"),
        "mean_delta_R2": ("delta_R2", "mean"),
        "median_found_rate": ("found_rate", "median"),
        "median_acc_at_rt": ("acc_at_rt", "median"),
        "n_participants": ("participant", "nunique"),
    }
    if "train_total_loss" in df.columns:
        agg_spec["mean_train_total_loss"] = ("train_total_loss", "mean")
    summary = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(**agg_spec)
        .sort_values(["median_delta_R2", "mean_delta_R2"], ascending=[False, False], kind="stable")
    )
    return summary


def build_run_summary(selection_df: pd.DataFrame) -> pd.DataFrame:
    if selection_df.empty:
        return pd.DataFrame()
    summary = (
        selection_df.groupby(["selected_run_name", "selected_rf", "selected_noise"], as_index=False)
        .agg(
            n_selected_participants=("test_participant", "nunique"),
            median_heldout_delta_R2=("heldout_delta_R2", "median"),
            mean_heldout_delta_R2=("heldout_delta_R2", "mean"),
            n_positive_heldout_delta_R2=("heldout_delta_R2", lambda s: int(np.sum(pd.to_numeric(s, errors="coerce") > 0))),
            median_heldout_found_rate=("heldout_found_rate", "median"),
            median_heldout_acc_at_rt=("heldout_acc_at_rt", "median"),
            median_train_total_loss=("train_total_loss", "median"),
            n_fallback_cases=("fallback_used", lambda s: int(np.sum(s.astype(bool)))),
        )
        .sort_values(
            ["median_heldout_delta_R2", "n_positive_heldout_delta_R2", "mean_heldout_delta_R2"],
            ascending=[False, False, False],
            kind="stable",
        )
    )
    return summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input_csv)
    df = standardize_input(raw)

    selection_df = build_cv_selection_table(
        df=df,
        min_found_rate=float(args.min_found_rate),
        min_acc_at_rt=float(args.min_acc_at_rt),
        require_positive_a_r=bool(args.require_positive_a_r),
        require_positive_delta=bool(args.require_positive_delta),
    )
    candidate_summary_df = build_candidate_summary(df)
    run_summary_df = build_run_summary(selection_df)

    selection_path = args.out_dir / "cv_selected_checkpoint_per_participant.csv"
    candidate_summary_path = args.out_dir / "cv_candidate_summary.csv"
    run_summary_path = args.out_dir / "cv_run_summary.csv"
    meta_path = args.out_dir / "cv_run_metadata.json"

    selection_df.to_csv(selection_path, index=False)
    candidate_summary_df.to_csv(candidate_summary_path, index=False)
    run_summary_df.to_csv(run_summary_path, index=False)

    meta = {
        "input_csv": str(args.input_csv.resolve()),
        "n_rows_input": int(len(raw)),
        "n_rows_standardized": int(len(df)),
        "n_participants": int(df["participant"].nunique()),
        "n_candidates": int(df["candidate_key"].nunique()),
        "min_found_rate": float(args.min_found_rate),
        "min_acc_at_rt": float(args.min_acc_at_rt),
        "require_positive_a_r": bool(args.require_positive_a_r),
        "require_positive_delta": bool(args.require_positive_delta),
        "outputs": {
            "selection_csv": str(selection_path),
            "candidate_summary_csv": str(candidate_summary_path),
            "run_summary_csv": str(run_summary_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote: {selection_path}")
    print(f"wrote: {candidate_summary_path}")
    print(f"wrote: {run_summary_path}")
    if not run_summary_df.empty:
        print("\nTop runs by median held-out delta_R2:")
        cols = [
            "selected_run_name",
            "selected_rf",
            "selected_noise",
            "median_heldout_delta_R2",
            "n_positive_heldout_delta_R2",
            "median_heldout_found_rate",
            "median_heldout_acc_at_rt",
            "median_train_total_loss",
        ]
        print(run_summary_df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
