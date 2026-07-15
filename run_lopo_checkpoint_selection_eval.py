#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from run_crossvalidated_merged_participant_train_loss import evaluate_candidate_group


DEFAULT_FALLBACK_CKPTS = (
    "best_rt_candidate.pt",
    "best_isi700.pt",
    "best_isi300.pt",
    "best_isi100.pt",
    "best_isi0.pt",
    "best.pt",
    "latest.pt",
    "last.pt",
)

READOUT_KEY_COLS = (
    "readout_mode",
    "readout_type",
    "w",
    "timeout_ms",
    "cost_threshold",
    "kappa",
    "p_threshold",
    "k_consec",
    "effective_threshold_formula",
)


@dataclass(frozen=True)
class EvalJob:
    combination_index: int
    combination_total: int
    checkpoint_index: int
    checkpoint_total: int
    combination: str
    combo_dir: Path
    ckpt_paths: tuple[Path, ...]
    rf: float
    noise: float
    eval_seed: int
    eval_dir: Path

    @property
    def ckpt_path(self) -> Path:
        return self.ckpt_paths[0]

    @property
    def ckpt_label(self) -> str:
        if len(self.ckpt_paths) == 1:
            return self.ckpt_paths[0].name
        return f"{len(self.ckpt_paths)} checkpoints"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate checkpoints once with eval_models_on_human.py, then run "
            "leave-one-participant-out checkpoint/readout selection on the "
            "participant-level behavioral metrics."
        )
    )
    p.add_argument("--combo_root", type=Path, required=True, help="Root containing combination directories.")
    p.add_argument("--combo_glob", type=str, default="rf*_noise*", help="Glob under combo_root for combinations.")
    p.add_argument("--out_dir", type=Path, required=True, help="Output/cache directory for LOPO evaluation.")
    p.add_argument("--human_csv", type=Path, required=True)
    p.add_argument("--eval_script", type=Path, default=Path("eval_models_on_human.py"))
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--checkpoint_subdir", type=str, default="checkpoints_by_epoch")
    p.add_argument("--checkpoint_glob", type=str, default="*.pt")
    p.add_argument("--max_checkpoints_per_combo", type=int, default=0, help="0 means no limit.")
    p.add_argument("--eval_batch_mode", choices=["combo_seed", "checkpoint_seed"], default="combo_seed")
    p.add_argument("--exclude_combination", action="append", default=[])
    p.add_argument("--force_eval", action="store_true", help="Re-run checkpoint evals even when cached CSVs exist.")
    p.add_argument("--eval_seed_list", type=str, default="0", help="Space/comma-separated eval seeds. Multiple seeds are averaged before LOPO.")
    p.add_argument("--selection_scope", choices=["per_combo", "global"], default="per_combo")
    p.add_argument("--min_found_rate", type=float, default=0.60)
    p.add_argument("--min_acc_at_rt", type=float, default=0.70)
    p.add_argument("--require_positive_a_r", action="store_true", default=True)
    p.add_argument("--allow_nonpositive_a_r", dest="require_positive_a_r", action="store_false")
    p.add_argument("--require_positive_delta", action="store_true", default=True)
    p.add_argument("--allow_nonpositive_delta", dest="require_positive_delta", action="store_false")
    p.add_argument("--isi_filter", type=int, default=700)
    p.add_argument("--tone_ms", type=int, default=50)
    p.add_argument("--token_ms", type=int, default=10)
    p.add_argument("--f_min_hz", type=float, default=1300.0)
    p.add_argument("--f_max_hz", type=float, default=1700.0)
    p.add_argument("--n_bins", type=int, default=128)
    p.add_argument("--add_eos", action="store_true", default=True)
    p.add_argument("--no_add_eos", dest="add_eos", action="store_false")
    p.add_argument("--add_bos", action="store_true", default=True)
    p.add_argument("--no_add_bos", dest="add_bos", action="store_false")
    p.add_argument("--eos_mode", type=str, default="separate")
    p.add_argument("--encoding_mode", type=str, default="gaussian_rf")
    p.add_argument("--sigma_rf", type=float, default=float("nan"), help="Override RF width; NaN parses from combo name.")
    p.add_argument("--sigma_rf_noise", type=float, default=float("nan"), help="Override RF noise; NaN parses from combo name.")
    p.add_argument("--rf_normalization", type=str, default="peak")
    p.add_argument("--rf_noise_per_token", type=str, default="false")
    p.add_argument("--sigma_silence_noise", type=float, default=0.0)
    p.add_argument("--rt_readout_mode", type=str, default="simple_threshold")
    p.add_argument("--rt_p_thresh", type=float, default=0.40)
    p.add_argument("--rt_k_consec", type=int, default=1)
    p.add_argument("--p_correct_threshold_list", type=str, default="0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.80 0.90")
    p.add_argument("--cost_k_consec_list", type=str, default="1")
    p.add_argument("--cost_readout_window", type=str, default="deviant_onset_to_second_next_tone_onset")
    p.add_argument("--eval_sequence_mode", type=str, default="real_boundary_continuous")
    p.add_argument("--eval_context_rho", type=float, default=1.0)
    p.add_argument("--drop_first_n_trials_per_participant", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--chunk_len", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_progress(done: int, total: int, start_time: float, current: str, avg_sec: float) -> None:
    elapsed = time.time() - start_time
    pct = (done / total * 100.0) if total else 100.0
    eta = avg_sec * (total - done) if done > 0 and np.isfinite(avg_sec) else float("nan")
    width = 20
    filled = int(round(width * done / total)) if total else width
    bar = "█" * filled + "░" * (width - filled)
    line = (
        f"Overall {bar} {done}/{total} ({pct:5.1f}%) "
        f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)} "
        f"avg/checkpoint={avg_sec:5.1f}s current={current}"
    )
    term_width = shutil.get_terminal_size((140, 20)).columns
    if len(line) >= term_width:
        line = line[: max(1, term_width - 1)]
    print("\r" + line + "\033[K", end="", flush=True)


def parse_rf_noise(combo_name: str) -> tuple[float, float]:
    m = re.search(r"rf(?P<rf>\d+(?:\.\d+)?)_noise(?P<noise>\d+(?:\.\d+)?)", combo_name)
    if not m:
        return float("nan"), float("nan")
    rf = float(m.group("rf"))
    noise_tag = m.group("noise")
    if "." in noise_tag:
        noise = float(noise_tag)
    elif noise_tag == "000":
        noise = 0.0
    elif len(noise_tag) >= 3:
        noise = float(int(noise_tag)) / 100.0
    else:
        noise = float(noise_tag)
    return rf, noise


def parse_seed_list(seed_text: str) -> list[int]:
    vals = [x for x in re.split(r"[\s,]+", str(seed_text).strip()) if x]
    if not vals:
        return [0]
    return [int(x) for x in vals]


def collect_checkpoints(combo_dir: Path, checkpoint_subdir: str, checkpoint_glob: str, max_n: int) -> list[Path]:
    ckpt_dir = combo_dir / checkpoint_subdir
    ckpts = sorted(ckpt_dir.glob(checkpoint_glob)) if ckpt_dir.exists() else []
    if not ckpts:
        ckpts = [combo_dir / name for name in DEFAULT_FALLBACK_CKPTS if (combo_dir / name).is_file()]
    if max_n and max_n > 0:
        ckpts = ckpts[:max_n]
    return ckpts


def collect_jobs(args: argparse.Namespace) -> list[EvalJob]:
    combo_dirs = sorted(p for p in args.combo_root.glob(args.combo_glob) if p.is_dir())
    excluded = set(str(x) for x in args.exclude_combination)
    combo_dirs = [p for p in combo_dirs if p.name not in excluded and str(p) not in excluded]
    eval_seeds = parse_seed_list(str(args.eval_seed_list))
    jobs: list[EvalJob] = []
    for combo_i, combo_dir in enumerate(combo_dirs, start=1):
        parsed_rf, parsed_noise = parse_rf_noise(combo_dir.name)
        rf = float(args.sigma_rf) if np.isfinite(float(args.sigma_rf)) else parsed_rf
        noise = float(args.sigma_rf_noise) if np.isfinite(float(args.sigma_rf_noise)) else parsed_noise
        if not np.isfinite(rf) or not np.isfinite(noise):
            raise ValueError(f"Could not infer rf/noise for {combo_dir}; pass --sigma_rf/--sigma_rf_noise.")
        ckpts = collect_checkpoints(
            combo_dir=combo_dir,
            checkpoint_subdir=str(args.checkpoint_subdir),
            checkpoint_glob=str(args.checkpoint_glob),
            max_n=int(args.max_checkpoints_per_combo),
        )
        if args.eval_batch_mode == "combo_seed":
            ckpt_batches = [(1, tuple(ckpts))]
        else:
            ckpt_batches = [(ckpt_i, (ckpt,)) for ckpt_i, ckpt in enumerate(ckpts, start=1)]
        for ckpt_i, ckpt_batch in ckpt_batches:
            if not ckpt_batch:
                continue
            for eval_seed in eval_seeds:
                batch_name = "all_checkpoints" if len(ckpt_batch) > 1 else ckpt_batch[0].stem
                eval_dir = args.out_dir / "eval_cache" / combo_dir.name / batch_name / f"seed_{int(eval_seed):04d}"
                jobs.append(
                    EvalJob(
                        combination_index=combo_i,
                        combination_total=len(combo_dirs),
                        checkpoint_index=ckpt_i,
                        checkpoint_total=len(ckpt_batches),
                        combination=combo_dir.name,
                        combo_dir=combo_dir,
                        ckpt_paths=tuple(p.resolve() for p in ckpt_batch),
                        rf=rf,
                        noise=noise,
                        eval_seed=int(eval_seed),
                        eval_dir=eval_dir,
                    )
                )
    return jobs


def build_eval_cmd(args: argparse.Namespace, job: EvalJob) -> list[str]:
    cmd = [
        str(args.python),
        str(args.eval_script),
        "--human_csv",
        str(args.human_csv),
        "--out_dir",
        str(job.eval_dir),
        "--ckpt_list",
        ",".join(str(p) for p in job.ckpt_paths),
        "--isi_filter",
        str(args.isi_filter),
        "--tone_ms",
        str(args.tone_ms),
        "--token_ms",
        str(args.token_ms),
        "--f_min_hz",
        str(args.f_min_hz),
        "--f_max_hz",
        str(args.f_max_hz),
        "--n_bins",
        str(args.n_bins),
        "--eos_mode",
        str(args.eos_mode),
        "--encoding_mode",
        str(args.encoding_mode),
        "--sigma_rf",
        str(job.rf),
        "--rf_normalization",
        str(args.rf_normalization),
        "--sigma_rf_noise",
        str(job.noise),
        "--rf_noise_per_token",
        str(args.rf_noise_per_token),
        "--sigma_silence_noise",
        str(args.sigma_silence_noise),
        "--rt_readout_mode",
        str(args.rt_readout_mode),
        "--rt_p_thresh",
        str(args.rt_p_thresh),
        "--rt_k_consec",
        str(args.rt_k_consec),
        "--p_correct_threshold_list",
        str(args.p_correct_threshold_list),
        "--cost_k_consec_list",
        str(args.cost_k_consec_list),
        "--cost_readout_window",
        str(args.cost_readout_window),
        "--eval_sequence_mode",
        str(args.eval_sequence_mode),
        "--eval_context_rho",
        str(args.eval_context_rho),
        "--seed",
        str(job.eval_seed),
        "--drop_first_n_trials_per_participant",
        str(args.drop_first_n_trials_per_participant),
        "--batch_size",
        str(args.batch_size),
        "--chunk_len",
        str(args.chunk_len),
        "--device",
        str(args.device),
        "--no_trial_level_csv",
        "--no_figures",
    ]
    if bool(args.add_eos):
        cmd.append("--add_eos")
    if bool(args.add_bos):
        cmd.append("--add_bos")
    return cmd


def eval_csv_path(job: EvalJob) -> Path:
    return job.eval_dir / "participant_level_rt_eval.csv"


def run_eval_job(args: argparse.Namespace, job: EvalJob, done: int, total: int, start_time: float, durations: list[float]) -> Path:
    out_csv = eval_csv_path(job)
    if out_csv.is_file() and not bool(args.force_eval):
        return out_csv

    job.eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_eval_cmd(args, job)
    log_path = job.eval_dir / "eval_stdout_stderr.log"
    command_path = job.eval_dir / "eval_command.json"
    command_path.write_text(json.dumps({"cmd": cmd}, indent=2), encoding="utf-8")

    current = (
        f"Combination {job.combination_index}/{job.combination_total} {job.combination} | "
        f"Checkpoint {job.checkpoint_index}/{job.checkpoint_total} {job.ckpt_label} | seed {job.eval_seed}"
    )
    proc_start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).resolve().parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if line:
                log_f.write(line)
                log_f.flush()
            if proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    log_f.write(rest)
                break
            avg = float(np.mean(durations)) if durations else (time.time() - proc_start)
            render_progress(done, total, start_time, current=current, avg_sec=avg)
            time.sleep(0.5)

    elapsed = time.time() - proc_start
    if proc.returncode != 0:
        print()
        raise RuntimeError(f"Evaluation failed for {job.ckpt_label}. See {log_path}")
    if not out_csv.is_file():
        print()
        raise FileNotFoundError(f"Expected participant eval CSV was not produced: {out_csv}")
    durations.append(elapsed)
    return out_csv


def first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def parse_epoch_from_path(path: Any) -> float:
    m = re.search(r"epoch[_-]?0*([0-9]+)", str(path))
    if m:
        return float(int(m.group(1)))
    m = re.search(r"step[_-]?0*([0-9]+)", str(path))
    if m:
        return float(int(m.group(1)))
    return float("nan")


def standardize_participant_eval(df: pd.DataFrame, job: EvalJob) -> pd.DataFrame:
    out = df.copy()
    participant_col = first_present(out, ["participant", "participant_id", "subject_id", "sub"])
    if participant_col is None:
        raise ValueError(f"Missing participant column in {eval_csv_path(job)}")
    if participant_col != "participant":
        out["participant"] = out[participant_col]
    out["participant"] = out["participant"].astype(str)

    out["combination"] = job.combination
    out["combination_dir"] = str(job.combo_dir)
    if "checkpoint_path" in out.columns:
        out["checkpoint_path"] = out["checkpoint_path"].astype(str)
    else:
        if len(job.ckpt_paths) != 1:
            raise ValueError(f"Missing checkpoint_path column in batched eval output: {eval_csv_path(job)}")
        out["checkpoint_path"] = str(job.ckpt_path)
    out["checkpoint_file"] = out["checkpoint_path"].map(lambda x: Path(str(x)).name)
    out["checkpoint_stem"] = out["checkpoint_path"].map(lambda x: Path(str(x)).stem)
    out["rf"] = float(job.rf)
    out["noise"] = float(job.noise)
    out["eval_seed"] = int(job.eval_seed)
    out["eval_dir"] = str(job.eval_dir)

    if "epoch" not in out.columns:
        out["epoch"] = out["checkpoint_path"].map(parse_epoch_from_path)
    out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")

    if "standalone_delta_R2" in out.columns and "delta_R2" not in out.columns:
        out["delta_R2"] = pd.to_numeric(out["standalone_delta_R2"], errors="coerce")
    if "c_R2_surprisal" in out.columns and "c_R2" not in out.columns:
        out["c_R2"] = pd.to_numeric(out["c_R2_surprisal"], errors="coerce")
    if "delta_R2" not in out.columns and {"a_R2", "c_R2"}.issubset(out.columns):
        out["delta_R2"] = pd.to_numeric(out["a_R2"], errors="coerce") - pd.to_numeric(out["c_R2"], errors="coerce")

    for col in ["a_R2", "a_r", "c_R2", "delta_R2", "found_rate", "acc_at_rt", "mean_rt_ms", "p_threshold", "k_consec", "w", "timeout_ms", "cost_threshold"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in READOUT_KEY_COLS:
        if col not in out.columns:
            out[col] = np.nan

    key_cols = ["combination", "checkpoint_path", "epoch", *READOUT_KEY_COLS]
    out["candidate_key"] = out[key_cols].map(str).agg(" | ".join, axis=1)

    required = ["participant", "combination", "checkpoint_path", "candidate_key", "a_R2", "a_r", "c_R2", "delta_R2", "found_rate", "acc_at_rt"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns in {eval_csv_path(job)}: {missing}")
    return out.dropna(subset=["participant", "candidate_key", "a_R2", "a_r", "delta_R2", "found_rate", "acc_at_rt"]).copy()


def aggregate_eval_seed_metrics(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return seed_df.copy()
    if "eval_seed" not in seed_df.columns or seed_df["eval_seed"].nunique(dropna=False) <= 1:
        out = seed_df.copy()
        out["n_eval_seeds"] = 1
        out["eval_seed_list"] = out.get("eval_seed", pd.Series([0] * len(out))).astype(str)
        return out

    group_cols = ["participant", "candidate_key"]
    numeric_cols = [
        col
        for col in seed_df.columns
        if col not in group_cols and pd.api.types.is_numeric_dtype(seed_df[col]) and col != "eval_seed"
    ]
    rows: list[dict[str, Any]] = []
    for _, g in seed_df.groupby(group_cols, sort=False):
        row = g.iloc[0].to_dict()
        for col in numeric_cols:
            row[col] = float(pd.to_numeric(g[col], errors="coerce").mean())
        seeds = sorted(pd.to_numeric(g["eval_seed"], errors="coerce").dropna().astype(int).unique().tolist())
        row["eval_seed"] = "mean"
        row["eval_seed_list"] = ",".join(str(x) for x in seeds)
        row["n_eval_seeds"] = int(len(seeds))
        if {"a_R2", "c_R2"}.issubset(row):
            row["delta_R2"] = float(row["a_R2"] - row["c_R2"])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_validation_candidate(
    g: pd.DataFrame,
    min_found_rate: float,
    min_acc_at_rt: float,
    require_positive_a_r: bool,
    require_positive_delta: bool,
) -> dict[str, Any]:
    stats = evaluate_candidate_group(
        g=g,
        min_found_rate=min_found_rate,
        min_acc_at_rt=min_acc_at_rt,
        require_positive_a_r=require_positive_a_r,
        require_positive_delta=require_positive_delta,
    )
    meta = g.iloc[0].to_dict()
    return {
        "candidate_key": str(meta["candidate_key"]),
        "combination": str(meta["combination"]),
        "checkpoint_path": str(meta["checkpoint_path"]),
        "checkpoint_file": str(meta.get("checkpoint_file", "")),
        "checkpoint_stem": str(meta.get("checkpoint_stem", "")),
        "epoch": float(meta.get("epoch", np.nan)),
        "rf": float(meta.get("rf", np.nan)),
        "noise": float(meta.get("noise", np.nan)),
        "readout_mode": str(meta.get("readout_mode", "")),
        "p_threshold": float(meta.get("p_threshold", np.nan)) if pd.notna(meta.get("p_threshold", np.nan)) else np.nan,
        "k_consec": float(meta.get("k_consec", np.nan)) if pd.notna(meta.get("k_consec", np.nan)) else np.nan,
        "validation_n_participants": int(g["participant"].nunique()),
        **stats,
    }


def build_lopo_tables(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participants = sorted(df["participant"].unique().tolist())
    scopes = [("__global__", df)] if args.selection_scope == "global" else list(df.groupby("combination", sort=True))
    selection_rows: list[dict[str, Any]] = []
    candidate_rows_all: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []

    for scope_name, scope_df in scopes:
        for heldout_participant in participants:
            heldout = scope_df[scope_df["participant"] == heldout_participant].copy()
            validation = scope_df[scope_df["participant"] != heldout_participant].copy()
            if heldout.empty or validation.empty:
                continue

            candidate_rows = [
                aggregate_validation_candidate(
                    g=g,
                    min_found_rate=float(args.min_found_rate),
                    min_acc_at_rt=float(args.min_acc_at_rt),
                    require_positive_a_r=bool(args.require_positive_a_r),
                    require_positive_delta=bool(args.require_positive_delta),
                )
                for _, g in validation.groupby("candidate_key", sort=False)
            ]
            cand_df = pd.DataFrame(candidate_rows)
            if cand_df.empty:
                continue
            cand_df.insert(0, "selection_scope", str(scope_name))
            cand_df.insert(1, "heldout_participant", str(heldout_participant))
            candidate_rows_all.extend(cand_df.to_dict("records"))

            selected = cand_df.sort_values(
                ["fallback_used", "score", "validation_mean_delta_R2", "validation_n_positive"],
                ascending=[True, False, False, False],
                kind="stable",
            ).iloc[0]

            heldout_match = heldout[heldout["candidate_key"] == selected["candidate_key"]].copy()
            if heldout_match.empty:
                raise ValueError(f"No held-out row for selected candidate {selected['candidate_key']}")
            heldout_row = heldout_match.sort_values(["delta_R2", "a_R2"], ascending=[False, False], kind="stable").iloc[0]

            selected_dict = selected.to_dict()
            selection_rows.append(
                {
                    "selection_scope": str(scope_name),
                    "heldout_participant": str(heldout_participant),
                    "selected_candidate_key": str(selected["candidate_key"]),
                    "selected_combination": str(selected["combination"]),
                    "selected_checkpoint_path": str(selected["checkpoint_path"]),
                    "selected_checkpoint_file": str(selected.get("checkpoint_file", "")),
                    "selected_checkpoint_stem": str(selected.get("checkpoint_stem", "")),
                    "selected_epoch": float(selected.get("epoch", np.nan)),
                    "selected_rf": float(selected.get("rf", np.nan)),
                    "selected_noise": float(selected.get("noise", np.nan)),
                    "selected_readout_mode": str(selected.get("readout_mode", "")),
                    "selected_p_threshold": float(selected.get("p_threshold", np.nan)) if pd.notna(selected.get("p_threshold", np.nan)) else np.nan,
                    "selected_k_consec": float(selected.get("k_consec", np.nan)) if pd.notna(selected.get("k_consec", np.nan)) else np.nan,
                    "validation_score": float(selected["score"]),
                    "validation_mean_delta_R2": float(selected["validation_mean_delta_R2"]),
                    "validation_median_delta_R2": float(selected["validation_median_delta_R2"]),
                    "validation_n_valid": int(selected["validation_n_valid"]),
                    "validation_n_positive": int(selected["validation_n_positive"]),
                    "validation_n_participants": int(selected["validation_n_participants"]),
                    "fallback_used": bool(selected["fallback_used"]),
                    "heldout_a_r": float(heldout_row["a_r"]),
                    "heldout_a_R2": float(heldout_row["a_R2"]),
                    "heldout_c_R2": float(heldout_row["c_R2"]),
                    "heldout_delta_R2": float(heldout_row["delta_R2"]),
                    "heldout_found_rate": float(heldout_row["found_rate"]),
                    "heldout_acc_at_rt": float(heldout_row["acc_at_rt"]),
                    "heldout_mean_rt_ms": float(heldout_row.get("mean_rt_ms", np.nan)) if pd.notna(heldout_row.get("mean_rt_ms", np.nan)) else np.nan,
                }
            )

            heldout_payload = heldout_row.to_dict()
            heldout_payload.update(
                {
                    "selection_scope": str(scope_name),
                    "heldout_participant": str(heldout_participant),
                    "selected_candidate_key": str(selected_dict["candidate_key"]),
                    "validation_score": float(selected_dict["score"]),
                    "validation_median_delta_R2": float(selected_dict["validation_median_delta_R2"]),
                    "fallback_used": bool(selected_dict["fallback_used"]),
                }
            )
            heldout_rows.append(heldout_payload)

    return pd.DataFrame(selection_rows), pd.DataFrame(heldout_rows), pd.DataFrame(candidate_rows_all)


def save_outputs(seed_df: pd.DataFrame, df: pd.DataFrame, selection_df: pd.DataFrame, heldout_df: pd.DataFrame, candidate_df: pd.DataFrame, args: argparse.Namespace, jobs: list[EvalJob]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics_path = args.out_dir / "all_participant_checkpoint_seed_metrics.csv"
    all_metrics_path = args.out_dir / "all_participant_checkpoint_metrics.csv"
    selection_path = args.out_dir / "lopo_checkpoint_selection.csv"
    heldout_path = args.out_dir / "lopo_heldout_participant_metrics.csv"
    candidate_path = args.out_dir / "lopo_validation_candidate_scores.csv"
    summary_path = args.out_dir / "lopo_summary.json"

    seed_df.to_csv(seed_metrics_path, index=False)
    df.to_csv(all_metrics_path, index=False)
    selection_df.to_csv(selection_path, index=False)
    heldout_df.to_csv(heldout_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)

    summary = {
        "n_eval_jobs": len(jobs),
        "eval_seed_list": parse_seed_list(str(args.eval_seed_list)),
        "eval_batch_mode": str(args.eval_batch_mode),
        "n_combinations": len({j.combination for j in jobs}),
        "n_checkpoints": len({str(p) for j in jobs for p in j.ckpt_paths}),
        "n_seed_metric_rows": int(len(seed_df)),
        "n_metric_rows": int(len(df)),
        "n_participants": int(df["participant"].nunique()) if "participant" in df.columns else 0,
        "n_lopo_rows": int(len(selection_df)),
        "selection_scope": str(args.selection_scope),
        "min_found_rate": float(args.min_found_rate),
        "min_acc_at_rt": float(args.min_acc_at_rt),
        "require_positive_a_r": bool(args.require_positive_a_r),
        "require_positive_delta": bool(args.require_positive_delta),
        "outputs": {
            "all_participant_checkpoint_seed_metrics": str(seed_metrics_path),
            "all_participant_checkpoint_metrics": str(all_metrics_path),
            "lopo_checkpoint_selection": str(selection_path),
            "lopo_heldout_participant_metrics": str(heldout_path),
            "lopo_validation_candidate_scores": str(candidate_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = collect_jobs(args)
    if not jobs:
        raise SystemExit("No checkpoint evaluation jobs found.")

    start = time.time()
    durations: list[float] = []
    frames: list[pd.DataFrame] = []
    total = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        avg = float(np.mean(durations)) if durations else 0.0
        current = (
            f"Combination {job.combination_index}/{job.combination_total} {job.combination} | "
            f"Checkpoint {job.checkpoint_index}/{job.checkpoint_total} {job.ckpt_label} | seed {job.eval_seed}"
        )
        render_progress(idx - 1, total, start, current=current, avg_sec=avg)
        csv_path = run_eval_job(args, job, done=idx - 1, total=total, start_time=start, durations=durations)
        part_df = pd.read_csv(csv_path)
        frames.append(standardize_participant_eval(part_df, job))
        avg = float(np.mean(durations)) if durations else 0.0
        render_progress(idx, total, start, current=current, avg_sec=avg)

    print()
    seed_metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_metrics = aggregate_eval_seed_metrics(seed_metrics)
    selection_df, heldout_df, candidate_df = build_lopo_tables(all_metrics, args)
    save_outputs(seed_metrics, all_metrics, selection_df, heldout_df, candidate_df, args, jobs)

    elapsed = time.time() - start
    print(
        "Finished LOPO checkpoint selection: "
        f"{len(jobs)} checkpoint eval jobs, "
        f"{all_metrics['combination'].nunique() if not all_metrics.empty else 0} combinations, "
        f"{all_metrics['participant'].nunique() if not all_metrics.empty else 0} participants, "
        f"elapsed {format_duration(elapsed)}."
    )
    print(f"wrote: {args.out_dir / 'all_participant_checkpoint_metrics.csv'}")
    print(f"wrote: {args.out_dir / 'lopo_checkpoint_selection.csv'}")
    print(f"wrote: {args.out_dir / 'lopo_heldout_participant_metrics.csv'}")


if __name__ == "__main__":
    main()
