#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DOWNLOADS = Path.home() / "Downloads"
INPUT_FILES = [
    DOWNLOADS / "separate_eos_family_eval_merged.csv",
    DOWNLOADS / "participant_level_rt_eval.csv",
    DOWNLOADS / "participant_concat_with_acc_at_rt_eval.csv",
]
OUT_DIR = Path(__file__).resolve().parent / "crossvalidated_behavioral_checkpoint_analysis"


sns.set_theme(style="whitegrid", context="talk")


REQUIRED_SYNONYMS = {
    "participant": ["participant", "participant_id", "subject", "sub"],
    "rf": ["rf"],
    "noise": ["noise", "sigma_noise", "sig_noise"],
    "epoch": ["epoch", "step"],
    "checkpoint": ["ckpt", "checkpoint_path", "checkpoint"],
    "a_R2": ["a_R2", "R2_model", "r2_model"],
    "c_R2": ["c_R2", "c_R2_prior", "R2_prior", "c_R2_surprisal"],
    "delta_R2": ["delta_R2", "standalone_delta_R2", "a_R2_minus_c_R2"],
    "a_r": ["a_r", "r_model", "corr_model"],
    "found_rate": ["found_rate", "model_found_rate"],
    "acc_at_rt": ["acc_at_rt", "accuracy_at_rt", "acc"],
    "family": ["model_family", "family"],
}


def find_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {key: find_column(df.columns, aliases) for key, aliases in REQUIRED_SYNONYMS.items()}


def score_table(df: pd.DataFrame, detected: dict[str, str | None]) -> tuple[int, dict[str, int]]:
    cols_present = sum(detected[k] is not None for k in ["participant", "rf", "noise", "a_R2", "c_R2", "a_r", "found_rate", "acc_at_rt"])
    candidate_cols = ["participant", "rf", "noise", "checkpoint", "epoch", "a_R2", "c_R2", "delta_R2", "a_r", "found_rate", "acc_at_rt"]
    non_null_score = 0
    detail = {}
    for key in candidate_cols:
        col = detected.get(key)
        if col is None:
            detail[key] = 0
            continue
        n = int(df[col].notna().sum())
        detail[key] = n
        non_null_score += min(n, 1000)
    participant_bonus = 0
    part_col = detected.get("participant")
    if part_col is not None:
        participant_bonus = int(df[part_col].nunique()) * 100
    score = cols_present * 10_000 + participant_bonus + non_null_score
    return score, detail


def inspect_and_select_input() -> tuple[Path, pd.DataFrame, dict[str, str | None], dict[str, dict[str, int]]]:
    inspections: dict[str, dict[str, int]] = {}
    best: tuple[int, Path, pd.DataFrame, dict[str, str | None]] | None = None
    for path in INPUT_FILES:
        df = pd.read_csv(path)
        detected = detect_columns(df)
        score, detail = score_table(df, detected)
        inspections[path.name] = detail
        if best is None or score > best[0]:
            best = (score, path, df, detected)
    assert best is not None
    _, path, df, detected = best
    return path, df, detected, inspections


def standardize_table(df: pd.DataFrame, detected: dict[str, str | None]) -> pd.DataFrame:
    rename_map = {}
    for key, col in detected.items():
        if col is not None:
            rename_map[col] = key
    out = df.rename(columns=rename_map).copy()

    if "participant" not in out.columns:
        raise ValueError("No participant column detected.")
    out["participant"] = out["participant"].astype(str)

    for col in ["rf", "noise", "epoch", "a_R2", "c_R2", "a_r", "found_rate", "acc_at_rt"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "delta_R2" in out.columns:
        out["delta_R2"] = pd.to_numeric(out["delta_R2"], errors="coerce")
    else:
        out["delta_R2"] = np.nan
    if out["delta_R2"].isna().all() or out["delta_R2"].isna().any():
        if {"a_R2", "c_R2"}.issubset(out.columns):
            missing = out["delta_R2"].isna()
            out.loc[missing, "delta_R2"] = out.loc[missing, "a_R2"] - out.loc[missing, "c_R2"]

    if {"rf", "noise"}.issubset(out.columns):
        out["model_family"] = out.apply(
            lambda r: f"rf{float(r['rf']):g}_noise{float(r['noise']):g}"
            if pd.notna(r["rf"]) and pd.notna(r["noise"])
            else np.nan,
            axis=1,
        )
    elif "family" in out.columns:
        out["model_family"] = out["family"]
    else:
        out["model_family"] = np.nan

    if "checkpoint" not in out.columns:
        out["checkpoint"] = np.nan
    checkpoint_base = out["checkpoint"].fillna("")
    epoch_piece = out["epoch"].apply(lambda x: f"epoch={int(x)}" if pd.notna(x) else "epoch=NA")
    if "readout_mode" in out.columns:
        readout = out["readout_mode"].fillna("NA").astype(str)
    else:
        readout = pd.Series(["NA"] * len(out), index=out.index)
    pthr = out["p_threshold"] if "p_threshold" in out.columns else pd.Series([np.nan] * len(out), index=out.index)
    kcon = out["k_consec"] if "k_consec" in out.columns else pd.Series([np.nan] * len(out), index=out.index)
    out["selected_checkpoint"] = [
        f"{ck} | {ep} | readout={rm} | p={pt if pd.notna(pt) else 'NA'} | k={int(k) if pd.notna(k) else 'NA'}"
        for ck, ep, rm, pt, k in zip(checkpoint_base, epoch_piece, readout, pthr, kcon)
    ]

    essential = ["participant", "model_family", "rf", "noise", "selected_checkpoint", "a_R2", "c_R2", "delta_R2", "a_r", "found_rate", "acc_at_rt"]
    missing_essential = [c for c in essential if c not in out.columns]
    if missing_essential:
        raise ValueError(f"Missing essential columns after standardization: {missing_essential}")

    out = out.dropna(subset=["participant", "model_family", "selected_checkpoint", "a_R2", "c_R2", "delta_R2", "a_r", "found_rate", "acc_at_rt"]).copy()
    return out


def choose_checkpoint_for_family(df_family: pd.DataFrame, test_participant: str) -> dict[str, object]:
    validation = df_family[df_family["participant"] != test_participant].copy()
    heldout = df_family[df_family["participant"] == test_participant].copy()
    if heldout.empty:
        raise ValueError(f"No held-out rows for {test_participant}.")
    if test_participant in set(validation["participant"]):
        raise AssertionError("Held-out participant leaked into validation set.")

    rows = []
    for ckpt, g in validation.groupby("selected_checkpoint", sort=False):
        valid_mask = (
            (g["a_r"] > 0)
            & (g["delta_R2"] > 0)
            & (g["found_rate"] > 0.6)
            & (g["acc_at_rt"] > 0.7)
        )
        valid = g[valid_mask]
        rows.append(
            {
                "selected_checkpoint": ckpt,
                "validation_median_delta_R2": float(valid["delta_R2"].median()) if not valid.empty else np.nan,
                "validation_mean_delta_R2": float(valid["delta_R2"].mean()) if not valid.empty else np.nan,
                "validation_n_valid": int(len(valid)),
                "validation_n_positive": int((valid["delta_R2"] > 0).sum()),
                "fallback_median_delta_R2": float(g["delta_R2"].median()),
                "fallback_mean_delta_R2": float(g["delta_R2"].mean()),
                "fallback_n_valid": int(len(g)),
                "fallback_n_positive": int((g["delta_R2"] > 0).sum()),
            }
        )
    val_df = pd.DataFrame(rows)

    preferred = val_df[val_df["validation_n_valid"] > 0].copy()
    fallback_used = False
    if preferred.empty:
        fallback_used = True
        preferred = val_df.sort_values(
            ["fallback_median_delta_R2", "fallback_mean_delta_R2", "fallback_n_positive"],
            ascending=[False, False, False],
            kind="stable",
        ).head(1)
        selected = preferred.iloc[0]
        val_median = float(selected["fallback_median_delta_R2"])
        val_mean = float(selected["fallback_mean_delta_R2"])
        val_n = int(selected["fallback_n_valid"])
        val_n_pos = int(selected["fallback_n_positive"])
    else:
        preferred = preferred.sort_values(
            ["validation_median_delta_R2", "validation_mean_delta_R2", "validation_n_positive"],
            ascending=[False, False, False],
            kind="stable",
        ).head(1)
        selected = preferred.iloc[0]
        val_median = float(selected["validation_median_delta_R2"])
        val_mean = float(selected["validation_mean_delta_R2"])
        val_n = int(selected["validation_n_valid"])
        val_n_pos = int(selected["validation_n_positive"])

    ckpt_key = str(selected["selected_checkpoint"])
    heldout_match = heldout[heldout["selected_checkpoint"] == ckpt_key].copy()
    if heldout_match.empty:
        raise ValueError(f"No held-out evaluation for selected checkpoint: {ckpt_key}")
    heldout_row = heldout_match.iloc[0]

    return {
        "selected_checkpoint": ckpt_key,
        "validation_median_delta_R2": val_median,
        "validation_mean_delta_R2": val_mean,
        "validation_n_valid": val_n,
        "validation_n_positive": val_n_pos,
        "heldout_a_R2": float(heldout_row["a_R2"]),
        "heldout_c_R2": float(heldout_row["c_R2"]),
        "heldout_delta_R2": float(heldout_row["delta_R2"]),
        "heldout_a_r": float(heldout_row["a_r"]),
        "heldout_found_rate": float(heldout_row["found_rate"]),
        "heldout_acc_at_rt": float(heldout_row["acc_at_rt"]),
        "fallback_used": bool(fallback_used),
        "sanity_validation_participants": int(validation["participant"].nunique()),
    }


def build_selection_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_family, fam_df in df.groupby("model_family", sort=False):
        rf = float(fam_df["rf"].dropna().iloc[0]) if fam_df["rf"].notna().any() else np.nan
        noise = float(fam_df["noise"].dropna().iloc[0]) if fam_df["noise"].notna().any() else np.nan
        participants = sorted(fam_df["participant"].unique().tolist())
        checkpoints = fam_df["selected_checkpoint"].nunique()
        print(f"Family {model_family}: {len(participants)} participants, {checkpoints} checkpoints")
        for test_participant in participants:
            result = choose_checkpoint_for_family(fam_df, test_participant)
            if result["sanity_validation_participants"] != len(participants) - 1:
                raise AssertionError(
                    f"Expected {len(participants) - 1} validation participants for {model_family}/{test_participant}, "
                    f"got {result['sanity_validation_participants']}"
                )
            rows.append(
                {
                    "model_family": model_family,
                    "rf": rf,
                    "noise": noise,
                    "test_participant": test_participant,
                    **{k: v for k, v in result.items() if k != "sanity_validation_participants"},
                }
            )
    out = pd.DataFrame(rows)
    return out


def summarize_selection(selection_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_family, g in selection_df.groupby("model_family", sort=False):
        counts = g["selected_checkpoint"].value_counts().to_dict()
        most_freq = next(iter(g["selected_checkpoint"].mode().tolist()), "")
        top_count = max(counts.values()) if counts else 0
        rows.append(
            {
                "model_family": model_family,
                "rf": g["rf"].iloc[0],
                "noise": g["noise"].iloc[0],
                "median_heldout_delta_R2": float(g["heldout_delta_R2"].median()),
                "mean_heldout_delta_R2": float(g["heldout_delta_R2"].mean()),
                "n_heldout_delta_R2_gt_0": int((g["heldout_delta_R2"] > 0).sum()),
                "n_heldout_a_r_gt_0": int((g["heldout_a_r"] > 0).sum()),
                "median_heldout_found_rate": float(g["heldout_found_rate"].median()),
                "median_heldout_acc_at_rt": float(g["heldout_acc_at_rt"].median()),
                "n_fallback_cases": int(g["fallback_used"].sum()),
                "most_frequent_selected_checkpoint": most_freq,
                "most_frequent_selected_checkpoint_count": int(top_count),
                "n_unique_selected_checkpoints": int(len(counts)),
                "selected_checkpoint_counts": json.dumps(counts, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def sort_families(summary_df: pd.DataFrame) -> list[str]:
    ordered = summary_df.sort_values(
        ["median_heldout_delta_R2", "n_heldout_delta_R2_gt_0", "mean_heldout_delta_R2"],
        ascending=[False, False, False],
        kind="stable",
    )
    return ordered["model_family"].tolist()


def save_fig(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(selection_df: pd.DataFrame, family_order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(family_order) * 0.9), 6))
    sns.boxplot(data=selection_df, x="model_family", y="heldout_delta_R2", order=family_order, ax=ax, color="#D8E6F2", fliersize=0)
    sns.stripplot(data=selection_df, x="model_family", y="heldout_delta_R2", order=family_order, ax=ax, color="#2D6A9F", size=5, alpha=0.8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Model family")
    ax.set_ylabel("Held-out delta_R2")
    ax.set_title("Held-out delta_R2 by model family")
    ax.tick_params(axis="x", rotation=60)
    save_fig(fig, OUT_DIR / "heldout_delta_R2_boxplot_by_model_family")


def plot_positive_bar(summary_df: pd.DataFrame, family_order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(family_order) * 0.9), 5))
    data = summary_df.set_index("model_family").loc[family_order].reset_index()
    sns.barplot(data=data, x="model_family", y="n_heldout_delta_R2_gt_0", ax=ax, color="#5B8DB8")
    ax.set_xlabel("Model family")
    ax.set_ylabel("n(heldout delta_R2 > 0)")
    ax.set_title("Positive held-out delta_R2 counts by model family")
    ax.tick_params(axis="x", rotation=60)
    save_fig(fig, OUT_DIR / "positive_heldout_delta_R2_count_by_model_family")


def plot_heatmap(selection_df: pd.DataFrame, family_order: list[str]) -> None:
    pivot = selection_df.pivot(index="test_participant", columns="model_family", values="heldout_delta_R2")
    pivot = pivot.reindex(sorted(pivot.index), axis=0).reindex(family_order, axis=1)
    fig, ax = plt.subplots(figsize=(max(10, len(family_order) * 0.8), max(6, pivot.shape[0] * 0.4)))
    sns.heatmap(pivot, cmap="RdBu_r", center=0.0, ax=ax)
    ax.set_xlabel("Model family")
    ax.set_ylabel("Participant")
    ax.set_title("Held-out delta_R2 heatmap")
    ax.tick_params(axis="x", rotation=60)
    save_fig(fig, OUT_DIR / "participant_by_model_family_heldout_delta_R2_heatmap")


def plot_participant_lines(selection_df: pd.DataFrame, family_order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(family_order) * 0.9), 6))
    plot_df = selection_df.copy()
    plot_df["model_family"] = pd.Categorical(plot_df["model_family"], categories=family_order, ordered=True)
    sns.lineplot(
        data=plot_df.sort_values(["test_participant", "model_family"]),
        x="model_family",
        y="heldout_delta_R2",
        hue="test_participant",
        marker="o",
        linewidth=1.2,
        ax=ax,
        legend=False,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Model family")
    ax.set_ylabel("Held-out delta_R2")
    ax.set_title("Participant-specific held-out delta_R2 across model families")
    ax.tick_params(axis="x", rotation=60)
    save_fig(fig, OUT_DIR / "participant_specific_heldout_delta_R2_across_model_families")


def plot_selected_checkpoint_counts(selection_df: pd.DataFrame, family_order: list[str]) -> None:
    rows = []
    for fam in family_order:
        counts = selection_df.loc[selection_df["model_family"] == fam, "selected_checkpoint"].value_counts()
        rows.append(
            {
                "model_family": fam,
                "top_selected_checkpoint_count": int(counts.max()) if not counts.empty else 0,
                "n_unique_selected_checkpoints": int(len(counts)),
            }
        )
    count_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(10, len(family_order) * 0.9), 6))
    sns.barplot(data=count_df, x="model_family", y="top_selected_checkpoint_count", ax=ax, color="#7FB069")
    ax2 = ax.twinx()
    sns.lineplot(
        data=count_df,
        x="model_family",
        y="n_unique_selected_checkpoints",
        ax=ax2,
        color="#2D6A9F",
        marker="o",
        linewidth=1.5,
    )
    ax.set_xlabel("Model family")
    ax.set_ylabel("Top checkpoint selection count")
    ax2.set_ylabel("Unique selected checkpoints")
    ax.set_title("Checkpoint selection stability by model family")
    ax.tick_params(axis="x", rotation=60)
    save_fig(fig, OUT_DIR / "selected_checkpoint_count_by_model_family")


def plot_validation_vs_heldout(selection_df: pd.DataFrame, family_order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=selection_df,
        x="validation_median_delta_R2",
        y="heldout_delta_R2",
        hue="model_family",
        hue_order=family_order,
        ax=ax,
        s=60,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Validation median delta_R2")
    ax.set_ylabel("Held-out delta_R2")
    ax.set_title("Validation median delta_R2 vs held-out delta_R2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_fig(fig, OUT_DIR / "validation_median_delta_R2_vs_heldout_delta_R2")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_path, selected_raw_df, detected, inspections = inspect_and_select_input()
    print(f"Selected file: {selected_path}")
    print(f"Detected columns: {json.dumps(detected, indent=2, default=str)}")
    print("Inspection summary:")
    print(json.dumps(inspections, indent=2))

    df = standardize_table(selected_raw_df, detected)
    participants = sorted(df["participant"].unique().tolist())
    model_families = sorted(df["model_family"].unique().tolist())
    checkpoints_per_family = df.groupby("model_family")["selected_checkpoint"].nunique().sort_values(ascending=False)

    print(f"Participants: {len(participants)}")
    print(f"Model families: {len(model_families)}")
    print("Checkpoints per family:")
    print(checkpoints_per_family.to_string())

    selection_df = build_selection_table(df)
    selection_path = OUT_DIR / "crossvalidated_behavioral_checkpoint_selection.csv"
    selection_df.to_csv(selection_path, index=False)

    summary_df = summarize_selection(selection_df)
    family_order = sort_families(summary_df)
    summary_df = summary_df.set_index("model_family").loc[family_order].reset_index()
    summary_path = OUT_DIR / "crossvalidated_behavioral_checkpoint_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_boxplot(selection_df, family_order)
    plot_positive_bar(summary_df, family_order)
    plot_heatmap(selection_df, family_order)
    plot_participant_lines(selection_df, family_order)
    plot_selected_checkpoint_counts(selection_df, family_order)
    plot_validation_vs_heldout(selection_df, family_order)

    print("Top 5 families by median heldout_delta_R2:")
    print(summary_df[["model_family", "median_heldout_delta_R2"]].head(5).to_string(index=False))
    print("Top 5 families by n positive participants:")
    print(summary_df.sort_values(["n_heldout_delta_R2_gt_0", "median_heldout_delta_R2"], ascending=[False, False]).head(5)[["model_family", "n_heldout_delta_R2_gt_0"]].to_string(index=False))
    print(f"Fallback count: {int(summary_df['n_fallback_cases'].sum())}")
    print("Checkpoint selection used the other 18 participants only. Held-out delta_R2 therefore tests behavioral generalization without participant-specific checkpoint peak-picking.")


if __name__ == "__main__":
    main()
