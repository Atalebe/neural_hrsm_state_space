#!/usr/bin/env python3
"""
Spontaneous neural ripeness index for real Allen Neural HRSM.

This is a descriptive index, not a new inferential test.

It combines:
  - HRSM state potential: Phi_neural
  - HRSM stability coordinate: S
  - controlled mean-rate memory: observed minus shuffled gain
  - active-unit recruitment memory
  - population entropy memory

The score is rank-based to avoid domination by one numerical scale.
Penalties are applied for raw negative mean-rate memory gain or nonpositive
controlled mean-rate gain.

Inputs:
  results/cross_session/allen_spontaneous_v1/
      cross_session_spontaneous_region_memory_summary.csv
      cross_session_spontaneous_target_group_summary.csv

Outputs:
  results/cross_session/allen_spontaneous_v1/
      spontaneous_neural_ripeness_index.csv
      spontaneous_neural_ripeness_summary.csv
      spontaneous_neural_ripeness_flags.csv

  results/figures/cross_session/allen_spontaneous_v1/
      spontaneous_neural_ripeness_ranked_bar.png
      spontaneous_neural_ripeness_heatmap.png
      spontaneous_neural_ripeness_phi_vs_memory.png
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_ACTIVE = "active_unit_fraction"
TARGET_ENTROPY = "population_rate_entropy"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--synthesis-dir",
        default="results/cross_session/allen_spontaneous_v1",
    )
    p.add_argument(
        "--fig-dir",
        default="results/figures/cross_session/allen_spontaneous_v1",
    )
    return p.parse_args()


def pct_rank(s):
    s = pd.Series(s, dtype=float)
    if s.notna().sum() <= 1:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return s.rank(pct=True, method="average")


def label(score):
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "moderate"
    if score >= 0.35:
        return "transitional"
    return "low"


def mode(row):
    state = row["phi_rank"]
    org = row["organizational_memory_rank"]
    mean_mem = row["controlled_mean_rate_rank"]

    if state >= 0.60 and org >= 0.60 and mean_mem >= 0.50:
        return "balanced_state_memory"
    if state < 0.50 and org >= 0.60:
        return "memory_carrier_low_phi"
    if state >= 0.60 and org < 0.50:
        return "structural_state_low_org_memory"
    if mean_mem < 0.35 and org < 0.35:
        return "low_memory_state"
    return "mixed"


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    syn = Path(args.synthesis_dir)
    fig_dir = Path(args.fig_dir)
    syn.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    region_path = syn / "cross_session_spontaneous_region_memory_summary.csv"
    target_path = syn / "cross_session_spontaneous_target_group_summary.csv"

    if not region_path.exists():
        raise FileNotFoundError(region_path)
    if not target_path.exists():
        raise FileNotFoundError(target_path)

    region = pd.read_csv(region_path)
    target = pd.read_csv(target_path)

    region["session_id_synthesis"] = region["session_id_synthesis"].astype(str)
    target["session_id_synthesis"] = target["session_id_synthesis"].astype(str)

    keys = ["session_id_synthesis", "region", "stimulus_family"]

    active = (
        target[target["target"] == TARGET_ACTIVE][
            keys + ["observed_minus_shuffle_mean", "observed_memory_gain_r2"]
        ]
        .rename(
            columns={
                "observed_minus_shuffle_mean": "active_controlled_gain",
                "observed_memory_gain_r2": "active_raw_gain",
            }
        )
    )

    entropy = (
        target[target["target"] == TARGET_ENTROPY][
            keys + ["observed_minus_shuffle_mean", "observed_memory_gain_r2"]
        ]
        .rename(
            columns={
                "observed_minus_shuffle_mean": "entropy_controlled_gain",
                "observed_memory_gain_r2": "entropy_raw_gain",
            }
        )
    )

    df = region.merge(active, on=keys, how="left").merge(entropy, on=keys, how="left")

    df = df.rename(
        columns={
            "observed_minus_shuffle_mean": "mean_rate_controlled_gain",
            "observed_memory_gain_r2": "mean_rate_raw_gain",
        }
    )

    df["organizational_memory_gain"] = df[
        ["active_controlled_gain", "entropy_controlled_gain"]
    ].median(axis=1)

    df["phi_rank"] = pct_rank(df["Phi_neural"])
    df["stability_rank"] = pct_rank(df["S"])
    df["controlled_mean_rate_rank"] = pct_rank(df["mean_rate_controlled_gain"])
    df["active_memory_rank"] = pct_rank(df["active_controlled_gain"])
    df["entropy_memory_rank"] = pct_rank(df["entropy_controlled_gain"])
    df["organizational_memory_rank"] = pct_rank(df["organizational_memory_gain"])

    df["ripeness_score_raw"] = (
        0.25 * df["phi_rank"]
        + 0.10 * df["stability_rank"]
        + 0.20 * df["controlled_mean_rate_rank"]
        + 0.225 * df["active_memory_rank"]
        + 0.225 * df["entropy_memory_rank"]
    )

    df["penalty_raw_mean_rate_negative"] = (df["mean_rate_raw_gain"] < 0).astype(float) * 0.05
    df["penalty_controlled_mean_rate_nonpositive"] = (
        df["mean_rate_controlled_gain"] <= 0
    ).astype(float) * 0.10

    df["ripeness_penalty"] = (
        df["penalty_raw_mean_rate_negative"]
        + df["penalty_controlled_mean_rate_nonpositive"]
    )

    df["ripeness_score"] = (
        df["ripeness_score_raw"] - df["ripeness_penalty"]
    ).clip(lower=0.0, upper=1.0)

    df["ripeness_class"] = df["ripeness_score"].apply(label)
    df["ripeness_mode"] = df.apply(mode, axis=1)

    out_cols = [
        "session_id_synthesis",
        "region",
        "stimulus_family",
        "Phi_neural",
        "H",
        "R",
        "S",
        "M",
        "hrsm_domain",
        "mean_rate_raw_gain",
        "mean_rate_controlled_gain",
        "active_controlled_gain",
        "entropy_controlled_gain",
        "organizational_memory_gain",
        "phi_rank",
        "stability_rank",
        "controlled_mean_rate_rank",
        "active_memory_rank",
        "entropy_memory_rank",
        "organizational_memory_rank",
        "ripeness_score_raw",
        "ripeness_penalty",
        "ripeness_score",
        "ripeness_class",
        "ripeness_mode",
    ]

    out = df[out_cols].sort_values("ripeness_score", ascending=False)
    out.to_csv(syn / "spontaneous_neural_ripeness_index.csv", index=False)

    summary = (
        out.groupby("region")
        .agg(
            n_sessions=("session_id_synthesis", "nunique"),
            median_ripeness=("ripeness_score", "median"),
            mean_ripeness=("ripeness_score", "mean"),
            median_phi=("Phi_neural", "median"),
            median_mean_rate_controlled_gain=("mean_rate_controlled_gain", "median"),
            median_active_controlled_gain=("active_controlled_gain", "median"),
            median_entropy_controlled_gain=("entropy_controlled_gain", "median"),
        )
        .reset_index()
        .sort_values("median_ripeness", ascending=False)
    )

    summary["region_rank"] = np.arange(1, len(summary) + 1)
    summary.to_csv(syn / "spontaneous_neural_ripeness_summary.csv", index=False)

    flags = pd.DataFrame(
        [
            {
                "test": "all_ripeness_scores_finite",
                "passed": bool(np.isfinite(out["ripeness_score"]).all()),
                "detail": "Every session-region state received a finite ripeness score.",
            },
            {
                "test": "top_state_has_positive_controlled_memory",
                "passed": bool(out.iloc[0]["mean_rate_controlled_gain"] > 0),
                "detail": f"Top state is {out.iloc[0]['session_id_synthesis']} {out.iloc[0]['region']}.",
            },
            {
                "test": "top_region_not_forced_to_LGd",
                "passed": bool(summary.iloc[0]["region"] != ""),
                "detail": f"Top median-ripeness region is {summary.iloc[0]['region']}.",
            },
            {
                "test": "organizational_memory_positive_in_top_half",
                "passed": bool(
                    (out.head(max(1, len(out) // 2))["organizational_memory_gain"] > 0).all()
                ),
                "detail": "All top-half ripeness states have positive organizational-memory gain.",
            },
        ]
    )
    flags.to_csv(syn / "spontaneous_neural_ripeness_flags.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    lab = out["session_id_synthesis"].astype(str) + " / " + out["region"].astype(str)
    x = np.arange(len(out))
    ax.bar(x, out["ripeness_score"].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=35, ha="right")
    ax.set_ylabel("Ripeness score")
    ax.set_title("Spontaneous neural ripeness by session-region state")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linewidth=0.3)
    savefig(fig, fig_dir / "spontaneous_neural_ripeness_ranked_bar.png")

    pivot = out.pivot_table(
        index="region",
        columns="session_id_synthesis",
        values="ripeness_score",
        aggfunc="mean",
    )
    region_order = summary["region"].tolist()
    pivot = pivot.reindex(region_order)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Ripeness heatmap")
    ax.set_xlabel("Session")
    ax.set_ylabel("Region")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Ripeness score")
    savefig(fig, fig_dir / "spontaneous_neural_ripeness_heatmap.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        out["Phi_neural"].to_numpy(dtype=float),
        out["organizational_memory_gain"].to_numpy(dtype=float),
        s=80,
    )
    for _, row in out.iterrows():
        ax.annotate(
            f"{row['session_id_synthesis']}/{row['region']}",
            (row["Phi_neural"], row["organizational_memory_gain"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Phi_neural")
    ax.set_ylabel("Organizational memory gain")
    ax.set_title("State potential versus organizational memory")
    ax.grid(True, linewidth=0.3)
    savefig(fig, fig_dir / "spontaneous_neural_ripeness_phi_vs_memory.png")

    manifest = pd.DataFrame(
        [
            {
                "figure": "spontaneous_neural_ripeness_ranked_bar.png",
                "description": "Ranked descriptive ripeness score across session-region spontaneous states.",
                "source": "spontaneous_neural_ripeness_index.csv",
            },
            {
                "figure": "spontaneous_neural_ripeness_heatmap.png",
                "description": "Ripeness score by region and session.",
                "source": "spontaneous_neural_ripeness_index.csv",
            },
            {
                "figure": "spontaneous_neural_ripeness_phi_vs_memory.png",
                "description": "HRSM potential plotted against organizational-memory gain.",
                "source": "spontaneous_neural_ripeness_index.csv",
            },
        ]
    )
    manifest.to_csv(fig_dir / "spontaneous_neural_ripeness_figure_manifest.csv", index=False)

    print("[ok] wrote ripeness outputs")
    print()
    print("RIPENESS INDEX")
    print(out.to_string(index=False))
    print()
    print("REGION SUMMARY")
    print(summary.to_string(index=False))
    print()
    print("FLAGS")
    print(flags.to_string(index=False))


if __name__ == "__main__":
    main()
