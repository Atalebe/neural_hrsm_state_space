#!/usr/bin/env python3
"""
Make figures for real Allen memory negative controls.

Inputs:
    real_memory_negative_control_summary.csv
    real_memory_negative_control_shuffles.csv
    real_memory_negative_control_pooled_summary.csv

Outputs:
    PNG figures plus control_figure_manifest.csv
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_labels(df):
    return df["region"] + " / " + df["stimulus_family"].str.replace("_", " ", regex=False)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "real_memory_negative_control_summary.csv")
    shuffles = pd.read_csv(results_dir / "real_memory_negative_control_shuffles.csv")
    pooled = pd.read_csv(results_dir / "real_memory_negative_control_pooled_summary.csv")

    ok = summary[summary["status"] == "ok"].copy()
    ok["label"] = make_labels(ok)
    ok = ok.sort_values("observed_minus_shuffle_mean")

    manifest = []

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ok))
    ax.bar(x, ok["observed_minus_shuffle_mean"].to_numpy(dtype=float))
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ok["label"], rotation=35, ha="right")
    ax.set_ylabel("Observed gain minus shuffled-lag mean")
    ax.set_title("Aligned lag history versus shuffled lag control")
    ax.grid(True, axis="y", linewidth=0.3)

    p1 = out_dir / "real_allen_observed_minus_shuffle_gain.png"
    savefig(fig, p1)
    manifest.append({
        "figure": p1.name,
        "description": "Difference between observed memory gain and the mean shuffled-lag memory gain for each region-stimulus group.",
        "source": "real_memory_negative_control_summary.csv",
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ok))
    ax.bar(x, ok["observed_memory_gain_r2"].to_numpy(dtype=float), label="Observed")
    ax.scatter(x, ok["shuffle_gain_mean"].to_numpy(dtype=float), marker="o", label="Shuffle mean")
    ax.errorbar(
        x,
        ok["shuffle_gain_mean"].to_numpy(dtype=float),
        yerr=ok["shuffle_gain_sd"].to_numpy(dtype=float),
        fmt="none",
        capsize=3,
    )
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ok["label"], rotation=35, ha="right")
    ax.set_ylabel("Memory gain, ΔR²")
    ax.set_title("Observed memory gain against shuffled-lag null")
    ax.legend()
    ax.grid(True, axis="y", linewidth=0.3)

    p2 = out_dir / "real_allen_observed_vs_shuffle_memory_gain.png"
    savefig(fig, p2)
    manifest.append({
        "figure": p2.name,
        "description": "Observed memory gain compared with shuffled-lag null mean and standard deviation.",
        "source": "real_memory_negative_control_summary.csv",
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    ok_p = ok.sort_values("empirical_p_shuffle_ge_observed")
    x = np.arange(len(ok_p))
    ax.bar(x, ok_p["empirical_p_shuffle_ge_observed"].to_numpy(dtype=float))
    ax.axhline(0.05, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ok_p["label"], rotation=35, ha="right")
    ax.set_ylabel("Empirical p, shuffle gain ≥ observed gain")
    ax.set_title("Shuffled-lag empirical p-values")
    ax.grid(True, axis="y", linewidth=0.3)

    p3 = out_dir / "real_allen_shuffle_empirical_p_values.png"
    savefig(fig, p3)
    manifest.append({
        "figure": p3.name,
        "description": "Empirical p-values from the shuffled-lag negative control. Lower values mean observed gain exceeds most shuffled controls.",
        "source": "real_memory_negative_control_summary.csv",
    })

    sh = shuffles.copy()
    sh["label"] = make_labels(sh)
    order = ok["label"].tolist()
    data = [sh.loc[sh["label"] == label, "shuffle_memory_gain_r2"].to_numpy(dtype=float) for label in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    obs_map = dict(zip(ok["label"], ok["observed_memory_gain_r2"]))
    ax.scatter(
        np.arange(1, len(order) + 1),
        [obs_map[label] for label in order],
        marker="D",
        label="Observed",
    )
    ax.axhline(0, linewidth=0.8)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Memory gain, ΔR²")
    ax.set_title("Observed memory gain over shuffled-lag distributions")
    ax.legend()
    ax.grid(True, axis="y", linewidth=0.3)

    p4 = out_dir / "real_allen_shuffle_distribution_with_observed.png"
    savefig(fig, p4)
    manifest.append({
        "figure": p4.name,
        "description": "Distribution of shuffled-lag memory gains with observed gain overlaid as diamond markers.",
        "source": "real_memory_negative_control_shuffles.csv",
    })

    run_note = pooled.to_dict(orient="records")[0] if len(pooled) else {}
    manifest_df = pd.DataFrame(manifest)
    manifest_df["run_note"] = str(run_note)
    manifest_path = out_dir / "control_figure_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"[ok] wrote figures to {out_dir}")
    print(f"[ok] wrote {manifest_path}")
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
