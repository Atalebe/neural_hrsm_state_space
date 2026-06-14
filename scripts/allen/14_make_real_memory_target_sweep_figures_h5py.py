#!/usr/bin/env python3
"""
Make figures for real Allen memory target-variable sweep.

Inputs:
    real_memory_target_sweep_target_summary.csv
    real_memory_target_sweep_group_summary.csv

Outputs:
    PNG figures plus target_sweep_figure_manifest.csv
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


def pretty_target(x):
    return (
        x.replace("population_", "")
        .replace("_rate_hz", " rate")
        .replace("_", " ")
    )


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = pd.read_csv(results_dir / "real_memory_target_sweep_target_summary.csv")
    group = pd.read_csv(results_dir / "real_memory_target_sweep_group_summary.csv")

    target = target.sort_values("rank_by_controlled_memory").copy()
    target["target_label"] = target["target"].map(pretty_target)

    ok = group[group["status"] == "ok"].copy()
    ok["target_label"] = ok["target"].map(pretty_target)
    ok["label"] = (
        ok["region"]
        + " / "
        + ok["stimulus_family"].str.replace("_", " ", regex=False)
    )

    manifest = []

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(target))
    ax.bar(x, target["median_observed_minus_shuffle"].to_numpy(dtype=float))
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(target["target_label"], rotation=30, ha="right")
    ax.set_ylabel("Median observed minus shuffled gain")
    ax.set_title("Controlled memory strength by target variable")
    ax.grid(True, axis="y", linewidth=0.3)
    p = out_dir / "real_allen_target_sweep_ranked_controlled_memory.png"
    savefig(fig, p)
    manifest.append({
        "figure": p.name,
        "description": "Ranked target variables by median observed-minus-shuffled memory gain.",
        "source": "real_memory_target_sweep_target_summary.csv",
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(target))
    ax.bar(x, target["median_observed_gain"].to_numpy(dtype=float), label="Observed")
    ax.scatter(x, target["median_shuffle_gain_mean"].to_numpy(dtype=float), marker="o", label="Shuffle mean")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(target["target_label"], rotation=30, ha="right")
    ax.set_ylabel("Median memory gain, ΔR²")
    ax.set_title("Observed target gain against shuffled-lag null")
    ax.legend()
    ax.grid(True, axis="y", linewidth=0.3)
    p = out_dir / "real_allen_target_sweep_observed_vs_shuffle.png"
    savefig(fig, p)
    manifest.append({
        "figure": p.name,
        "description": "Observed median memory gain compared with median shuffled-lag gain for each target.",
        "source": "real_memory_target_sweep_target_summary.csv",
    })

    top = ok.sort_values("observed_minus_shuffle_mean", ascending=False).head(20).copy()
    top["combined"] = top["target_label"] + " | " + top["label"]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(top))
    ax.barh(y, top["observed_minus_shuffle_mean"].to_numpy(dtype=float))
    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["combined"])
    ax.invert_yaxis()
    ax.set_xlabel("Observed gain minus shuffled-lag mean")
    ax.set_title("Top controlled memory effects across targets")
    ax.grid(True, axis="x", linewidth=0.3)
    p = out_dir / "real_allen_target_sweep_top_group_effects.png"
    savefig(fig, p)
    manifest.append({
        "figure": p.name,
        "description": "Top region-stimulus-target combinations ranked by observed-minus-shuffled memory gain.",
        "source": "real_memory_target_sweep_group_summary.csv",
    })

    pivot = ok.pivot_table(
        index="target_label",
        columns="region",
        values="observed_minus_shuffle_mean",
        aggfunc="median",
    )
    pivot = pivot.reindex(index=target["target_label"])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Region")
    ax.set_ylabel("Target")
    ax.set_title("Controlled memory by target and region")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Observed minus shuffled gain")
    p = out_dir / "real_allen_target_region_controlled_memory_heatmap.png"
    savefig(fig, p)
    manifest.append({
        "figure": p.name,
        "description": "Heatmap of controlled memory strength by target variable and brain region.",
        "source": "real_memory_target_sweep_group_summary.csv",
    })

    manifest_df = pd.DataFrame(manifest)
    manifest_df["run_note"] = (
        "Target sweep shows strongest controlled memory for entropy and active-unit fraction. "
        "Population state speed is not supported in this run."
    )
    manifest_path = out_dir / "target_sweep_figure_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"[ok] wrote figures to {out_dir}")
    print(f"[ok] wrote {manifest_path}")
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
