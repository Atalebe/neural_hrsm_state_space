#!/usr/bin/env python3
"""
Lag-ablation grid for spontaneous-focused real Allen Neural HRSM.

Runs the existing target-sweep memory audit over multiple lag sets:
    lag1, lag2, lag12, lag123, lag1234

This tests whether the memory result depends narrowly on the chosen lag pair
or whether the recruitment/entropy effect persists across lag choices.
"""

from pathlib import Path
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SESSIONS = ["715093703", "719161530", "750749662"]

DEFAULT_TARGETS = [
    "active_unit_fraction",
    "population_rate_entropy",
    "population_mean_rate_hz",
    "population_state_speed",
]

DEFAULT_LAGSETS = {
    "lag1": [1],
    "lag2": [2],
    "lag12": [1, 2],
    "lag123": [1, 2, 3],
    "lag1234": [1, 2, 3, 4],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--n-shuffles", type=int, default=50)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--min-supervised-rows", type=int, default=200)
    p.add_argument("--processed-root", default="data/processed/allen_neuropixels_real")
    p.add_argument("--out-root", default="results/ablation/allen_spontaneous_v1")
    p.add_argument("--fig-dir", default="results/figures/ablation/allen_spontaneous_v1")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def run_one(args, session, lag_label, lags):
    pop = Path(args.processed_root) / f"session_{session}_spontaneous_v1" / "population_state_matrix.csv"
    out_dir = Path(args.out_root) / f"session_{session}" / lag_label

    if not pop.exists():
        raise FileNotFoundError(pop)

    target_summary = out_dir / "real_memory_target_sweep_target_summary.csv"
    if args.skip_existing and target_summary.exists():
        print(f"[skip] {session} {lag_label}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "scripts/allen/13_real_memory_target_sweep_h5py.py",
        "--population", str(pop),
        "--out-dir", str(out_dir),
        "--targets", *args.targets,
        "--lags", *[str(x) for x in lags],
        "--alpha", str(args.alpha),
        "--min-supervised-rows", str(args.min_supervised_rows),
        "--n-shuffles", str(args.n_shuffles),
        "--seed", str(600000 + int(session[-3:]) + len(lags) * 100),
    ]

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def collect(args):
    target_rows = []
    group_rows = []

    for session in args.sessions:
        for lag_label, lags in DEFAULT_LAGSETS.items():
            out_dir = Path(args.out_root) / f"session_{session}" / lag_label

            t = pd.read_csv(out_dir / "real_memory_target_sweep_target_summary.csv")
            g = pd.read_csv(out_dir / "real_memory_target_sweep_group_summary.csv")

            t["session_id_synthesis"] = str(session)
            t["lag_label"] = lag_label
            t["lags"] = ",".join(map(str, lags))
            t["n_lags"] = len(lags)

            g["session_id_synthesis"] = str(session)
            g["lag_label"] = lag_label
            g["lags"] = ",".join(map(str, lags))
            g["n_lags"] = len(lags)

            target_rows.append(t)
            group_rows.append(g)

    target_all = pd.concat(target_rows, ignore_index=True)
    group_all = pd.concat(group_rows, ignore_index=True)

    agg = (
        target_all.groupby(["target", "lag_label", "lags", "n_lags"])
        .agg(
            n_sessions=("session_id_synthesis", "nunique"),
            median_rank=("rank_by_controlled_memory", "median"),
            mean_rank=("rank_by_controlled_memory", "mean"),
            median_observed_gain=("median_observed_gain", "median"),
            mean_observed_gain=("mean_observed_gain", "mean"),
            median_observed_minus_shuffle=("median_observed_minus_shuffle", "median"),
            mean_observed_minus_shuffle=("mean_observed_minus_shuffle", "mean"),
            min_fraction_above_shuffle=("fraction_observed_above_shuffle_mean", "min"),
            median_empirical_p=("median_empirical_p", "median"),
        )
        .reset_index()
    )

    lag_order = {k: i for i, k in enumerate(DEFAULT_LAGSETS.keys())}
    agg["lag_order"] = agg["lag_label"].map(lag_order)
    agg = agg.sort_values(["target", "lag_order"])

    return target_all, group_all, agg


def flags(agg):
    lag12 = agg[agg["lag_label"] == "lag12"].copy()
    top_two_lag12 = set(
        lag12.sort_values(
            ["median_observed_minus_shuffle", "mean_observed_minus_shuffle"],
            ascending=[False, False],
        )
        .head(2)["target"]
        .tolist()
    )

    desired = {"active_unit_fraction", "population_rate_entropy"}

    recruitment_entropy = agg[agg["target"].isin(desired)].copy()
    all_positive = bool((recruitment_entropy["median_observed_minus_shuffle"] > 0).all())

    rows = [
        {
            "test": "lag12_top_two_are_recruitment_and_entropy",
            "passed": top_two_lag12 == desired,
            "detail": f"lag12 top two targets: {sorted(top_two_lag12)}",
        },
        {
            "test": "recruitment_entropy_positive_all_lagsets",
            "passed": all_positive,
            "detail": "Active-unit fraction and entropy have positive median controlled gain across all tested lag sets.",
        },
    ]
    return pd.DataFrame(rows)


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_lag_curves(agg, fig_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    lag_order = list(DEFAULT_LAGSETS.keys())
    x_map = {lab: i for i, lab in enumerate(lag_order)}

    for target, sub in agg.groupby("target"):
        sub = sub.sort_values("lag_order")
        x = [x_map[v] for v in sub["lag_label"]]
        y = sub["median_observed_minus_shuffle"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", label=target)

    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(np.arange(len(lag_order)))
    ax.set_xticklabels(lag_order)
    ax.set_xlabel("Lag set")
    ax.set_ylabel("Median observed minus shuffled gain")
    ax.set_title("Lag-ablation controlled memory curves")
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=8)
    savefig(fig, fig_dir / "lag_ablation_controlled_memory_curves.png")


def plot_rank_heatmap(agg, fig_dir):
    pivot = agg.pivot_table(
        index="target",
        columns="lag_label",
        values="median_rank",
        aggfunc="median",
    )
    lag_order = list(DEFAULT_LAGSETS.keys())
    pivot = pivot.reindex(columns=lag_order)

    target_order = (
        agg.groupby("target")["median_rank"].median().sort_values().index.tolist()
    )
    pivot = pivot.reindex(index=target_order)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Target rank stability across lag ablations")
    ax.set_xlabel("Lag set")
    ax.set_ylabel("Target")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median rank")
    savefig(fig, fig_dir / "lag_ablation_target_rank_heatmap.png")


def main():
    args = parse_args()

    for session in args.sessions:
        for lag_label, lags in DEFAULT_LAGSETS.items():
            run_one(args, session, lag_label, lags)

    out_root = Path(args.out_root)
    fig_dir = Path(args.fig_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    target_all, group_all, agg = collect(args)
    fl = flags(agg)

    target_all.to_csv(out_root / "lag_ablation_target_by_session.csv", index=False)
    group_all.to_csv(out_root / "lag_ablation_group_by_session.csv", index=False)
    agg.to_csv(out_root / "lag_ablation_cross_session_target_summary.csv", index=False)
    fl.to_csv(out_root / "lag_ablation_flags.csv", index=False)

    plot_lag_curves(agg, fig_dir)
    plot_rank_heatmap(agg, fig_dir)

    manifest = pd.DataFrame(
        [
            {
                "figure": "lag_ablation_controlled_memory_curves.png",
                "description": "Cross-session controlled memory strength across lag sets.",
                "source": "lag_ablation_cross_session_target_summary.csv",
            },
            {
                "figure": "lag_ablation_target_rank_heatmap.png",
                "description": "Target rank stability across lag ablations.",
                "source": "lag_ablation_cross_session_target_summary.csv",
            },
        ]
    )
    manifest.to_csv(fig_dir / "lag_ablation_figure_manifest.csv", index=False)

    print("[ok] wrote lag-ablation outputs")
    print()
    print("LAG ABLATION SUMMARY")
    print(agg.to_string(index=False))
    print()
    print("FLAGS")
    print(fl.to_string(index=False))


if __name__ == "__main__":
    main()
