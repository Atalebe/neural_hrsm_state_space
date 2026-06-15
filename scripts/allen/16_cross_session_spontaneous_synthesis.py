#!/usr/bin/env python3
"""
Cross-session synthesis for spontaneous-focused real Allen Neural HRSM.

Combines spontaneous-focused outputs from multiple Allen Neuropixels sessions.

Inputs per session:
    results/real_allen/<session_run>/
        real_orthogonality_audit.csv
        real_neural_hrsm_domain_summary.csv
        real_memory_kernel_pooled_summary.csv
        real_memory_negative_control_summary.csv
        real_memory_negative_control_pooled_summary.csv
        real_memory_target_sweep_target_summary.csv
        real_memory_target_sweep_group_summary.csv

Outputs:
    results/cross_session/allen_spontaneous_v1/*.csv
    results/figures/cross_session/allen_spontaneous_v1/*.png

Scientific role:
This synthesis tests whether the spontaneous-focused Neural HRSM memory result
replicates across Allen sessions, without changing the analysis logic.
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SESSION_DIRS = [
    "results/real_allen/session_715093703_spontaneous_v1",
    "results/real_allen/session_719161530_spontaneous_v1",
]

REGION_ORDER = ["VISp", "VISl", "LGd", "CA1"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dirs", nargs="+", default=DEFAULT_SESSION_DIRS)
    parser.add_argument(
        "--out-dir",
        default="results/cross_session/allen_spontaneous_v1",
    )
    parser.add_argument(
        "--fig-dir",
        default="results/figures/cross_session/allen_spontaneous_v1",
    )
    return parser.parse_args()


def session_id_from_path(path):
    m = re.search(r"session_(\d+)", str(path))
    if not m:
        return Path(path).name
    return m.group(1)


def read_csv(session_dir, name):
    path = Path(session_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_session(session_dir):
    session_dir = Path(session_dir)
    sid = session_id_from_path(session_dir)

    audit = read_csv(session_dir, "real_orthogonality_audit.csv")
    domain = read_csv(session_dir, "real_neural_hrsm_domain_summary.csv")
    kernel_pooled = read_csv(session_dir, "real_memory_kernel_pooled_summary.csv")
    control = read_csv(session_dir, "real_memory_negative_control_summary.csv")
    control_pooled = read_csv(session_dir, "real_memory_negative_control_pooled_summary.csv")
    target = read_csv(session_dir, "real_memory_target_sweep_target_summary.csv")
    target_group = read_csv(session_dir, "real_memory_target_sweep_group_summary.csv")

    for df in [audit, domain, kernel_pooled, control, control_pooled, target, target_group]:
        df["session_id_synthesis"] = sid

    return {
        "session_dir": str(session_dir),
        "session_id": sid,
        "audit": audit,
        "domain": domain,
        "kernel_pooled": kernel_pooled,
        "control": control,
        "control_pooled": control_pooled,
        "target": target,
        "target_group": target_group,
    }


def build_session_summary(sessions):
    rows = []

    for s in sessions:
        sid = s["session_id"]
        audit = s["audit"].iloc[0].to_dict()
        domain = s["domain"].copy()
        kernel = s["kernel_pooled"].iloc[0].to_dict()
        control = s["control_pooled"].iloc[0].to_dict()
        target = s["target"].copy()

        top_target = target.sort_values("rank_by_controlled_memory").iloc[0]

        best_phi = domain.sort_values("Phi_neural", ascending=False).iloc[0]

        rows.append(
            {
                "session_id": sid,
                "n_hrsm_rows": audit.get("n_rows", np.nan),
                "max_abs_offdiag_corr_orthogonalized": audit.get(
                    "max_abs_offdiag_corr_orthogonalized", np.nan
                ),
                "best_phi_region": best_phi["region"],
                "best_phi_value": best_phi["Phi_neural"],
                "best_phi_domain": best_phi["hrsm_domain"],
                "median_memory_gain_r2": kernel.get("median_memory_gain_r2", np.nan),
                "mean_memory_gain_r2": kernel.get("mean_memory_gain_r2", np.nan),
                "fraction_positive_gain": kernel.get("fraction_positive_gain", np.nan),
                "median_observed_minus_shuffle": control.get(
                    "median_observed_minus_shuffle", np.nan
                ),
                "mean_observed_minus_shuffle": control.get(
                    "mean_observed_minus_shuffle", np.nan
                ),
                "fraction_observed_above_shuffle_mean": control.get(
                    "fraction_observed_above_shuffle_mean", np.nan
                ),
                "median_empirical_p": control.get("median_empirical_p", np.nan),
                "top_target": top_target["target"],
                "top_target_median_observed_minus_shuffle": top_target[
                    "median_observed_minus_shuffle"
                ],
                "top_target_fraction_above_shuffle": top_target[
                    "fraction_observed_above_shuffle_mean"
                ],
            }
        )

    return pd.DataFrame(rows)


def build_region_summary(sessions):
    rows = []

    for s in sessions:
        sid = s["session_id"]

        domain = s["domain"].copy()
        control = s["control"].copy()

        merged = pd.merge(
            control,
            domain[
                [
                    "session_id",
                    "region",
                    "stimulus_family",
                    "H",
                    "R",
                    "S",
                    "M",
                    "Phi_neural",
                    "hrsm_domain",
                ]
            ],
            on=["session_id", "region", "stimulus_family"],
            how="left",
        )

        merged["session_id_synthesis"] = sid
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["region", "session_id_synthesis"])
    return out


def build_target_summary(sessions):
    rows = []
    group_rows = []

    for s in sessions:
        sid = s["session_id"]

        target = s["target"].copy()
        target["session_id_synthesis"] = sid
        rows.append(target)

        group = s["target_group"].copy()
        group["session_id_synthesis"] = sid
        group_rows.append(group)

    target_all = pd.concat(rows, ignore_index=True)
    group_all = pd.concat(group_rows, ignore_index=True)

    agg = (
        target_all.groupby("target")
        .agg(
            n_sessions=("session_id_synthesis", "nunique"),
            median_rank=("rank_by_controlled_memory", "median"),
            mean_rank=("rank_by_controlled_memory", "mean"),
            median_observed_gain=("median_observed_gain", "median"),
            mean_observed_gain=("mean_observed_gain", "mean"),
            median_observed_minus_shuffle=("median_observed_minus_shuffle", "median"),
            mean_observed_minus_shuffle=("mean_observed_minus_shuffle", "mean"),
            min_fraction_above_shuffle=(
                "fraction_observed_above_shuffle_mean",
                "min",
            ),
            median_empirical_p=("median_empirical_p", "median"),
        )
        .reset_index()
        .sort_values(
            [
                "median_rank",
                "median_observed_minus_shuffle",
                "mean_observed_minus_shuffle",
            ],
            ascending=[True, False, False],
        )
    )

    agg["cross_session_rank"] = np.arange(1, len(agg) + 1)
    return target_all, group_all, agg


def build_replication_flags(session_summary, region_summary, target_agg):
    rows = []

    rows.append(
        {
            "test": "all_sessions_memory_gain_positive",
            "passed": bool((session_summary["fraction_positive_gain"] == 1.0).all()),
            "detail": "Every session has positive mean-rate memory gain in all spontaneous regions.",
        }
    )

    rows.append(
        {
            "test": "all_sessions_above_shuffle",
            "passed": bool(
                (session_summary["fraction_observed_above_shuffle_mean"] == 1.0).all()
            ),
            "detail": "Every session has observed memory gain above shuffled-lag mean in all spontaneous regions.",
        }
    )

    rows.append(
        {
            "test": "orthogonality_numeric_precision",
            "passed": bool(
                (session_summary["max_abs_offdiag_corr_orthogonalized"] < 1e-12).all()
            ),
            "detail": "All sessions have orthogonalized HRSM axes decorrelated to numerical precision.",
        }
    )

    top_two = target_agg.sort_values("cross_session_rank").head(2)["target"].tolist()
    rows.append(
        {
            "test": "top_targets_are_recruitment_and_entropy",
            "passed": set(top_two) == {"active_unit_fraction", "population_rate_entropy"},
            "detail": "The top two cross-session targets are active-unit fraction and population-rate entropy.",
        }
    )

    lgd_best = (
        region_summary.sort_values(["session_id_synthesis", "Phi_neural"], ascending=[True, False])
        .groupby("session_id_synthesis")
        .head(1)
    )
    rows.append(
        {
            "test": "LGd_best_phi_in_each_session",
            "passed": bool((lgd_best["region"] == "LGd").all()),
            "detail": "LGd has the highest Phi_neural among spontaneous regions in each session.",
        }
    )

    return pd.DataFrame(rows)


def plot_session_memory(session_summary, fig_dir):
    df = session_summary.copy()

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df))
    ax.bar(x, df["median_observed_minus_shuffle"].to_numpy(dtype=float))
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["session_id"])
    ax.set_ylabel("Median observed minus shuffled gain")
    ax.set_title("Spontaneous memory control replicated across sessions")
    ax.grid(True, axis="y", linewidth=0.3)

    savefig(fig, fig_dir / "cross_session_spontaneous_memory_control.png")


def plot_region_memory(region_summary, fig_dir):
    df = region_summary.copy()
    df["label"] = df["session_id_synthesis"].astype(str) + " / " + df["region"].astype(str)
    df = df.sort_values(["region", "session_id_synthesis"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax.bar(x, df["observed_minus_shuffle_mean"].to_numpy(dtype=float))
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.set_ylabel("Observed minus shuffled gain")
    ax.set_title("Region-level memory control across sessions")
    ax.grid(True, axis="y", linewidth=0.3)

    savefig(fig, fig_dir / "cross_session_region_memory_control.png")


def plot_target_heatmap(target_all, fig_dir):
    df = target_all.copy()
    pivot = df.pivot_table(
        index="target",
        columns="session_id_synthesis",
        values="median_observed_minus_shuffle",
        aggfunc="median",
    )

    order = (
        df.groupby("target")["rank_by_controlled_memory"]
        .median()
        .sort_values()
        .index.tolist()
    )
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Session")
    ax.set_ylabel("Target")
    ax.set_title("Controlled memory by target across sessions")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median observed minus shuffled gain")
    savefig(fig, fig_dir / "cross_session_target_memory_heatmap.png")


def plot_phi_by_region(region_summary, fig_dir):
    df = region_summary.copy()
    df["label"] = df["session_id_synthesis"].astype(str) + " / " + df["region"].astype(str)
    df = df.sort_values(["region", "session_id_synthesis"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax.bar(x, df["Phi_neural"].to_numpy(dtype=float))
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.set_ylabel("Phi_neural")
    ax.set_title("Spontaneous HRSM potential across sessions")
    ax.grid(True, axis="y", linewidth=0.3)

    savefig(fig, fig_dir / "cross_session_phi_by_region.png")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sessions = [load_session(d) for d in args.session_dirs]

    session_summary = build_session_summary(sessions)
    region_summary = build_region_summary(sessions)
    target_all, target_group_all, target_agg = build_target_summary(sessions)
    flags = build_replication_flags(session_summary, region_summary, target_agg)

    session_summary.to_csv(out_dir / "cross_session_spontaneous_session_summary.csv", index=False)
    region_summary.to_csv(out_dir / "cross_session_spontaneous_region_memory_summary.csv", index=False)
    target_all.to_csv(out_dir / "cross_session_spontaneous_target_by_session.csv", index=False)
    target_group_all.to_csv(out_dir / "cross_session_spontaneous_target_group_summary.csv", index=False)
    target_agg.to_csv(out_dir / "cross_session_spontaneous_target_synthesis.csv", index=False)
    flags.to_csv(out_dir / "cross_session_spontaneous_replication_flags.csv", index=False)

    plot_session_memory(session_summary, fig_dir)
    plot_region_memory(region_summary, fig_dir)
    plot_target_heatmap(target_all, fig_dir)
    plot_phi_by_region(region_summary, fig_dir)

    manifest = pd.DataFrame(
        [
            {
                "figure": "cross_session_spontaneous_memory_control.png",
                "description": "Session-level median observed-minus-shuffled memory gain for spontaneous activity.",
                "source": "cross_session_spontaneous_session_summary.csv",
            },
            {
                "figure": "cross_session_region_memory_control.png",
                "description": "Region-level observed-minus-shuffled memory gain across spontaneous-focused sessions.",
                "source": "cross_session_spontaneous_region_memory_summary.csv",
            },
            {
                "figure": "cross_session_target_memory_heatmap.png",
                "description": "Target-level controlled memory strength across sessions.",
                "source": "cross_session_spontaneous_target_by_session.csv",
            },
            {
                "figure": "cross_session_phi_by_region.png",
                "description": "Spontaneous HRSM Phi_neural by region across sessions.",
                "source": "cross_session_spontaneous_region_memory_summary.csv",
            },
        ]
    )
    manifest.to_csv(fig_dir / "cross_session_figure_manifest.csv", index=False)

    print(f"[ok] wrote synthesis tables to {out_dir}")
    print(f"[ok] wrote figures to {fig_dir}")
    print()
    print("SESSION SUMMARY")
    print(session_summary.to_string(index=False))
    print()
    print("TARGET SYNTHESIS")
    print(target_agg.to_string(index=False))
    print()
    print("REPLICATION FLAGS")
    print(flags.to_string(index=False))


if __name__ == "__main__":
    main()
