#!/usr/bin/env python3
"""
Variance-scaling audit for spontaneous-focused real Allen Neural HRSM.

Question:
Does the controlled memory signal merely scale with target variance,
target dispersion, or simple temporal autocorrelation?

This script joins:
  - population_state_matrix.csv from each spontaneous-focused session
  - real_memory_target_sweep_group_summary.csv from each session

It writes group-level variance diagnostics, correlation summaries, residualized
target rankings, replication flags, and simple figures.
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SESSIONS = [
    "715093703",
    "719161530",
    "750749662",
]

DEFAULT_TARGETS = [
    "population_mean_rate_hz",
    "population_std_rate_hz",
    "active_unit_fraction",
    "population_l2_rate_norm",
    "population_rate_entropy",
    "population_state_speed",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--processed-root", default="data/processed/allen_neuropixels_real")
    p.add_argument("--results-root", default="results/real_allen")
    p.add_argument("--out-dir", default="results/cross_session/allen_spontaneous_v1")
    p.add_argument("--fig-dir", default="results/figures/cross_session/allen_spontaneous_v1")
    return p.parse_args()


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    if method == "spearman":
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        return float(np.corrcoef(xr, yr)[0, 1])
    return float(np.corrcoef(x, y)[0, 1])


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def lag1_autocorr(group, target, trial_col=None, time_col=None):
    g = group.copy()

    if trial_col is None:
        if time_col is not None:
            g = g.sort_values(time_col)
        vals = g[target].to_numpy(dtype=float)
        if len(vals) < 3:
            return np.nan
        return safe_corr(vals[:-1], vals[1:])

    xs = []
    ys = []
    sort_cols = [trial_col]
    if time_col is not None:
        sort_cols.append(time_col)
    g = g.sort_values(sort_cols)

    for _, sub in g.groupby(trial_col):
        vals = sub[target].to_numpy(dtype=float)
        if len(vals) >= 3:
            xs.extend(vals[:-1])
            ys.extend(vals[1:])

    if len(xs) < 3:
        return np.nan
    return safe_corr(xs, ys)


def session_paths(session, processed_root, results_root):
    run = f"session_{session}_spontaneous_v1"
    population = Path(processed_root) / run / "population_state_matrix.csv"
    target_group = Path(results_root) / run / "real_memory_target_sweep_group_summary.csv"
    return run, population, target_group


def compute_population_stats(session, population_path, targets):
    pop = pd.read_csv(population_path)

    trial_col = find_col(
        pop,
        [
            "stimulus_presentation_id",
            "presentation_id",
            "stimulus_id",
            "trial_id",
            "presentation_index",
            "stimulus_block",
        ],
    )
    time_col = find_col(
        pop,
        [
            "bin_start_time",
            "bin_start",
            "time",
            "t",
            "bin_index",
            "bin_id",
            "row_index",
        ],
    )

    rows = []
    for region, rg in pop.groupby("region"):
        for target in targets:
            if target not in rg.columns:
                continue

            vals = rg[target].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue

            q25, q75 = np.percentile(vals, [25, 75])
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            var = float(np.var(vals, ddof=1)) if len(vals) > 1 else np.nan
            iqr = float(q75 - q25)
            cv_abs = float(sd / (abs(mean) + 1e-12))

            rows.append(
                {
                    "session_id_synthesis": str(session),
                    "region": region,
                    "stimulus_family": "spontaneous",
                    "target": target,
                    "n_rows": int(len(vals)),
                    "target_mean": mean,
                    "target_sd": sd,
                    "target_var": var,
                    "target_iqr": iqr,
                    "target_cv_abs": cv_abs,
                    "target_min": float(np.min(vals)),
                    "target_max": float(np.max(vals)),
                    "target_lag1_autocorr": lag1_autocorr(rg, target, trial_col, time_col),
                    "trial_col_used": trial_col if trial_col is not None else "",
                    "time_col_used": time_col if time_col is not None else "",
                }
            )

    return pd.DataFrame(rows)


def load_memory_group(session, target_group_path):
    df = pd.read_csv(target_group_path)
    df = df[df["status"] == "ok"].copy()
    df["session_id_synthesis"] = str(session)
    keep = [
        "session_id_synthesis",
        "region",
        "stimulus_family",
        "target",
        "observed_memory_gain_r2",
        "shuffle_gain_mean",
        "observed_minus_shuffle_mean",
        "empirical_p_shuffle_ge_observed",
        "fraction_shuffle_positive",
        "n_valid_shuffles",
    ]
    return df[keep]


def zscore_matrix(X):
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def residualize(df):
    out = df.copy()
    y = out["observed_minus_shuffle_mean"].to_numpy(dtype=float)

    X = np.column_stack(
        [
            np.log1p(np.maximum(out["target_var"].to_numpy(dtype=float), 0.0)),
            out["target_lag1_autocorr"].fillna(0.0).to_numpy(dtype=float),
            np.log1p(out["n_rows"].to_numpy(dtype=float)),
        ]
    )
    X = zscore_matrix(X)
    X = np.column_stack([np.ones(len(X)), X])

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    resid = np.full(len(out), np.nan)

    if mask.sum() >= X.shape[1] + 2:
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        pred = X @ beta
        resid = y - pred

    out["controlled_gain_residual_variance_autocorr"] = resid
    return out


def make_correlations(group_metrics):
    rows = []
    metrics = [
        "target_sd",
        "target_var",
        "target_iqr",
        "target_cv_abs",
        "target_lag1_autocorr",
    ]

    for target, sub in group_metrics.groupby("target"):
        for m in metrics:
            rows.append(
                {
                    "scope": "within_target",
                    "target": target,
                    "scale_metric": m,
                    "n": int(sub[[m, "observed_minus_shuffle_mean"]].dropna().shape[0]),
                    "pearson_r": safe_corr(sub[m], sub["observed_minus_shuffle_mean"], "pearson"),
                    "spearman_r": safe_corr(sub[m], sub["observed_minus_shuffle_mean"], "spearman"),
                }
            )

    for m in metrics:
        rows.append(
            {
                "scope": "all_targets",
                "target": "ALL",
                "scale_metric": m,
                "n": int(group_metrics[[m, "observed_minus_shuffle_mean"]].dropna().shape[0]),
                "pearson_r": safe_corr(group_metrics[m], group_metrics["observed_minus_shuffle_mean"], "pearson"),
                "spearman_r": safe_corr(group_metrics[m], group_metrics["observed_minus_shuffle_mean"], "spearman"),
            }
        )

    return pd.DataFrame(rows)


def make_target_residual_summary(group_metrics):
    agg = (
        group_metrics.groupby("target")
        .agg(
            n_groups=("observed_minus_shuffle_mean", "size"),
            median_observed_minus_shuffle=("observed_minus_shuffle_mean", "median"),
            mean_observed_minus_shuffle=("observed_minus_shuffle_mean", "mean"),
            median_residual_gain=("controlled_gain_residual_variance_autocorr", "median"),
            mean_residual_gain=("controlled_gain_residual_variance_autocorr", "mean"),
            median_target_var=("target_var", "median"),
            median_lag1_autocorr=("target_lag1_autocorr", "median"),
        )
        .reset_index()
        .sort_values(
            ["median_observed_minus_shuffle", "mean_observed_minus_shuffle"],
            ascending=[False, False],
        )
    )
    agg["raw_controlled_rank"] = np.arange(1, len(agg) + 1)

    agg = agg.sort_values(
        ["median_residual_gain", "mean_residual_gain"],
        ascending=[False, False],
    )
    agg["variance_residual_rank"] = np.arange(1, len(agg) + 1)

    return agg.sort_values("raw_controlled_rank")


def make_flags(target_summary, corr_summary):
    top_two_raw = set(
        target_summary.sort_values("raw_controlled_rank").head(2)["target"].tolist()
    )
    top_two_resid = set(
        target_summary.sort_values("variance_residual_rank").head(2)["target"].tolist()
    )

    desired = {"active_unit_fraction", "population_rate_entropy"}

    all_target_var_corr = corr_summary[
        (corr_summary["scope"] == "all_targets") & (corr_summary["scale_metric"] == "target_var")
    ]["spearman_r"].iloc[0]

    rows = [
        {
            "test": "top_two_raw_are_recruitment_and_entropy",
            "passed": top_two_raw == desired,
            "detail": f"Top two raw controlled targets: {sorted(top_two_raw)}",
        },
        {
            "test": "top_two_residual_are_recruitment_and_entropy",
            "passed": top_two_resid == desired,
            "detail": f"Top two variance/autocorr residual targets: {sorted(top_two_resid)}",
        },
        {
            "test": "overall_variance_correlation_not_extreme",
            "passed": bool(abs(all_target_var_corr) < 0.85) if np.isfinite(all_target_var_corr) else False,
            "detail": f"Overall Spearman(target_var, controlled_gain) = {all_target_var_corr}",
        },
    ]
    return pd.DataFrame(rows)


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gain_vs_variance(df, fig_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for target, sub in df.groupby("target"):
        ax.scatter(
            np.log1p(sub["target_var"]),
            sub["observed_minus_shuffle_mean"],
            label=target,
            s=35,
            alpha=0.85,
        )
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("log(1 + target variance)")
    ax.set_ylabel("Observed minus shuffled memory gain")
    ax.set_title("Variance scaling audit")
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=7, loc="best")
    savefig(fig, fig_dir / "cross_session_variance_scaling_gain_vs_variance.png")


def plot_raw_vs_residual_rank(summary, fig_dir):
    s = summary.sort_values("raw_controlled_rank")
    x = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, s["median_observed_minus_shuffle"], width=0.4, label="Raw controlled gain")
    ax.bar(x + 0.2, s["median_residual_gain"], width=0.4, label="Residual gain")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(s["target"], rotation=35, ha="right")
    ax.set_ylabel("Median gain")
    ax.set_title("Target ranking before and after variance/autocorrelation residualization")
    ax.grid(True, axis="y", linewidth=0.3)
    ax.legend()
    savefig(fig, fig_dir / "cross_session_variance_residual_target_rank.png")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pop_stats = []
    memory = []

    for session in args.sessions:
        run, pop_path, target_group_path = session_paths(
            session, args.processed_root, args.results_root
        )
        if not pop_path.exists():
            raise FileNotFoundError(pop_path)
        if not target_group_path.exists():
            raise FileNotFoundError(target_group_path)

        pop_stats.append(compute_population_stats(session, pop_path, args.targets))
        memory.append(load_memory_group(session, target_group_path))

    pop_stats = pd.concat(pop_stats, ignore_index=True)
    memory = pd.concat(memory, ignore_index=True)

    group_metrics = pd.merge(
        memory,
        pop_stats,
        on=["session_id_synthesis", "region", "stimulus_family", "target"],
        how="left",
    )

    group_metrics = residualize(group_metrics)
    corr_summary = make_correlations(group_metrics)
    target_summary = make_target_residual_summary(group_metrics)
    flags = make_flags(target_summary, corr_summary)

    group_metrics.to_csv(out_dir / "variance_scaling_group_metrics.csv", index=False)
    corr_summary.to_csv(out_dir / "variance_scaling_correlation_summary.csv", index=False)
    target_summary.to_csv(out_dir / "variance_scaling_target_residual_summary.csv", index=False)
    flags.to_csv(out_dir / "variance_scaling_flags.csv", index=False)

    plot_gain_vs_variance(group_metrics, fig_dir)
    plot_raw_vs_residual_rank(target_summary, fig_dir)

    manifest = pd.DataFrame(
        [
            {
                "figure": "cross_session_variance_scaling_gain_vs_variance.png",
                "description": "Controlled memory gain plotted against target variance across sessions, regions, and targets.",
                "source": "variance_scaling_group_metrics.csv",
            },
            {
                "figure": "cross_session_variance_residual_target_rank.png",
                "description": "Target controlled-memory ranking before and after residualizing variance and lag-1 autocorrelation.",
                "source": "variance_scaling_target_residual_summary.csv",
            },
        ]
    )
    manifest.to_csv(fig_dir / "variance_scaling_figure_manifest.csv", index=False)

    print("[ok] wrote variance-scaling outputs")
    print()
    print("TARGET RESIDUAL SUMMARY")
    print(target_summary.to_string(index=False))
    print()
    print("FLAGS")
    print(flags.to_string(index=False))


if __name__ == "__main__":
    main()
