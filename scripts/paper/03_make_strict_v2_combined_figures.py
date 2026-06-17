#!/usr/bin/env python3
"""
Create combined-session manuscript figures for the strict-v2 Allen Neural HRSM paper.

This script is intentionally separate from the per-session diagnostic plotting scripts.
It makes the visual story for the paper:

1. Session-level controlled memory overview
2. Target-by-session controlled memory heatmap
3. Controlled gain distribution across sessions/regions
4. Raw versus orthogonalized HRSM axis correlations
5. Variance/autocorrelation residual comparison
6. Lag-ablation controlled-memory curves
7. Combined Phi-vs-organizational-memory landscape
8. Region-level HRSM profiles across strict sessions

Inputs are the committed strict-v2 CSV outputs.
Outputs are manuscript-ready PNGs and a figure manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRICT_SESSIONS = [
    "715093703",
    "719161530",
    "750749662",
    "751348571",
    "755434585",
    "756029989",
]

REGION_ORDER = ["CA1", "LGd", "VISl", "VISp"]

TARGET_ORDER = [
    "active_unit_fraction",
    "population_rate_entropy",
    "population_mean_rate_hz",
    "population_l2_rate_norm",
    "population_std_rate_hz",
    "population_state_speed",
]

ABLATION_TARGETS = [
    "active_unit_fraction",
    "population_rate_entropy",
    "population_mean_rate_hz",
    "population_state_speed",
]

TARGET_LABELS = {
    "active_unit_fraction": "Active-unit fraction",
    "population_rate_entropy": "Rate entropy",
    "population_mean_rate_hz": "Mean rate",
    "population_l2_rate_norm": "L2 rate norm",
    "population_std_rate_hz": "Rate SD",
    "population_state_speed": "State speed",
}

AXES = ["H", "R", "S", "M"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--strict-dir",
        default="results/cross_session/allen_spontaneous_strict_v2",
        help="Strict-v2 cross-session output directory.",
    )
    p.add_argument(
        "--ablation-dir",
        default="results/ablation/allen_spontaneous_strict_v2",
        help="Strict-v2 lag-ablation output directory.",
    )
    p.add_argument(
        "--session-root",
        default="results/real_allen",
        help="Directory containing per-session result folders.",
    )
    p.add_argument(
        "--out-dir",
        default="results/figures/manuscript/allen_spontaneous_strict_v2",
        help="Output directory for manuscript figures.",
    )
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt_target(t: str) -> str:
    return TARGET_LABELS.get(t, str(t).replace("_", " "))


def ensure_session_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Different strict-v2 CSVs use different session-id names.
    # Normalize them so plotting functions can use either safely.
    if "session_id" not in out.columns and "session_id_synthesis" in out.columns:
        out["session_id"] = out["session_id_synthesis"]

    if "session_id_synthesis" not in out.columns and "session_id" in out.columns:
        out["session_id_synthesis"] = out["session_id"]

    for col in ["session_id", "session_id_synthesis"]:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
            )

    return out


def save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def annotate_matrix(ax, mat: np.ndarray, fmt: str = ".3f") -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7)


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns found: {candidates}. Available: {list(df.columns)}")


def plot_session_overview(strict_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    df = ensure_session_str(read_csv(strict_dir / "cross_session_spontaneous_session_summary.csv"))
    df = df.set_index("session_id").reindex(STRICT_SESSIONS).reset_index()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(df))
    bars = ax.bar(x, df["median_observed_minus_shuffle"].astype(float).values)

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["session_id"], rotation=35, ha="right")
    ax.set_ylabel("Median observed-minus-shuffled gain")
    ax.set_title("Controlled memory across strict six-session cohort")

    for b, frac, top in zip(
        bars,
        df["fraction_observed_above_shuffle_mean"].astype(float).values,
        df["top_target"].astype(str).values,
    ):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{frac:.2f}\n{fmt_target(top).split()[0]}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    out = out_dir / "strict_v2_session_controlled_memory_overview.png"
    save(fig, out)
    manifest.append({
        "figure": out.name,
        "description": "Six-session overview of median controlled memory gain, annotated by fraction above shuffled controls and top target.",
        "source": "cross_session_spontaneous_session_summary.csv",
    })


def plot_target_by_session_heatmap(strict_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    df = ensure_session_str(read_csv(strict_dir / "cross_session_spontaneous_target_by_session.csv"))
    val_col = first_existing(df, [
        "median_observed_minus_shuffle",
        "observed_minus_shuffle_mean",
        "mean_observed_minus_shuffle",
    ])

    pivot = (
        df.pivot_table(index="target", columns="session_id", values=val_col, aggfunc="median")
        .reindex(TARGET_ORDER)
        .reindex(columns=STRICT_SESSIONS)
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    im = ax.imshow(pivot.values.astype(float), aspect="auto")
    ax.set_xticks(np.arange(len(STRICT_SESSIONS)))
    ax.set_yticks(np.arange(len(TARGET_ORDER)))
    ax.set_xticklabels(STRICT_SESSIONS, rotation=35, ha="right")
    ax.set_yticklabels([fmt_target(t) for t in TARGET_ORDER])
    ax.set_title("Target-specific controlled memory by session")
    ax.set_xlabel("Session")
    ax.set_ylabel("Target")

    annotate_matrix(ax, pivot.values.astype(float), ".3f")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Median observed-minus-shuffled gain")

    out = out_dir / "strict_v2_target_by_session_heatmap.png"
    save(fig, out)
    manifest.append({
        "figure": out.name,
        "description": "Heatmap of controlled memory gain for each target across the six strict sessions.",
        "source": "cross_session_spontaneous_target_by_session.csv",
    })


def plot_controlled_gain_distribution(strict_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    group_path = strict_dir / "cross_session_spontaneous_target_group_summary.csv"
    if group_path.exists():
        df = ensure_session_str(read_csv(group_path))
    else:
        df = ensure_session_str(read_csv(strict_dir / "cross_session_spontaneous_target_by_session.csv"))

    val_col = first_existing(df, [
        "observed_minus_shuffle_mean",
        "median_observed_minus_shuffle",
        "mean_observed_minus_shuffle",
        "observed_minus_shuffle",
    ])

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    data = []
    labels = []

    for t in TARGET_ORDER:
        vals = df.loc[df["target"] == t, val_col].astype(float).dropna().values
        if len(vals):
            data.append(vals)
            labels.append(fmt_target(t))

    ax.boxplot(data, tick_labels=labels, showfliers=False)

    for i, vals in enumerate(data, start=1):
        if len(vals):
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, alpha=0.75)

    ax.axhline(0, linewidth=1)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Observed-minus-shuffled gain")
    ax.set_title("Distribution of controlled memory across session-region groups")

    out = out_dir / "strict_v2_controlled_gain_distribution_by_target.png"
    save(fig, out)
    manifest.append({
        "figure": out.name,
        "description": "Box/point distribution of controlled memory gains across session-region groups for each target.",
        "source": group_path.name if group_path.exists() else "cross_session_spontaneous_target_by_session.csv",
    })


def corr_matrix_from_long(df: pd.DataFrame) -> pd.DataFrame:
    mat = pd.DataFrame(np.eye(len(AXES)), index=AXES, columns=AXES, dtype=float)
    g = df.groupby(["axis_a", "axis_b"], as_index=False)["corr"].mean()
    for _, r in g.iterrows():
        a = str(r["axis_a"]).replace("_raw", "")
        b = str(r["axis_b"]).replace("_raw", "")
        c = float(r["corr"])
        mat.loc[a, b] = c
        mat.loc[b, a] = c
    return mat.reindex(index=AXES, columns=AXES)


def plot_axis_correlation_comparison(strict_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    raw = read_csv(strict_dir / "raw_axis_correlation_summary.csv")
    ortho = read_csv(strict_dir / "orthogonal_axis_correlation_summary.csv")

    raw_mat = corr_matrix_from_long(raw)
    ortho_mat = corr_matrix_from_long(ortho)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)

    for ax, mat, title in [
        (axes[0], raw_mat, "Raw proxies"),
        (axes[1], ortho_mat, "After residualization"),
    ]:
        im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(np.arange(len(AXES)))
        ax.set_yticks(np.arange(len(AXES)))
        ax.set_xticklabels(AXES)
        ax.set_yticklabels(AXES)
        ax.set_title(title)
        annotate_matrix(ax, mat.values.astype(float), ".2f")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Mean correlation")
    fig.suptitle("HRSM axis correlations before and after residualization", y=1.05)

    out = out_dir / "strict_v2_raw_vs_orthogonal_axis_correlations.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    manifest.append({
        "figure": out.name,
        "description": "Side-by-side mean raw and residualized HRSM axis correlation matrices.",
        "source": "raw_axis_correlation_summary.csv; orthogonal_axis_correlation_summary.csv",
    })


def plot_variance_residual(strict_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    df = read_csv(strict_dir / "variance_scaling_target_residual_summary.csv")
    df = df.set_index("target").reindex(TARGET_ORDER).reset_index()

    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(x - width / 2, df["median_observed_minus_shuffle"].astype(float), width, label="Raw controlled")
    ax.bar(x + width / 2, df["median_residual_gain"].astype(float), width, label="Residual")
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([fmt_target(t) for t in df["target"]], rotation=35, ha="right")
    ax.set_ylabel("Median gain")
    ax.set_title("Controlled memory before and after variance/autocorrelation residualization")
    ax.legend(frameon=False)

    out = out_dir / "strict_v2_variance_residual_comparison.png"
    save(fig, out)
    manifest.append({
        "figure": out.name,
        "description": "Comparison of raw controlled gain and variance/autocorrelation residual gain across targets.",
        "source": "variance_scaling_target_residual_summary.csv",
    })


def plot_lag_ablation(ablation_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    df = read_csv(ablation_dir / "lag_ablation_cross_session_target_summary.csv")
    lags = ["lag1", "lag2", "lag12", "lag123", "lag1234"]

    fig, ax = plt.subplots(figsize=(8.6, 4.9))

    for target in ABLATION_TARGETS:
        sub = df[df["target"] == target].copy()
        sub["lag_label"] = pd.Categorical(sub["lag_label"], categories=lags, ordered=True)
        sub = sub.sort_values("lag_label")
        vals = sub["median_observed_minus_shuffle"].astype(float).values
        ax.plot(lags, vals, marker="o", label=fmt_target(target))

    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Median observed-minus-shuffled gain")
    ax.set_xlabel("Lag set")
    ax.set_title("Short-window lag ablation across strict six-session cohort")
    ax.legend(frameon=False)

    out = out_dir / "strict_v2_lag_ablation_curves.png"
    save(fig, out)
    manifest.append({
        "figure": out.name,
        "description": "Lag-ablation curves showing controlled memory gain across lag windows for four targets.",
        "source": "lag_ablation_cross_session_target_summary.csv",
    })


def plot_phi_memory_landscape(strict_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    df = ensure_session_str(read_csv(strict_dir / "spontaneous_neural_ripeness_index.csv"))

    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    for region in REGION_ORDER:
        sub = df[df["region"] == region]
        if sub.empty:
            continue
        sizes = 35 + 130 * sub["ripeness_score"].astype(float).clip(lower=0, upper=1)
        ax.scatter(
            sub["Phi_neural"].astype(float),
            sub["organizational_memory_gain"].astype(float),
            s=sizes,
            alpha=0.75,
            label=region,
        )

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(r"$\Phi_{\rm neural}$")
    ax.set_ylabel("Organizational memory gain")
    ax.set_title("Structural potential versus organizational memory")

    label_df = df.sort_values("ripeness_score", ascending=False).head(6).copy()
    low_df = df.sort_values("organizational_memory_gain").head(1).copy()
    label_df = pd.concat([label_df, low_df], ignore_index=True).drop_duplicates(
        subset=["session_id_synthesis", "region"]
    )

    for _, r in label_df.iterrows():
        label = f"{str(r['session_id_synthesis'])[-4:]}/{r['region']}"
        ax.text(
            float(r["Phi_neural"]),
            float(r["organizational_memory_gain"]),
            label,
            fontsize=7,
            ha="left",
            va="bottom",
        )

    ax.legend(frameon=False, title="Region")

    out = out_dir / "strict_v2_phi_memory_landscape_labeled.png"
    save(fig, out)
    manifest.append({
        "figure": out.name,
        "description": "Combined Phi-vs-organizational-memory landscape with selected high-ripeness and low-memory labels.",
        "source": "spontaneous_neural_ripeness_index.csv",
    })


def load_domain_summaries(session_root: Path) -> pd.DataFrame:
    rows = []
    for sid in STRICT_SESSIONS:
        p = session_root / f"session_{sid}_spontaneous_v1" / "real_neural_hrsm_domain_summary.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        df = read_csv(p)
        df["session_id"] = sid
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_region_hrsm_profiles(session_root: Path, out_dir: Path, manifest: list[dict]) -> None:
    df = load_domain_summaries(session_root)

    med = (
        df.groupby("region", as_index=False)[AXES + ["Phi_neural"]]
        .median()
        .set_index("region")
        .reindex(REGION_ORDER)
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(AXES))

    for region in REGION_ORDER:
        if region not in med.index:
            continue
        ax.plot(x, med.loc[region, AXES].astype(float).values, marker="o", label=region)

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(AXES)
    ax.set_ylabel("Median residualized coordinate")
    ax.set_title("Region-level HRSM profiles across strict sessions")
    ax.legend(frameon=False, title="Region")

    out = out_dir / "strict_v2_region_hrsm_profiles.png"
    save(fig, out)
    manifest.append({
        "figure": out.name,
        "description": "Median H/R/S/M profiles by region across strict sessions.",
        "source": "real_neural_hrsm_domain_summary.csv from strict session folders",
    })


def main() -> None:
    args = parse_args()
    strict_dir = Path(args.strict_dir)
    ablation_dir = Path(args.ablation_dir)
    session_root = Path(args.session_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    plot_session_overview(strict_dir, out_dir, manifest)
    plot_target_by_session_heatmap(strict_dir, out_dir, manifest)
    plot_controlled_gain_distribution(strict_dir, out_dir, manifest)
    plot_axis_correlation_comparison(strict_dir, out_dir, manifest)
    plot_variance_residual(strict_dir, out_dir, manifest)
    plot_lag_ablation(ablation_dir, out_dir, manifest)
    plot_phi_memory_landscape(strict_dir, out_dir, manifest)
    plot_region_hrsm_profiles(session_root, out_dir, manifest)

    pd.DataFrame(manifest).to_csv(out_dir / "strict_v2_combined_figure_manifest.csv", index=False)

    print(f"[ok] wrote {len(manifest)} combined manuscript figures to {out_dir}")
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
