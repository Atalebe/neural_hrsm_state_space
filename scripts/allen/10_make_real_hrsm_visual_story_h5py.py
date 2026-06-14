#!/usr/bin/env python3
"""
Make visual-story figures for the scaled real Allen Neural HRSM run.

Inputs are the small result CSVs from:
    results/real_allen/session_715093703_h5py_v1/

Optional input:
    population_state_summary.csv from data/processed/

Outputs:
    PNG figures and a figure_manifest.csv.

Scientific role:
These figures show the state geometry and memory audit without overstating the
result. The visual story should make clear that HRSM structure is stable, while
memory gain is weak and region/stimulus-dependent in scaled v1.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGION_ORDER = ["VISp", "VISl", "LGd", "CA1"]
FAMILY_ORDER = ["drifting_gratings", "natural_scenes", "spontaneous"]
AXES = ["H", "R", "S", "M"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing real Allen HRSM result CSVs.",
    )
    parser.add_argument(
        "--population-summary",
        default=None,
        help="Optional population_state_summary.csv path.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for PNG figures and figure manifest.",
    )
    return parser.parse_args()


def ordered(values, preferred):
    values = list(pd.Series(values).dropna().unique())
    return [x for x in preferred if x in values] + sorted([x for x in values if x not in preferred])


def safe_slug(x):
    return str(x).replace("/", "_").replace(" ", "_")


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def heatmap_from_summary(summary, value_col, title, out_path, manifest, description):
    regions = ordered(summary["region"], REGION_ORDER)
    families = ordered(summary["stimulus_family"], FAMILY_ORDER)

    pivot = (
        summary.pivot_table(
            index="region",
            columns="stimulus_family",
            values=value_col,
            aggfunc="median",
        )
        .reindex(index=regions, columns=families)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Stimulus family")
    ax.set_ylabel("Region")
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions)

    for i, region in enumerate(regions):
        for j, family in enumerate(families):
            val = pivot.loc[region, family]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    savefig(fig, out_path)

    manifest.append(
        {
            "figure": out_path.name,
            "description": description,
            "source": "real_neural_hrsm_domain_summary.csv",
        }
    )


def make_axis_heatmaps(summary, out_dir, manifest):
    for axis in AXES:
        heatmap_from_summary(
            summary=summary,
            value_col=axis,
            title=f"Real Allen HRSM {axis} axis by region and stimulus",
            out_path=out_dir / f"real_allen_hrsm_axis_{axis}.png",
            manifest=manifest,
            description=f"Heatmap of median {axis} coordinate across region and stimulus-family summaries.",
        )


def make_phi_heatmap(summary, out_dir, manifest):
    heatmap_from_summary(
        summary=summary,
        value_col="Phi_neural",
        title="Real Allen neural potential by region and stimulus",
        out_path=out_dir / "real_allen_phi_neural_heatmap.png",
        manifest=manifest,
        description="Heatmap of median Phi_neural across the scaled real Allen v1 region-stimulus summaries.",
    )


def make_state_speed_heatmap(summary, out_dir, manifest):
    heatmap_from_summary(
        summary=summary,
        value_col="mean_state_speed",
        title="Population state speed by region and stimulus",
        out_path=out_dir / "real_allen_population_state_speed_heatmap.png",
        manifest=manifest,
        description="Heatmap of mean population_state_speed, showing which region-stimulus blocks are dynamically more deforming.",
    )


def make_hr_phase_scatter(summary, out_dir, manifest):
    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in summary.iterrows():
        ax.scatter(row["H"], row["R"], s=max(35, min(220, row["n_state_rows"] / 3)))
        ax.text(
            row["H"],
            row["R"],
            f"{row['region']}:{row['stimulus_family'].replace('_', ' ')}",
            fontsize=7,
            ha="left",
            va="bottom",
        )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("H, reserve / response capacity")
    ax.set_ylabel("R, recoverability")
    ax.set_title("Real Allen HR phase portrait")
    ax.grid(True, linewidth=0.3)

    out_path = out_dir / "real_allen_hr_phase_portrait.png"
    savefig(fig, out_path)

    manifest.append(
        {
            "figure": out_path.name,
            "description": "Scatter plot of H against R for region-stimulus HRSM summaries. Marker size reflects n_state_rows.",
            "source": "real_neural_hrsm_domain_summary.csv",
        }
    )


def make_sm_phase_scatter(summary, out_dir, manifest):
    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in summary.iterrows():
        ax.scatter(row["S"], row["M"], s=max(35, min(220, row["n_state_rows"] / 3)))
        ax.text(
            row["S"],
            row["M"],
            f"{row['region']}:{row['stimulus_family'].replace('_', ' ')}",
            fontsize=7,
            ha="left",
            va="bottom",
        )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("S, stability / coherence")
    ax.set_ylabel("M, retained history proxy")
    ax.set_title("Real Allen SM phase portrait")
    ax.grid(True, linewidth=0.3)

    out_path = out_dir / "real_allen_sm_phase_portrait.png"
    savefig(fig, out_path)

    manifest.append(
        {
            "figure": out_path.name,
            "description": "Scatter plot of S against M for region-stimulus HRSM summaries. Marker size reflects n_state_rows.",
            "source": "real_neural_hrsm_domain_summary.csv",
        }
    )


def make_memory_gain_bar(memory, out_dir, manifest):
    ok = memory[memory["status"] == "ok"].copy()
    if ok.empty:
        return

    ok["label"] = ok["region"] + " / " + ok["stimulus_family"].str.replace("_", " ", regex=False)
    ok = ok.sort_values("memory_gain_r2")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(np.arange(len(ok)), ok["memory_gain_r2"].to_numpy(dtype=float))
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(np.arange(len(ok)))
    ax.set_xticklabels(ok["label"], rotation=35, ha="right")
    ax.set_ylabel("Memory gain, ΔR²")
    ax.set_title("Real Allen memory-kernel gain by region and stimulus")
    ax.grid(True, axis="y", linewidth=0.3)

    out_path = out_dir / "real_allen_memory_gain_bar.png"
    savefig(fig, out_path)

    manifest.append(
        {
            "figure": out_path.name,
            "description": "Bar chart of memory_gain_r2 from the real memory-kernel audit. Values near zero indicate weak or neutral lagged-history improvement.",
            "source": "real_memory_kernel_gain_summary.csv",
        }
    )


def make_baseline_vs_memory(memory, out_dir, manifest):
    ok = memory[memory["status"] == "ok"].copy()
    if ok.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ok["baseline_r2"], ok["memory_r2"])

    lim_min = float(np.nanmin([ok["baseline_r2"].min(), ok["memory_r2"].min()]))
    lim_max = float(np.nanmax([ok["baseline_r2"].max(), ok["memory_r2"].max()]))
    pad = 0.05 * max(1.0, lim_max - lim_min)
    lims = [lim_min - pad, lim_max + pad]

    ax.plot(lims, lims, linewidth=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    for _, row in ok.iterrows():
        ax.text(
            row["baseline_r2"],
            row["memory_r2"],
            f"{row['region']}:{row['stimulus_family'].replace('_', ' ')}",
            fontsize=7,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Present-state baseline R²")
    ax.set_ylabel("Memory-augmented R²")
    ax.set_title("Baseline versus memory model performance")
    ax.grid(True, linewidth=0.3)

    out_path = out_dir / "real_allen_baseline_vs_memory_r2.png"
    savefig(fig, out_path)

    manifest.append(
        {
            "figure": out_path.name,
            "description": "Scatter plot comparing baseline R² and memory-model R². Points above the diagonal favor the memory model.",
            "source": "real_memory_kernel_gain_summary.csv",
        }
    )


def make_phi_boxplot(metrics, out_dir, manifest):
    metrics = metrics.copy()
    metrics["label"] = metrics["region"] + " / " + metrics["stimulus_family"].str.replace("_", " ", regex=False)

    order = (
        metrics[["region", "stimulus_family", "label"]]
        .drop_duplicates()
        .sort_values(["region", "stimulus_family"])
    )["label"].tolist()

    data = [metrics.loc[metrics["label"] == label, "Phi_neural"].dropna().to_numpy() for label in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, labels=order, showfliers=False)
    ax.axhline(0, linewidth=0.8)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Phi_neural, bin-level")
    ax.set_title("Distribution of real Allen bin-level neural potential")
    ax.grid(True, axis="y", linewidth=0.3)

    out_path = out_dir / "real_allen_phi_neural_boxplot.png"
    savefig(fig, out_path)

    manifest.append(
        {
            "figure": out_path.name,
            "description": "Boxplot of bin-level Phi_neural distributions across region-stimulus blocks.",
            "source": "real_neural_hrsm_bin_level_metrics.csv",
        }
    )


def make_population_rate_bar(pop_summary, out_dir, manifest):
    if pop_summary is None or pop_summary.empty:
        return

    df = pop_summary.copy()
    df["label"] = df["region"] + " / " + df["stimulus_family"].str.replace("_", " ", regex=False)
    df = df.sort_values("mean_population_rate_hz")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(np.arange(len(df)), df["mean_population_rate_hz"].to_numpy(dtype=float))
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.set_ylabel("Mean population rate, Hz")
    ax.set_title("Mean population firing-rate summary")
    ax.grid(True, axis="y", linewidth=0.3)

    out_path = out_dir / "real_allen_population_rate_bar.png"
    savefig(fig, out_path)

    manifest.append(
        {
            "figure": out_path.name,
            "description": "Bar chart of mean population firing rate by region and stimulus family.",
            "source": "population_state_summary.csv",
        }
    )


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "real_neural_hrsm_domain_summary.csv"
    metrics_path = results_dir / "real_neural_hrsm_bin_level_metrics.csv"
    memory_path = results_dir / "real_memory_kernel_gain_summary.csv"
    pooled_path = results_dir / "real_memory_kernel_pooled_summary.csv"

    required = [summary_path, metrics_path, memory_path, pooled_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required result files: {missing}")

    summary = pd.read_csv(summary_path)
    metrics = pd.read_csv(metrics_path)
    memory = pd.read_csv(memory_path)
    pooled = pd.read_csv(pooled_path)

    pop_summary = None
    if args.population_summary:
        pop_path = Path(args.population_summary)
        if pop_path.exists():
            pop_summary = pd.read_csv(pop_path)

    manifest = []

    make_phi_heatmap(summary, out_dir, manifest)
    make_axis_heatmaps(summary, out_dir, manifest)
    make_state_speed_heatmap(summary, out_dir, manifest)
    make_hr_phase_scatter(summary, out_dir, manifest)
    make_sm_phase_scatter(summary, out_dir, manifest)
    make_memory_gain_bar(memory, out_dir, manifest)
    make_baseline_vs_memory(memory, out_dir, manifest)
    make_phi_boxplot(metrics, out_dir, manifest)
    make_population_rate_bar(pop_summary, out_dir, manifest)

    pooled_note = pooled.to_dict(orient="records")[0] if len(pooled) else {}
    manifest_df = pd.DataFrame(manifest)
    manifest_df["run_note"] = (
        "Scaled real Allen v1 visual story. HRSM axis separation is strong; "
        "memory gain is weak and mixed. Pooled memory summary: "
        + str(pooled_note)
    )

    manifest_path = out_dir / "figure_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"[ok] wrote figures to {out_dir}")
    print(f"[ok] wrote {manifest_path}")
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
