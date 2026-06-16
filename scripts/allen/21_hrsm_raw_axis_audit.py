#!/usr/bin/env python3
"""
Audit raw and orthogonalized HRSM axis correlations across Allen spontaneous sessions.

Purpose:
    Address the reviewer concern that near-zero H/R/S/M correlations are guaranteed
    by residualization and therefore should not be presented as an independent
    biological result.

Inputs:
    results/real_allen/session_<id>_spontaneous_v1/real_neural_hrsm_bin_level_metrics.csv

Outputs:
    results/cross_session/<out-name>/raw_axis_correlation_summary.csv
    results/cross_session/<out-name>/orthogonal_axis_correlation_summary.csv
    results/cross_session/<out-name>/axis_audit_summary.csv
    results/figures/cross_session/<out-name>/raw_axis_correlation_heatmap.png
    results/figures/cross_session/<out-name>/orthogonal_axis_correlation_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AXES = ["H", "R", "S", "M"]
RAW_AXES = [f"{a}_raw" for a in AXES]
ORTHO_AXES = AXES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--session-dirs",
        nargs="+",
        required=True,
        help="Session result directories under results/real_allen/.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for CSV audit summaries.",
    )
    p.add_argument(
        "--fig-dir",
        required=True,
        help="Output directory for audit figures.",
    )
    return p.parse_args()


def safe_session_id(session_dir: Path, df: pd.DataFrame) -> str:
    if "session_id" in df.columns and df["session_id"].notna().any():
        return str(df["session_id"].dropna().iloc[0])
    name = session_dir.name
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else name


def corr_long(df: pd.DataFrame, cols: list[str], session_id: str, kind: str) -> pd.DataFrame:
    corr = df[cols].astype(float).corr()
    rows = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            rows.append(
                {
                    "session_id": session_id,
                    "axis_a": a.replace("_raw", ""),
                    "axis_b": b.replace("_raw", ""),
                    "correlation_kind": kind,
                    "corr": float(corr.loc[a, b]),
                    "abs_corr": float(abs(corr.loc[a, b])),
                }
            )
    return pd.DataFrame(rows)


def plot_mean_corr(long_df: pd.DataFrame, title: str, out_path: Path) -> None:
    mat = pd.DataFrame(np.eye(len(AXES)), index=AXES, columns=AXES, dtype=float)

    grouped = (
        long_df.groupby(["axis_a", "axis_b"], as_index=False)["corr"]
        .mean()
        .copy()
    )
    for _, row in grouped.iterrows():
        a, b, c = row["axis_a"], row["axis_b"], float(row["corr"])
        mat.loc[a, b] = c
        mat.loc[b, a] = c

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(mat.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(AXES)))
    ax.set_yticks(range(len(AXES)))
    ax.set_xticklabels(AXES)
    ax.set_yticklabels(AXES)
    ax.set_title(title)

    for i in range(len(AXES)):
        for j in range(len(AXES)):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    ortho_rows = []
    summary_rows = []

    for session_dir_s in args.session_dirs:
        session_dir = Path(session_dir_s)
        metrics_path = session_dir / "real_neural_hrsm_bin_level_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)

        df = pd.read_csv(metrics_path)
        session_id = safe_session_id(session_dir, df)

        missing_raw = [c for c in RAW_AXES if c not in df.columns]
        missing_ortho = [c for c in ORTHO_AXES if c not in df.columns]
        if missing_raw or missing_ortho:
            raise ValueError(
                f"{session_dir}: missing raw={missing_raw}, missing_ortho={missing_ortho}"
            )

        raw = corr_long(df, RAW_AXES, session_id, "raw_pre_residualization")
        ortho = corr_long(df, ORTHO_AXES, session_id, "orthogonalized_post_residualization")

        raw_rows.append(raw)
        ortho_rows.append(ortho)

        summary_rows.append(
            {
                "session_id": session_id,
                "n_rows": int(len(df)),
                "max_abs_corr_raw": float(raw["abs_corr"].max()),
                "mean_abs_corr_raw": float(raw["abs_corr"].mean()),
                "max_abs_corr_orthogonalized": float(ortho["abs_corr"].max()),
                "mean_abs_corr_orthogonalized": float(ortho["abs_corr"].mean()),
            }
        )

    raw_long = pd.concat(raw_rows, ignore_index=True)
    ortho_long = pd.concat(ortho_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("session_id")

    raw_long.to_csv(out_dir / "raw_axis_correlation_summary.csv", index=False)
    ortho_long.to_csv(out_dir / "orthogonal_axis_correlation_summary.csv", index=False)
    summary.to_csv(out_dir / "axis_audit_summary.csv", index=False)

    plot_mean_corr(
        raw_long,
        "Mean raw HRSM axis correlation before residualization",
        fig_dir / "raw_axis_correlation_heatmap.png",
    )
    plot_mean_corr(
        ortho_long,
        "Mean HRSM axis correlation after residualization",
        fig_dir / "orthogonal_axis_correlation_heatmap.png",
    )

    print("[ok] wrote axis audit outputs")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
