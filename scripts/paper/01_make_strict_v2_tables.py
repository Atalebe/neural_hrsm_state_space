#!/usr/bin/env python3
"""
Generate LaTeX tables for the strict-v2 Neural HRSM manuscript.

The tables are generated from committed CSV outputs rather than hard-coded
manuscript values.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


STRICT_SESSIONS = [
    "715093703",
    "719161530",
    "750749662",
    "751348571",
    "755434585",
    "756029989",
]

ROOT = Path(".")
OUT = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)

SYN = Path("results/cross_session/allen_spontaneous_strict_v2")
ABL = Path("results/ablation/allen_spontaneous_strict_v2")


def esc(x) -> str:
    s = str(x)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def fmt(x, digits=3):
    if pd.isna(x):
        return "--"
    try:
        x = float(x)
    except Exception:
        return esc(x)
    return f"{x:.{digits}f}"


def write_tabular(path: Path, headers: list[str], rows: list[list[str]], align: str | None = None):
    if align is None:
        align = "l" + "r" * (len(headers) - 1)

    lines = []
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + align + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    path.write_text("\n".join(lines) + "\n")


def extraction_table():
    rows = []
    for sid in STRICT_SESSIONS:
        p = Path(f"data/interim/allen_neuropixels_real/session_{sid}_spontaneous_v1/extraction_summary.csv")
        df = pd.read_csv(p)
        rows.append(
            [
                sid,
                str(int(df["n_units"].sum())),
                str(int(df["n_units"].min())),
                str(int(df["n_presentations"].iloc[0])),
                str(int(df["n_bins"].sum())),
                esc(",".join(df["region"].tolist())),
            ]
        )

    write_tabular(
        OUT / "table_strict_v2_extraction_summary.tex",
        ["Session", "Units", "Min/region", "Presentations", "Binned rows", "Regions"],
        rows,
        align="lrrrrl",
    )


def session_summary_table():
    df = pd.read_csv(SYN / "cross_session_spontaneous_session_summary.csv")
    df["session_id"] = df["session_id"].astype(str)
    df = df[df["session_id"].isin(STRICT_SESSIONS)].copy()

    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                str(r["session_id"]),
                esc(r["best_phi_region"]),
                fmt(r["best_phi_value"], 3),
                fmt(r["median_observed_minus_shuffle"], 3),
                fmt(r["fraction_observed_above_shuffle_mean"], 2),
                esc(r["top_target"]),
            ]
        )

    write_tabular(
        OUT / "table_strict_v2_session_summary.tex",
        [
            "Session",
            "Best $\\Phi$ region",
            "Best $\\Phi$",
            "Median controlled gain",
            "Frac. above shuffle",
            "Top target",
        ],
        rows,
        align="llrrrl",
    )


def target_synthesis_table():
    df = pd.read_csv(SYN / "cross_session_spontaneous_target_synthesis.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(r["target"]),
                str(int(r["n_sessions"])),
                fmt(r["median_observed_minus_shuffle"], 3),
                fmt(r["mean_observed_minus_shuffle"], 3),
                fmt(r["min_fraction_above_shuffle"], 2),
                fmt(r["median_empirical_p"], 3),
                str(int(r["cross_session_rank"])),
            ]
        )

    write_tabular(
        OUT / "table_strict_v2_target_synthesis.tex",
        [
            "Target",
            "$n$",
            "Median controlled gain",
            "Mean controlled gain",
            "Min frac. above shuffle",
            "Median $p$",
            "Rank",
        ],
        rows,
        align="lrrrrrr",
    )


def variance_table():
    df = pd.read_csv(SYN / "variance_scaling_target_residual_summary.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(r["target"]),
                fmt(r["median_observed_minus_shuffle"], 3),
                str(int(r["raw_controlled_rank"])),
                fmt(r["median_residual_gain"], 3),
                str(int(r["variance_residual_rank"])),
            ]
        )

    write_tabular(
        OUT / "table_strict_v2_variance_residual.tex",
        ["Target", "Raw controlled gain", "Raw rank", "Residual gain", "Residual rank"],
        rows,
        align="lrrrr",
    )


def lag_ablation_table():
    df = pd.read_csv(ABL / "lag_ablation_cross_session_target_summary.csv")
    targets = [
        "active_unit_fraction",
        "population_rate_entropy",
        "population_mean_rate_hz",
        "population_state_speed",
    ]
    lags = ["lag1", "lag2", "lag12", "lag123", "lag1234"]

    rows = []
    for target in targets:
        sub = df[df["target"] == target].set_index("lag_label")
        rows.append(
            [esc(target)]
            + [fmt(sub.loc[lag, "median_observed_minus_shuffle"], 3) for lag in lags]
        )

    write_tabular(
        OUT / "table_strict_v2_lag_ablation.tex",
        ["Target", "lag1", "lag2", "lag12", "lag123", "lag1234"],
        rows,
        align="lrrrrr",
    )


def ripeness_table():
    expected = SYN / "spontaneous_neural_ripeness_region_summary.csv"

    candidates = [
        expected,
        SYN / "spontaneous_neural_ripeness_by_region.csv",
        SYN / "spontaneous_neural_ripeness_region_summary.csv",
        SYN / "ripeness_region_summary.csv",
    ]

    p_region = next((p for p in candidates if p.exists()), None)

    if p_region is not None:
        df = pd.read_csv(p_region)
    else:
        p_index = SYN / "spontaneous_neural_ripeness_index.csv"
        if not p_index.exists():
            raise FileNotFoundError(
                f"Could not find ripeness region summary or ripeness index in {SYN}"
            )

        idx = pd.read_csv(p_index)

        session_col = (
            "session_id_synthesis"
            if "session_id_synthesis" in idx.columns
            else "session_id"
        )

        required = [
            "region",
            session_col,
            "ripeness_score",
            "Phi_neural",
            "mean_rate_controlled_gain",
            "active_controlled_gain",
            "entropy_controlled_gain",
        ]
        missing = [c for c in required if c not in idx.columns]
        if missing:
            raise ValueError(f"Ripeness index missing required columns: {missing}")

        df = (
            idx.groupby("region", as_index=False)
            .agg(
                n_sessions=(session_col, "nunique"),
                median_ripeness=("ripeness_score", "median"),
                mean_ripeness=("ripeness_score", "mean"),
                median_phi=("Phi_neural", "median"),
                median_mean_rate_controlled_gain=("mean_rate_controlled_gain", "median"),
                median_active_controlled_gain=("active_controlled_gain", "median"),
                median_entropy_controlled_gain=("entropy_controlled_gain", "median"),
            )
            .sort_values("median_ripeness", ascending=False)
            .reset_index(drop=True)
        )

        df["region_rank"] = range(1, len(df) + 1)
        df.to_csv(expected, index=False)
        print(f"[ok] rebuilt missing ripeness region summary from {p_index}")

    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(r["region"]),
                str(int(r["n_sessions"])),
                fmt(r["median_ripeness"], 3),
                fmt(r["mean_ripeness"], 3),
                fmt(r["median_phi"], 3),
                fmt(r["median_active_controlled_gain"], 3),
                fmt(r["median_entropy_controlled_gain"], 3),
                str(int(r["region_rank"])),
            ]
        )

    write_tabular(
        OUT / "table_strict_v2_ripeness_region.tex",
        [
            "Region",
            "$n$",
            "Median ripeness",
            "Mean ripeness",
            "Median $\\Phi$",
            "Median active gain",
            "Median entropy gain",
            "Rank",
        ],
        rows,
        align="lrrrrrrr",
    )

def axis_audit_table():
    df = pd.read_csv(SYN / "axis_audit_summary.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                str(r["session_id"]),
                fmt(r["max_abs_corr_raw"], 3),
                fmt(r["mean_abs_corr_raw"], 3),
                f"{float(r['max_abs_corr_orthogonalized']):.2e}",
                f"{float(r['mean_abs_corr_orthogonalized']):.2e}",
            ]
        )

    write_tabular(
        OUT / "table_strict_v2_axis_audit.tex",
        [
            "Session",
            "Max raw $|r|$",
            "Mean raw $|r|$",
            "Max orth. $|r|$",
            "Mean orth. $|r|$",
        ],
        rows,
        align="lrrrr",
    )


def write_summary_macros():
    syn = pd.read_csv(SYN / "cross_session_spontaneous_target_synthesis.csv")
    ab = pd.read_csv(ABL / "lag_ablation_cross_session_target_summary.csv")

    active = syn[syn["target"] == "active_unit_fraction"].iloc[0]
    entropy = syn[syn["target"] == "population_rate_entropy"].iloc[0]
    speed = syn[syn["target"] == "population_state_speed"].iloc[0]

    active_lag = ab[ab["target"] == "active_unit_fraction"].set_index("lag_label")
    entropy_lag = ab[ab["target"] == "population_rate_entropy"].set_index("lag_label")

    lines = [
        rf"\newcommand{{\StrictSessionCount}}{{{len(STRICT_SESSIONS)}}}",
        rf"\newcommand{{\StrictActiveMedianGain}}{{{fmt(active['median_observed_minus_shuffle'], 3)}}}",
        rf"\newcommand{{\StrictEntropyMedianGain}}{{{fmt(entropy['median_observed_minus_shuffle'], 3)}}}",
        rf"\newcommand{{\StrictSpeedMedianGain}}{{{fmt(speed['median_observed_minus_shuffle'], 3)}}}",
        rf"\newcommand{{\StrictActiveLagOne}}{{{fmt(active_lag.loc['lag1', 'median_observed_minus_shuffle'], 3)}}}",
        rf"\newcommand{{\StrictActiveLagFour}}{{{fmt(active_lag.loc['lag1234', 'median_observed_minus_shuffle'], 3)}}}",
        rf"\newcommand{{\StrictEntropyLagOne}}{{{fmt(entropy_lag.loc['lag1', 'median_observed_minus_shuffle'], 3)}}}",
        rf"\newcommand{{\StrictEntropyLagFour}}{{{fmt(entropy_lag.loc['lag1234', 'median_observed_minus_shuffle'], 3)}}}",
    ]
    (OUT / "strict_v2_macros.tex").write_text("\n".join(lines) + "\n")


def main():
    extraction_table()
    session_summary_table()
    target_synthesis_table()
    variance_table()
    lag_ablation_table()
    ripeness_table()
    axis_audit_table()
    write_summary_macros()
    print("[ok] wrote strict-v2 manuscript tables to paper/tables")


if __name__ == "__main__":
    main()
