#!/usr/bin/env python3
"""
Compute real Allen Neural HRSM proxies from a population-state matrix.

This is the first real-data H, R, S, M proxy layer after low-memory HDF5
extraction. It is deliberately conservative.

Input:
    population_state_matrix.csv

Outputs:
    real_neural_hrsm_bin_level_metrics.csv
    real_neural_hrsm_domain_summary.csv
    real_orthogonality_audit.csv

Scientific caution:
This script computes a first proof-of-path HRSM projection for one small real
Allen session extraction. It is not yet the biological result.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


AXES = ["H", "R", "S", "M"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population",
        required=True,
        help="Path to population_state_matrix.csv.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for real HRSM tables.",
    )
    parser.add_argument(
        "--low-q",
        type=float,
        default=0.33,
        help="Lower quantile for domain binning.",
    )
    parser.add_argument(
        "--high-q",
        type=float,
        default=0.66,
        help="Upper quantile for domain binning.",
    )
    return parser.parse_args()


def robust_z(x):
    x = pd.Series(x).astype(float)
    med = float(x.median())
    mad = float((x - med).abs().median())

    if not np.isfinite(mad) or mad <= 1e-12:
        std = float(x.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - float(x.mean())) / std

    return 0.67448975 * (x - med) / mad


def safe_mean(cols):
    arr = np.vstack([np.asarray(c, dtype=float) for c in cols])
    return np.nanmean(arr, axis=0)


def add_lag_features(df):
    out = df.copy()
    traj_cols = ["session_id", "region", "stimulus_family", "trial_id"]

    lag_source_cols = [
        "population_mean_rate_hz",
        "active_unit_fraction",
        "population_l2_rate_norm",
        "population_rate_entropy",
    ]

    for col in lag_source_cols:
        lag_col = f"lag_{col}"
        out[lag_col] = out.groupby(traj_cols)[col].shift(1)

    out["has_lag_history"] = (
        out[[f"lag_{c}" for c in lag_source_cols]]
        .notna()
        .any(axis=1)
        .astype(int)
    )

    for col in lag_source_cols:
        lag_col = f"lag_{col}"
        out[lag_col] = out[lag_col].fillna(out[col].median())

    return out


def compute_raw_axes(df):
    out = add_lag_features(df)

    z_mean = robust_z(out["population_mean_rate_hz"])
    z_active = robust_z(out["active_unit_fraction"])
    z_l2 = robust_z(out["population_l2_rate_norm"])
    z_max = robust_z(out["population_max_rate_hz"])

    z_speed = robust_z(out["population_state_speed"])

    z_entropy = robust_z(out["population_rate_entropy"])
    z_std = robust_z(out["population_std_rate_hz"])
    z_abs_delta_mean = robust_z(out["delta_population_mean_rate_hz"].abs())
    z_abs_delta_entropy = robust_z(out["delta_population_rate_entropy"].abs())

    z_lag_mean = robust_z(out["lag_population_mean_rate_hz"])
    z_lag_active = robust_z(out["lag_active_unit_fraction"])
    z_lag_l2 = robust_z(out["lag_population_l2_rate_norm"])
    z_lag_entropy = robust_z(out["lag_population_rate_entropy"])

    out["H_raw"] = safe_mean([z_mean, z_active, z_l2, z_max])

    # Recoverability is high when the local trajectory speed is low.
    out["R_raw"] = -z_speed

    # Stability is treated as organized coherence with low dispersion and low
    # local deformation.
    out["S_raw"] = safe_mean(
        [
            z_entropy,
            -z_std,
            -z_abs_delta_mean,
            -z_abs_delta_entropy,
        ]
    )

    # Memory is the retained lagged structure available to the current state.
    # This is not yet the formal memory-kernel test. That comes next.
    out["M_raw"] = safe_mean(
        [
            z_lag_mean,
            z_lag_active,
            z_lag_l2,
            z_lag_entropy,
        ]
    ) * out["has_lag_history"].replace({0: 0.25, 1: 1.0})

    out["Phi_neural_raw"] = (
        out["H_raw"] + out["R_raw"] + out["S_raw"] + out["M_raw"]
    ) / 4.0

    return out


def residualize_axes(df):
    out = df.copy()
    used = []

    for axis in AXES:
        raw_col = f"{axis}_raw"
        y = out[raw_col].to_numpy(dtype=float)

        if used:
            X = out[[f"{a}_ortho" for a in used]].to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(X)), X])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
        else:
            resid = y.copy()

        out[f"{axis}_ortho"] = resid
        out[axis] = resid
        used.append(axis)

    corr = out[[f"{axis}_ortho" for axis in AXES]].corr()
    corr_arr = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_arr, 0.0)
    max_abs_offdiag = float(np.nanmax(np.abs(corr_arr)))

    out["Phi_neural"] = out[AXES].mean(axis=1)

    return out, max_abs_offdiag, corr


def domain_labels(series, low_q, high_q):
    low = float(series.quantile(low_q))
    high = float(series.quantile(high_q))

    def lab(x):
        if x <= low:
            return "low"
        if x >= high:
            return "high"
        return "mid"

    return series.map(lab), low, high


def add_domains(df, low_q, high_q):
    out = df.copy()
    thresholds = []

    for axis in AXES:
        labels, low, high = domain_labels(out[axis], low_q, high_q)
        out[f"{axis}_domain"] = labels
        thresholds.append(
            {
                "axis": axis,
                "low_q": low_q,
                "high_q": high_q,
                "low_threshold": low,
                "high_threshold": high,
            }
        )

    out["hrsm_domain"] = (
        out["H_domain"]
        + "_"
        + out["R_domain"]
        + "_"
        + out["S_domain"]
        + "_"
        + out["M_domain"]
    )

    return out, pd.DataFrame(thresholds)


def summarize_domains(df):
    group_cols = ["session_id", "region", "stimulus_family"]

    summary = (
        df.groupby(group_cols)
        .agg(
            H=("H", "median"),
            R=("R", "median"),
            S=("S", "median"),
            M=("M", "median"),
            Phi_neural=("Phi_neural", "median"),
            n_state_rows=("Phi_neural", "size"),
            n_presentations=("stimulus_presentation_id", "nunique"),
            mean_population_rate_hz=("population_mean_rate_hz", "mean"),
            mean_state_speed=("population_state_speed", "mean"),
            mean_active_fraction=("active_unit_fraction", "mean"),
            dominant_hrsm_domain=(
                "hrsm_domain",
                lambda x: x.value_counts().index[0],
            ),
        )
        .reset_index()
    )

    for axis in AXES:
        labels, _, _ = domain_labels(summary[axis], 0.33, 0.66)
        summary[f"{axis}_domain"] = labels

    summary["hrsm_domain"] = (
        summary["H_domain"]
        + "_"
        + summary["R_domain"]
        + "_"
        + summary["S_domain"]
        + "_"
        + summary["M_domain"]
    )

    summary["analysis_warning"] = np.where(
        summary["n_state_rows"] < 5,
        "very_small_block",
        "ok",
    )

    return summary.sort_values(["region", "stimulus_family"])


def main():
    args = parse_args()

    pop_path = Path(args.population)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pop_path)

    required = [
        "session_id",
        "stimulus_presentation_id",
        "trial_id",
        "stimulus_family",
        "region",
        "bin_index",
        "population_mean_rate_hz",
        "population_std_rate_hz",
        "population_max_rate_hz",
        "active_unit_fraction",
        "population_l2_rate_norm",
        "population_rate_entropy",
        "delta_population_mean_rate_hz",
        "delta_population_rate_entropy",
        "population_state_speed",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required population-state columns: {missing}")

    df = df.sort_values(
        ["session_id", "region", "stimulus_family", "trial_id", "bin_index"]
    ).reset_index(drop=True)

    hrsm = compute_raw_axes(df)
    hrsm, max_abs_offdiag, corr = residualize_axes(hrsm)
    hrsm, thresholds = add_domains(hrsm, args.low_q, args.high_q)
    summary = summarize_domains(hrsm)

    metrics_path = out_dir / "real_neural_hrsm_bin_level_metrics.csv"
    summary_path = out_dir / "real_neural_hrsm_domain_summary.csv"
    audit_path = out_dir / "real_orthogonality_audit.csv"
    corr_path = out_dir / "real_orthogonal_axis_correlation.csv"
    thresholds_path = out_dir / "real_hrsm_domain_thresholds.csv"

    hrsm.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)

    pd.DataFrame(
        [
            {
                "max_abs_offdiag_corr_orthogonalized": max_abs_offdiag,
                "n_rows": len(hrsm),
                "n_regions": hrsm["region"].nunique(),
                "n_stimulus_families": hrsm["stimulus_family"].nunique(),
            }
        ]
    ).to_csv(audit_path, index=False)

    corr.to_csv(corr_path)
    thresholds.to_csv(thresholds_path, index=False)

    print(f"[ok] wrote {metrics_path} rows={len(hrsm)}")
    print(f"[ok] wrote {summary_path} rows={len(summary)}")
    print(f"[ok] wrote {audit_path}")
    print(f"[ok] max_abs_offdiag_corr_orthogonalized={max_abs_offdiag:.6e}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
