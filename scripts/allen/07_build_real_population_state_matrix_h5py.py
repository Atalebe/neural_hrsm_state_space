#!/usr/bin/env python3
"""
Build a real Allen population-state matrix from a low-memory HDF5 binned spike table.

Input:
    real_binned_spike_table.csv

Output:
    population_state_matrix.csv

Each row is a region-level population state for one session, stimulus family,
presentation/trial, and time bin. Unit-level firing rates are summarized into
compact population descriptors so the downstream HRSM proxy layer can operate
without carrying a huge unit-by-bin matrix.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binned",
        required=True,
        help="Path to real_binned_spike_table.csv.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory.",
    )
    return parser.parse_args()


def entropy_from_rates(rates):
    rates = np.asarray(rates, dtype=float)
    total = rates.sum()
    if total <= 0:
        return 0.0
    p = rates / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    args = parse_args()

    binned_path = Path(args.binned)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(binned_path)

    required = {
        "session_id",
        "ecephys_session_id",
        "stimulus_presentation_id",
        "trial_id",
        "stimulus_name",
        "stimulus_family",
        "region",
        "unit_id",
        "bin_index",
        "bin_start_time",
        "bin_end_time",
        "bin_size_sec",
        "spike_count",
        "firing_rate_hz",
    }

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    group_cols = [
        "session_id",
        "ecephys_session_id",
        "stimulus_presentation_id",
        "trial_id",
        "stimulus_name",
        "stimulus_family",
        "region",
        "bin_index",
        "bin_start_time",
        "bin_end_time",
        "bin_size_sec",
    ]

    rows = []

    for keys, sub in df.groupby(group_cols, sort=True):
        key_record = dict(zip(group_cols, keys))

        rates = sub["firing_rate_hz"].to_numpy(dtype=float)
        counts = sub["spike_count"].to_numpy(dtype=float)

        n_units = int(sub["unit_id"].nunique())
        active_fraction = float((counts > 0).mean()) if len(counts) else 0.0

        row = {
            **key_record,
            "n_units": n_units,
            "population_total_spikes": float(counts.sum()),
            "population_mean_rate_hz": float(rates.mean()) if len(rates) else 0.0,
            "population_median_rate_hz": float(np.median(rates)) if len(rates) else 0.0,
            "population_std_rate_hz": float(rates.std(ddof=0)) if len(rates) else 0.0,
            "population_max_rate_hz": float(rates.max()) if len(rates) else 0.0,
            "active_unit_fraction": active_fraction,
            "population_l2_rate_norm": float(np.linalg.norm(rates)) if len(rates) else 0.0,
            "population_rate_entropy": entropy_from_rates(rates),
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    # Within each session-region-stimulus trajectory, compute local temporal change.
    out = out.sort_values(
        ["session_id", "region", "stimulus_family", "trial_id", "bin_index"]
    ).reset_index(drop=True)

    traj_cols = ["session_id", "region", "stimulus_family", "trial_id"]

    for col in [
        "population_mean_rate_hz",
        "population_std_rate_hz",
        "active_unit_fraction",
        "population_l2_rate_norm",
        "population_rate_entropy",
    ]:
        out[f"delta_{col}"] = (
            out.groupby(traj_cols)[col]
            .diff()
            .fillna(0.0)
            .astype(float)
        )

    out["population_state_speed"] = np.sqrt(
        out[
            [
                "delta_population_mean_rate_hz",
                "delta_population_std_rate_hz",
                "delta_active_unit_fraction",
                "delta_population_l2_rate_norm",
                "delta_population_rate_entropy",
            ]
        ]
        .pow(2)
        .sum(axis=1)
    )

    out_path = out_dir / "population_state_matrix.csv"
    out.to_csv(out_path, index=False)

    summary = (
        out.groupby(["session_id", "region", "stimulus_family"])
        .agg(
            n_state_rows=("population_mean_rate_hz", "size"),
            n_presentations=("stimulus_presentation_id", "nunique"),
            mean_population_rate_hz=("population_mean_rate_hz", "mean"),
            mean_active_fraction=("active_unit_fraction", "mean"),
            mean_state_speed=("population_state_speed", "mean"),
        )
        .reset_index()
    )

    summary_path = out_dir / "population_state_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[ok] wrote {out_path} rows={len(out)}")
    print(f"[ok] wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
