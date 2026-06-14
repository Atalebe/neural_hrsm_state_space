#!/usr/bin/env python3
"""
Fit a conservative real Allen memory-kernel audit.

This script tests whether lagged population-state history improves prediction
of the next neural population state beyond a present-state baseline.

Input:
    population_state_matrix.csv

Output:
    real_memory_kernel_gain_summary.csv

This is not yet a full biological claim. It is a proof-of-path real-data
non-Markovian audit on the low-memory Allen HDF5 extraction.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


BASE_FEATURES = [
    "population_mean_rate_hz",
    "population_std_rate_hz",
    "active_unit_fraction",
    "population_l2_rate_norm",
    "population_rate_entropy",
    "population_state_speed",
]

DEFAULT_TARGET = "population_mean_rate_hz"


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
        help="Output directory.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Population-state variable to predict one step ahead.",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Lag orders to include in the memory model.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge penalty. Intercept is not penalized.",
    )
    parser.add_argument(
        "--min-supervised-rows",
        type=int,
        default=8,
        help="Minimum supervised rows required per region/family.",
    )
    return parser.parse_args()


def ridge_fit_predict(X_train, y_train, X_test, alpha):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd <= 1e-12] = 1.0

    Xtr = (X_train - mu) / sd
    Xte = (X_test - mu) / sd

    Xtr = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xte = np.column_stack([np.ones(len(Xte)), Xte])

    penalty = np.eye(Xtr.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0

    beta = np.linalg.solve(Xtr.T @ Xtr + penalty, Xtr.T @ y_train)
    return Xte @ beta


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 1e-12:
        return np.nan

    ss_res = float(((y_true - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def add_supervised_rows(df, target, lags):
    out = df.copy()
    traj_cols = ["session_id", "region", "stimulus_family", "trial_id"]

    out = out.sort_values(
        ["session_id", "region", "stimulus_family", "trial_id", "bin_index"]
    ).reset_index(drop=True)

    out[f"next_{target}"] = out.groupby(traj_cols)[target].shift(-1)

    for lag in lags:
        for col in BASE_FEATURES:
            out[f"lag{lag}_{col}"] = out.groupby(traj_cols)[col].shift(lag)

    needed = [f"next_{target}"] + BASE_FEATURES
    for lag in lags:
        needed.extend([f"lag{lag}_{col}" for col in BASE_FEATURES])

    return out.dropna(subset=needed).copy()


def leave_trial_out_cv(group, target, lags, alpha):
    trials = sorted(group["trial_id"].unique())

    if len(trials) < 2:
        return None

    baseline_features = BASE_FEATURES
    memory_features = BASE_FEATURES[:]
    for lag in lags:
        memory_features.extend([f"lag{lag}_{col}" for col in BASE_FEATURES])

    y_all = []
    yhat_base_all = []
    yhat_mem_all = []

    for test_trial in trials:
        train = group[group["trial_id"] != test_trial].copy()
        test = group[group["trial_id"] == test_trial].copy()

        if len(train) < 2 or len(test) < 1:
            continue

        y_train = train[f"next_{target}"].to_numpy(dtype=float)
        y_test = test[f"next_{target}"].to_numpy(dtype=float)

        try:
            yhat_base = ridge_fit_predict(
                train[baseline_features],
                y_train,
                test[baseline_features],
                alpha=alpha,
            )
            yhat_mem = ridge_fit_predict(
                train[memory_features],
                y_train,
                test[memory_features],
                alpha=alpha,
            )
        except np.linalg.LinAlgError:
            return None

        y_all.extend(y_test.tolist())
        yhat_base_all.extend(yhat_base.tolist())
        yhat_mem_all.extend(yhat_mem.tolist())

    if len(y_all) < 2:
        return None

    baseline_r2 = r2_score(y_all, yhat_base_all)
    memory_r2 = r2_score(y_all, yhat_mem_all)

    return {
        "baseline_r2": baseline_r2,
        "memory_r2": memory_r2,
        "memory_gain_r2": memory_r2 - baseline_r2
        if np.isfinite(baseline_r2) and np.isfinite(memory_r2)
        else np.nan,
        "n_cv_rows": len(y_all),
        "n_trials": len(trials),
    }


def main():
    args = parse_args()

    pop_path = Path(args.population)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pop_path)

    required = [
        "session_id",
        "region",
        "stimulus_family",
        "trial_id",
        "bin_index",
        args.target,
    ] + BASE_FEATURES

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sup = add_supervised_rows(df, args.target, args.lags)

    records = []

    for keys, group in sup.groupby(["session_id", "region", "stimulus_family"]):
        session_id, region, family = keys

        if len(group) < args.min_supervised_rows:
            records.append(
                {
                    "session_id": session_id,
                    "region": region,
                    "stimulus_family": family,
                    "status": "skipped_too_few_rows",
                    "n_supervised_rows": len(group),
                    "baseline_r2": np.nan,
                    "memory_r2": np.nan,
                    "memory_gain_r2": np.nan,
                    "n_cv_rows": 0,
                    "n_trials": group["trial_id"].nunique(),
                }
            )
            continue

        result = leave_trial_out_cv(
            group=group,
            target=args.target,
            lags=args.lags,
            alpha=args.alpha,
        )

        if result is None:
            records.append(
                {
                    "session_id": session_id,
                    "region": region,
                    "stimulus_family": family,
                    "status": "skipped_cv_failed",
                    "n_supervised_rows": len(group),
                    "baseline_r2": np.nan,
                    "memory_r2": np.nan,
                    "memory_gain_r2": np.nan,
                    "n_cv_rows": 0,
                    "n_trials": group["trial_id"].nunique(),
                }
            )
            continue

        records.append(
            {
                "session_id": session_id,
                "region": region,
                "stimulus_family": family,
                "status": "ok",
                "n_supervised_rows": len(group),
                **result,
            }
        )

    summary = pd.DataFrame(records).sort_values(
        ["region", "stimulus_family"]
    )

    out_path = out_dir / "real_memory_kernel_gain_summary.csv"
    summary.to_csv(out_path, index=False)

    ok = summary[summary["status"] == "ok"].copy()

    if len(ok):
        pooled = pd.DataFrame(
            [
                {
                    "n_ok_groups": len(ok),
                    "median_baseline_r2": ok["baseline_r2"].median(),
                    "median_memory_r2": ok["memory_r2"].median(),
                    "median_memory_gain_r2": ok["memory_gain_r2"].median(),
                    "mean_memory_gain_r2": ok["memory_gain_r2"].mean(),
                    "fraction_positive_gain": float((ok["memory_gain_r2"] > 0).mean()),
                }
            ]
        )
    else:
        pooled = pd.DataFrame(
            [
                {
                    "n_ok_groups": 0,
                    "median_baseline_r2": np.nan,
                    "median_memory_r2": np.nan,
                    "median_memory_gain_r2": np.nan,
                    "mean_memory_gain_r2": np.nan,
                    "fraction_positive_gain": np.nan,
                }
            ]
        )

    pooled_path = out_dir / "real_memory_kernel_pooled_summary.csv"
    pooled.to_csv(pooled_path, index=False)

    print(f"[ok] wrote {out_path} rows={len(summary)}")
    print(f"[ok] wrote {pooled_path}")
    print(summary.to_string(index=False))
    print()
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
