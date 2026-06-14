#!/usr/bin/env python3
"""
Target-variable sweep for real Allen memory-kernel controls.

This script repeats the real Allen memory audit and shuffled-lag negative control
across multiple target variables.

Question:
Does lagged neural history improve prediction only for mean firing rate, or more
strongly for population-state speed, active-unit recruitment, L2 norm, entropy,
or firing-rate dispersion?

Outputs:
    real_memory_target_sweep_group_summary.csv
    real_memory_target_sweep_target_summary.csv
    real_memory_target_sweep_shuffles.csv

Interpretation rule:
A target is more interesting if observed gains are positive AND observed gains
exceed shuffled-lag controls. A raw positive gain alone is not enough.
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

DEFAULT_TARGETS = [
    "population_mean_rate_hz",
    "population_std_rate_hz",
    "active_unit_fraction",
    "population_l2_rate_norm",
    "population_rate_entropy",
    "population_state_speed",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-supervised-rows", type=int, default=20)
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=271828)
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


def feature_sets(lags):
    baseline_features = BASE_FEATURES[:]
    lag_features = []
    for lag in lags:
        lag_features.extend([f"lag{lag}_{col}" for col in BASE_FEATURES])
    memory_features = baseline_features + lag_features
    return baseline_features, lag_features, memory_features


def leave_trial_out_cv(group, target, baseline_features, memory_features, alpha):
    trials = sorted(group["trial_id"].unique())

    if len(trials) < 2:
        return None

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


def shuffled_lag_group(group, lag_features, rng):
    out = group.copy()
    perm = rng.permutation(len(out))
    lag_values = out[lag_features].to_numpy(copy=True)
    out.loc[:, lag_features] = lag_values[perm, :]
    return out


def run_one_target(df, target, lags, alpha, min_supervised_rows, n_shuffles, rng):
    sup = add_supervised_rows(df, target, lags)
    baseline_features, lag_features, memory_features = feature_sets(lags)

    base_groups = (
        df[["session_id", "region", "stimulus_family"]]
        .drop_duplicates()
        .sort_values(["region", "stimulus_family"])
    )

    records = []
    shuffle_records = []

    for _, gmeta in base_groups.iterrows():
        session_id = gmeta["session_id"]
        region = gmeta["region"]
        family = gmeta["stimulus_family"]

        group = sup[
            (sup["session_id"] == session_id)
            & (sup["region"] == region)
            & (sup["stimulus_family"] == family)
        ].copy()

        base_record = {
            "target": target,
            "session_id": session_id,
            "region": region,
            "stimulus_family": family,
            "n_supervised_rows": len(group),
            "n_trials": group["trial_id"].nunique() if len(group) else 0,
        }

        if len(group) < min_supervised_rows:
            records.append(
                {
                    **base_record,
                    "status": "skipped_too_few_rows",
                    "observed_baseline_r2": np.nan,
                    "observed_memory_r2": np.nan,
                    "observed_memory_gain_r2": np.nan,
                    "shuffle_gain_mean": np.nan,
                    "shuffle_gain_sd": np.nan,
                    "observed_minus_shuffle_mean": np.nan,
                    "empirical_p_shuffle_ge_observed": np.nan,
                    "fraction_shuffle_positive": np.nan,
                    "n_valid_shuffles": 0,
                }
            )
            continue

        obs = leave_trial_out_cv(
            group=group,
            target=target,
            baseline_features=baseline_features,
            memory_features=memory_features,
            alpha=alpha,
        )

        if obs is None:
            records.append(
                {
                    **base_record,
                    "status": "skipped_cv_failed",
                    "observed_baseline_r2": np.nan,
                    "observed_memory_r2": np.nan,
                    "observed_memory_gain_r2": np.nan,
                    "shuffle_gain_mean": np.nan,
                    "shuffle_gain_sd": np.nan,
                    "observed_minus_shuffle_mean": np.nan,
                    "empirical_p_shuffle_ge_observed": np.nan,
                    "fraction_shuffle_positive": np.nan,
                    "n_valid_shuffles": 0,
                }
            )
            continue

        gains = []

        for i in range(n_shuffles):
            shuf = shuffled_lag_group(group, lag_features, rng)

            res = leave_trial_out_cv(
                group=shuf,
                target=target,
                baseline_features=baseline_features,
                memory_features=memory_features,
                alpha=alpha,
            )

            if res is None or not np.isfinite(res["memory_gain_r2"]):
                continue

            gain = float(res["memory_gain_r2"])
            gains.append(gain)

            shuffle_records.append(
                {
                    "target": target,
                    "session_id": session_id,
                    "region": region,
                    "stimulus_family": family,
                    "shuffle_id": i,
                    "shuffle_memory_gain_r2": gain,
                }
            )

        gains = np.asarray(gains, dtype=float)

        if len(gains):
            shuffle_mean = float(gains.mean())
            shuffle_sd = float(gains.std(ddof=1)) if len(gains) > 1 else 0.0
            emp_p = float((np.sum(gains >= obs["memory_gain_r2"]) + 1) / (len(gains) + 1))
            frac_pos = float((gains > 0).mean())
            obs_minus = float(obs["memory_gain_r2"] - shuffle_mean)
        else:
            shuffle_mean = np.nan
            shuffle_sd = np.nan
            emp_p = np.nan
            frac_pos = np.nan
            obs_minus = np.nan

        records.append(
            {
                **base_record,
                "status": "ok",
                "observed_baseline_r2": obs["baseline_r2"],
                "observed_memory_r2": obs["memory_r2"],
                "observed_memory_gain_r2": obs["memory_gain_r2"],
                "shuffle_gain_mean": shuffle_mean,
                "shuffle_gain_sd": shuffle_sd,
                "observed_minus_shuffle_mean": obs_minus,
                "empirical_p_shuffle_ge_observed": emp_p,
                "fraction_shuffle_positive": frac_pos,
                "n_valid_shuffles": int(len(gains)),
            }
        )

    return pd.DataFrame(records), pd.DataFrame(shuffle_records)


def summarize_targets(group_summary):
    ok = group_summary[group_summary["status"] == "ok"].copy()

    if ok.empty:
        return pd.DataFrame()

    rows = []

    for target, sub in ok.groupby("target"):
        rows.append(
            {
                "target": target,
                "n_ok_groups": len(sub),
                "median_baseline_r2": sub["observed_baseline_r2"].median(),
                "median_memory_r2": sub["observed_memory_r2"].median(),
                "median_observed_gain": sub["observed_memory_gain_r2"].median(),
                "mean_observed_gain": sub["observed_memory_gain_r2"].mean(),
                "fraction_observed_positive": float((sub["observed_memory_gain_r2"] > 0).mean()),
                "median_shuffle_gain_mean": sub["shuffle_gain_mean"].median(),
                "mean_shuffle_gain_mean": sub["shuffle_gain_mean"].mean(),
                "median_observed_minus_shuffle": sub["observed_minus_shuffle_mean"].median(),
                "mean_observed_minus_shuffle": sub["observed_minus_shuffle_mean"].mean(),
                "fraction_observed_above_shuffle_mean": float((sub["observed_minus_shuffle_mean"] > 0).mean()),
                "median_empirical_p": sub["empirical_p_shuffle_ge_observed"].median(),
                "min_empirical_p": sub["empirical_p_shuffle_ge_observed"].min(),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        [
            "median_observed_minus_shuffle",
            "fraction_observed_above_shuffle_mean",
            "median_observed_gain",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    out["rank_by_controlled_memory"] = np.arange(1, len(out) + 1)
    return out


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
    ] + BASE_FEATURES

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    target_missing = [t for t in args.targets if t not in df.columns]
    if target_missing:
        raise ValueError(f"Requested targets not found in population matrix: {target_missing}")

    rng = np.random.default_rng(args.seed)

    all_groups = []
    all_shuffles = []

    for target in args.targets:
        print(f"[info] running target={target}")
        group_summary, shuffle_rows = run_one_target(
            df=df,
            target=target,
            lags=args.lags,
            alpha=args.alpha,
            min_supervised_rows=args.min_supervised_rows,
            n_shuffles=args.n_shuffles,
            rng=rng,
        )
        all_groups.append(group_summary)
        all_shuffles.append(shuffle_rows)

    group_summary = pd.concat(all_groups, axis=0, ignore_index=True)
    shuffles = pd.concat(all_shuffles, axis=0, ignore_index=True)
    target_summary = summarize_targets(group_summary)

    group_path = out_dir / "real_memory_target_sweep_group_summary.csv"
    target_path = out_dir / "real_memory_target_sweep_target_summary.csv"
    shuf_path = out_dir / "real_memory_target_sweep_shuffles.csv"

    group_summary.to_csv(group_path, index=False)
    target_summary.to_csv(target_path, index=False)
    shuffles.to_csv(shuf_path, index=False)

    print(f"[ok] wrote {group_path} rows={len(group_summary)}")
    print(f"[ok] wrote {target_path} rows={len(target_summary)}")
    print(f"[ok] wrote {shuf_path} rows={len(shuffles)}")
    print()
    print(target_summary.to_string(index=False))


if __name__ == "__main__":
    main()
