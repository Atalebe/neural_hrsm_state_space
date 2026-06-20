#!/usr/bin/env python3
"""
Reviewer-facing strict-v2 memory controls.

This script addresses the main methodological review points that can be tested
from existing strict-v2 derived CSVs:

1. Baseline R2 context.
2. AR(p) / target-autocorrelation controls.
3. Incremental value of organizational lag features over AR(p).
4. Incremental value of HRSM lag features over AR(p).
5. Joint-row shuffled-lag nulls with configurable permutation count.
6. Session-blocked summaries to avoid treating all 24 region-groups as
   independent primary evidence.

The script does not condition on running speed or pupil because those covariates
are not guaranteed to be present in the derived HRSM CSVs. A separate inventory
script should check behavioral-covariate availability before running that control.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STRICT_SESSIONS = [
    "715093703",
    "719161530",
    "750749662",
    "751348571",
    "755434585",
    "756029989",
]

TARGET_CANDIDATES = OrderedDict([
    ("active_unit_fraction", ["active_unit_fraction", "fraction_active", "active_fraction"]),
    ("population_rate_entropy", ["population_rate_entropy", "rate_entropy", "entropy"]),
    ("population_mean_rate_hz", ["population_mean_rate_hz", "mean_rate_hz", "population_mean_rate", "mean_rate"]),
    ("population_l2_rate_norm", ["population_l2_rate_norm", "l2_rate_norm", "population_l2"]),
    ("population_std_rate_hz", ["population_std_rate_hz", "std_rate_hz", "population_std_rate", "rate_std"]),
    ("population_state_speed", ["population_state_speed", "state_speed", "speed"]),
])

HRSM_CANDIDATES = OrderedDict([
    ("H", ["H"]),
    ("R", ["R"]),
    ("S", ["S"]),
    ("M", ["M"]),
    ("Phi_neural", ["Phi_neural", "Phi", "phi_neural"]),
])

REGION_CANDIDATES = ["region", "structure_acronym", "ecephys_structure_acronym"]
SEQUENCE_CANDIDATES = [
    "presentation_id",
    "spontaneous_presentation_id",
    "stimulus_presentation_id",
    "presentation_index",
    "trial_id",
    "trajectory_id",
]
TIME_CANDIDATES = [
    "bin_index",
    "time_bin",
    "time_bin_index",
    "bin_start_s",
    "bin_start_time",
    "time_s",
    "t",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--session-root", default="results/real_allen")
    p.add_argument("--out-dir", default="results/reviewer_tests/allen_spontaneous_strict_v2")
    p.add_argument("--fig-dir", default="results/figures/reviewer_tests/allen_spontaneous_strict_v2")
    p.add_argument("--lags", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--n-shuffles", type=int, default=1000)
    p.add_argument("--min-rows", type=int, default=120)
    p.add_argument("--test-frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=20260620)
    return p.parse_args()


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def canonical_columns(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], str, str | None, str | None]:
    target_cols = {}
    for canonical, candidates in TARGET_CANDIDATES.items():
        c = find_col(df, candidates)
        if c is not None:
            target_cols[canonical] = c

    hrsm_cols = {}
    for canonical, candidates in HRSM_CANDIDATES.items():
        c = find_col(df, candidates)
        if c is not None:
            hrsm_cols[canonical] = c

    region_col = find_col(df, REGION_CANDIDATES)
    if region_col is None:
        raise ValueError("No region column found. Available columns: " + ", ".join(df.columns))

    seq_col = find_col(df, SEQUENCE_CANDIDATES)
    time_col = find_col(df, TIME_CANDIDATES)

    return target_cols, hrsm_cols, region_col, seq_col, time_col


def read_session(session_root: Path, sid: str) -> pd.DataFrame:
    p = session_root / f"session_{sid}_spontaneous_v1" / "real_neural_hrsm_bin_level_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["session_id"] = sid
    return df


def make_lagged(
    df_region: pd.DataFrame,
    target_col: str,
    extra_cols: list[str],
    seq_col: str | None,
    time_col: str | None,
    max_lag: int,
) -> pd.DataFrame:
    sort_cols = []
    if seq_col is not None:
        sort_cols.append(seq_col)
    if time_col is not None:
        sort_cols.append(time_col)

    if sort_cols:
        df_region = df_region.sort_values(sort_cols).copy()
    else:
        df_region = df_region.copy()

    groups = df_region.groupby(seq_col, sort=False) if seq_col is not None else [("__all__", df_region)]

    blocks = []
    for seq_value, g in groups:
        g = g.copy()
        if time_col is not None:
            g = g.sort_values(time_col)

        block = pd.DataFrame(index=g.index)
        block["y"] = pd.to_numeric(g[target_col], errors="coerce")
        block["sequence_id"] = str(seq_value)

        for lag in range(1, max_lag + 1):
            block[f"y_lag{lag}"] = block["y"].shift(lag)
            for c in extra_cols:
                safe = c.replace(" ", "_")
                block[f"{safe}_lag{lag}"] = pd.to_numeric(g[c], errors="coerce").shift(lag)

        block = block.dropna()
        if not block.empty:
            blocks.append(block)

    if not blocks:
        return pd.DataFrame()

    return pd.concat(blocks, ignore_index=True)


def split_rows(lagged: pd.DataFrame, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    groups = lagged["sequence_id"].astype(str).values
    unique = np.unique(groups)

    if len(unique) >= 4:
        n_test = max(1, int(round(test_frac * len(unique))))
        test_groups = set(rng.choice(unique, size=n_test, replace=False))
        test_mask = np.array([g in test_groups for g in groups])
        train_mask = ~test_mask
        if train_mask.sum() > 20 and test_mask.sum() > 20:
            return train_mask, test_mask

    n = len(lagged)
    cut = int(round((1.0 - test_frac) * n))
    cut = min(max(cut, 20), n - 20)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:cut] = True
    test_mask = ~train_mask
    return train_mask, test_mask


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return np.nan
    if np.nanstd(y_true) <= 1e-12:
        return np.nan
    return float(r2_score(y_true, y_pred))


def fit_r2(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, alpha: float) -> float:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return safe_r2(y_test, pred)


def shuffle_delta(
    X_ar_train: np.ndarray,
    X_extra_train: np.ndarray,
    y_train: np.ndarray,
    X_ar_test: np.ndarray,
    X_extra_test: np.ndarray,
    y_test: np.ndarray,
    r2_ar: float,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    perm_train = rng.permutation(len(X_extra_train))
    perm_test = rng.permutation(len(X_extra_test))

    X_train = np.concatenate([X_ar_train, X_extra_train[perm_train]], axis=1)
    X_test = np.concatenate([X_ar_test, X_extra_test[perm_test]], axis=1)

    r2_shuf = fit_r2(X_train, y_train, X_test, y_test, alpha)
    return r2_shuf - r2_ar


def run_group_models(
    lagged: pd.DataFrame,
    ar_cols: list[str],
    org_extra_cols: list[str],
    hrsm_extra_cols: list[str],
    alpha: float,
    n_shuffles: int,
    test_frac: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    train_mask, test_mask = split_rows(lagged, test_frac, rng)

    y = lagged["y"].to_numpy(dtype=float)
    y_train = y[train_mask]
    y_test = y[test_mask]

    X_ar = lagged[ar_cols].to_numpy(dtype=float)
    X_org_extra = lagged[org_extra_cols].to_numpy(dtype=float) if org_extra_cols else np.empty((len(lagged), 0))
    X_hrsm_extra = lagged[hrsm_extra_cols].to_numpy(dtype=float) if hrsm_extra_cols else np.empty((len(lagged), 0))
    X_full_extra = np.concatenate([X_org_extra, X_hrsm_extra], axis=1)

    X_ar_train, X_ar_test = X_ar[train_mask], X_ar[test_mask]
    X_org_train, X_org_test = X_org_extra[train_mask], X_org_extra[test_mask]
    X_hrsm_train, X_hrsm_test = X_hrsm_extra[train_mask], X_hrsm_extra[test_mask]
    X_full_train, X_full_test = X_full_extra[train_mask], X_full_extra[test_mask]

    r2_mean = 0.0
    r2_ar = fit_r2(X_ar_train, y_train, X_ar_test, y_test, alpha)

    if X_org_extra.shape[1]:
        r2_org = fit_r2(
            np.concatenate([X_ar_train, X_org_train], axis=1),
            y_train,
            np.concatenate([X_ar_test, X_org_test], axis=1),
            y_test,
            alpha,
        )
    else:
        r2_org = np.nan

    if X_hrsm_extra.shape[1]:
        r2_hrsm = fit_r2(
            np.concatenate([X_ar_train, X_hrsm_train], axis=1),
            y_train,
            np.concatenate([X_ar_test, X_hrsm_test], axis=1),
            y_test,
            alpha,
        )
    else:
        r2_hrsm = np.nan

    if X_full_extra.shape[1]:
        r2_full = fit_r2(
            np.concatenate([X_ar_train, X_full_train], axis=1),
            y_train,
            np.concatenate([X_ar_test, X_full_test], axis=1),
            y_test,
            alpha,
        )
    else:
        r2_full = np.nan

    delta_org = r2_org - r2_ar if np.isfinite(r2_org) and np.isfinite(r2_ar) else np.nan
    delta_hrsm = r2_hrsm - r2_ar if np.isfinite(r2_hrsm) and np.isfinite(r2_ar) else np.nan
    delta_full = r2_full - r2_ar if np.isfinite(r2_full) and np.isfinite(r2_ar) else np.nan

    shuffle_full = []
    if X_full_extra.shape[1] and np.isfinite(r2_ar):
        for _ in range(n_shuffles):
            shuffle_full.append(
                shuffle_delta(
                    X_ar_train,
                    X_full_train,
                    y_train,
                    X_ar_test,
                    X_full_test,
                    y_test,
                    r2_ar,
                    alpha,
                    rng,
                )
            )

    shuffle_full = np.array(shuffle_full, dtype=float)
    shuffle_full = shuffle_full[np.isfinite(shuffle_full)]

    if len(shuffle_full) and np.isfinite(delta_full):
        p_full = (1.0 + np.sum(shuffle_full >= delta_full)) / (len(shuffle_full) + 1.0)
        controlled_full = delta_full - float(np.mean(shuffle_full))
        shuffle_mean = float(np.mean(shuffle_full))
        shuffle_median = float(np.median(shuffle_full))
        n_eff = int(len(shuffle_full))
    else:
        p_full = np.nan
        controlled_full = np.nan
        shuffle_mean = np.nan
        shuffle_median = np.nan
        n_eff = 0

    return {
        "n_rows": int(len(lagged)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "mean_model_r2_reference": r2_mean,
        "ar_r2": r2_ar,
        "org_lag_r2": r2_org,
        "hrsm_lag_r2": r2_hrsm,
        "full_lag_r2": r2_full,
        "delta_org_over_ar": delta_org,
        "delta_hrsm_over_ar": delta_hrsm,
        "delta_full_over_ar": delta_full,
        "shuffle_delta_full_mean": shuffle_mean,
        "shuffle_delta_full_median": shuffle_median,
        "controlled_delta_full_over_ar": controlled_full,
        "empirical_p_full_over_ar": p_full,
        "n_shuffles_effective": n_eff,
    }


def main() -> None:
    args = parse_args()
    session_root = Path(args.session_root)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    rows = []
    column_inventory = []

    for sid in STRICT_SESSIONS:
        df = read_session(session_root, sid)
        target_cols, hrsm_cols, region_col, seq_col, time_col = canonical_columns(df)

        column_inventory.append({
            "session_id": sid,
            "target_columns_found": ";".join(f"{k}:{v}" for k, v in target_cols.items()),
            "hrsm_columns_found": ";".join(f"{k}:{v}" for k, v in hrsm_cols.items()),
            "region_col": region_col,
            "sequence_col": seq_col or "",
            "time_col": time_col or "",
            "n_rows": len(df),
        })

        if len(target_cols) < 2:
            print(f"[warn] session {sid}: too few target columns found; skipping")
            continue

        for region, df_region in df.groupby(region_col):
            for target_name, target_col in target_cols.items():
                other_target_cols = [c for name, c in target_cols.items() if name != target_name]
                hrsm_feature_cols = list(hrsm_cols.values())
                extra_cols = other_target_cols + hrsm_feature_cols

                lagged = make_lagged(
                    df_region=df_region,
                    target_col=target_col,
                    extra_cols=extra_cols,
                    seq_col=seq_col,
                    time_col=time_col,
                    max_lag=args.lags,
                )

                if len(lagged) < args.min_rows:
                    continue

                ar_cols = [f"y_lag{i}" for i in range(1, args.lags + 1)]

                org_extra_cols = []
                for c in other_target_cols:
                    safe = c.replace(" ", "_")
                    org_extra_cols.extend([f"{safe}_lag{i}" for i in range(1, args.lags + 1)])

                hrsm_extra_cols = []
                for c in hrsm_feature_cols:
                    safe = c.replace(" ", "_")
                    hrsm_extra_cols.extend([f"{safe}_lag{i}" for i in range(1, args.lags + 1)])

                try:
                    metrics = run_group_models(
                        lagged=lagged,
                        ar_cols=ar_cols,
                        org_extra_cols=org_extra_cols,
                        hrsm_extra_cols=hrsm_extra_cols,
                        alpha=args.alpha,
                        n_shuffles=args.n_shuffles,
                        test_frac=args.test_frac,
                        rng=rng,
                    )
                except Exception as e:
                    print(f"[warn] failed {sid} {region} {target_name}: {e}")
                    continue

                rows.append({
                    "session_id": sid,
                    "region": region,
                    "target": target_name,
                    "lags": args.lags,
                    "alpha": args.alpha,
                    **metrics,
                })

    group = pd.DataFrame(rows)
    inventory = pd.DataFrame(column_inventory)

    inventory.to_csv(out_dir / "reviewer_test_column_inventory.csv", index=False)
    group.to_csv(out_dir / "reviewer_test_ar_markov_group_results.csv", index=False)

    if group.empty:
        raise SystemExit("[error] no reviewer-test results were produced")

    numeric_cols = [
        "n_rows",
        "n_train",
        "n_test",
        "ar_r2",
        "org_lag_r2",
        "hrsm_lag_r2",
        "full_lag_r2",
        "delta_org_over_ar",
        "delta_hrsm_over_ar",
        "delta_full_over_ar",
        "shuffle_delta_full_mean",
        "shuffle_delta_full_median",
        "controlled_delta_full_over_ar",
        "empirical_p_full_over_ar",
        "n_shuffles_effective",
    ]

    target_summary = (
        group.groupby("target", as_index=False)[numeric_cols]
        .median(numeric_only=True)
        .sort_values("controlled_delta_full_over_ar", ascending=False)
    )
    target_summary["reviewer_rank"] = np.arange(1, len(target_summary) + 1)
    target_summary.to_csv(out_dir / "reviewer_test_ar_markov_target_summary.csv", index=False)

    session_blocked = (
        group.groupby(["session_id", "target"], as_index=False)[numeric_cols]
        .median(numeric_only=True)
    )
    session_blocked.to_csv(out_dir / "reviewer_test_session_blocked_target_summary.csv", index=False)

    target_session_summary = (
        session_blocked.groupby("target", as_index=False)
        .agg(
            n_sessions=("session_id", "nunique"),
            median_session_delta_full_over_ar=("delta_full_over_ar", "median"),
            median_session_controlled_delta_full_over_ar=("controlled_delta_full_over_ar", "median"),
            min_session_controlled_delta_full_over_ar=("controlled_delta_full_over_ar", "min"),
            n_sessions_positive_controlled=("controlled_delta_full_over_ar", lambda x: int(np.sum(np.asarray(x) > 0))),
            median_session_ar_r2=("ar_r2", "median"),
            median_session_full_lag_r2=("full_lag_r2", "median"),
        )
        .sort_values("median_session_controlled_delta_full_over_ar", ascending=False)
    )
    target_session_summary["session_blocked_rank"] = np.arange(1, len(target_session_summary) + 1)
    target_session_summary.to_csv(out_dir / "reviewer_test_session_blocked_final_summary.csv", index=False)

    top_two = target_session_summary["target"].head(2).tolist()
    flags = pd.DataFrame([
        {
            "test": "top_two_after_ar_control_are_recruitment_and_entropy",
            "passed": set(top_two) == {"active_unit_fraction", "population_rate_entropy"},
            "detail": ",".join(top_two),
        },
        {
            "test": "active_entropy_positive_in_all_sessions_after_ar_control",
            "passed": bool(
                target_session_summary[
                    target_session_summary["target"].isin(["active_unit_fraction", "population_rate_entropy"])
                ]["min_session_controlled_delta_full_over_ar"].min() > 0
            ),
            "detail": "Uses session-blocked controlled delta over AR(p).",
        },
        {
            "test": "hrsm_lags_add_over_target_ar",
            "passed": bool(group["delta_hrsm_over_ar"].median() > 0),
            "detail": f"Median HRSM-lag delta over AR(p) = {group['delta_hrsm_over_ar'].median():.6g}",
        },
    ])
    flags.to_csv(out_dir / "reviewer_test_flags.csv", index=False)

    # Figure: session-blocked controlled delta by target.
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labels = target_session_summary["target"].str.replace("_", " ").tolist()
    values = target_session_summary["median_session_controlled_delta_full_over_ar"].astype(float).values
    ax.bar(np.arange(len(values)), values)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Median session-blocked controlled ΔR² over AR(p)")
    ax.set_title("Reviewer AR(p) control: incremental memory after target autocorrelation")
    fig.tight_layout()
    fig.savefig(fig_dir / "reviewer_test_ar_markov_target_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[ok] wrote reviewer-test outputs to", out_dir)
    print("\nTARGET SUMMARY")
    print(target_summary.to_string(index=False))
    print("\nSESSION-BLOCKED SUMMARY")
    print(target_session_summary.to_string(index=False))
    print("\nFLAGS")
    print(flags.to_string(index=False))


if __name__ == "__main__":
    main()
