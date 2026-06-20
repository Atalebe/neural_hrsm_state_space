#!/usr/bin/env python3
"""
Reviewer control: state-speed memory beyond AR(p) and current-state geometry.

Question:
Does population_state_speed remain predictable from recent history after the
baseline already knows:
  1. target AR(p), i.e. recent speed itself
  2. current reduced-state magnitude/geometry at t-1
  3. distance from the region/session reduced-state centroid at t-1

This is the cheap reviewer test that can run from the current strict-v2
derived CSVs. It does not use running/pupil covariates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
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

REGION_CANDIDATES = ["region", "structure_acronym", "ecephys_structure_acronym"]
SEQ_CANDIDATES = [
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

TARGET_CANDIDATES = ["population_state_speed", "state_speed", "speed"]

GEOM_CANDIDATES = [
    "population_mean_rate_hz",
    "mean_rate_hz",
    "population_std_rate_hz",
    "std_rate_hz",
    "active_unit_fraction",
    "population_l2_rate_norm",
    "population_rate_entropy",
    "H",
    "R",
    "S",
    "M",
    "Phi_neural",
]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def find_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    lower = {c.lower(): c for c in df.columns}
    out = []
    for c in candidates:
        if c.lower() in lower:
            out.append(lower[c.lower()])
    return list(dict.fromkeys(out))


def safe_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad <= 1e-12:
        sd = x.std()
        if not np.isfinite(sd) or sd <= 1e-12:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - x.mean()) / sd
    return (x - med) / (1.4826 * mad)


def add_centroid_distance(df: pd.DataFrame, region_col: str, geom_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    zcols = []
    for c in geom_cols:
        zc = "__z_" + c.replace(" ", "_")
        out[zc] = out.groupby(region_col)[c].transform(safe_z)
        zcols.append(zc)

    dists = []
    for _, g in out.groupby(region_col):
        X = g[zcols].to_numpy(dtype=float)
        centroid = np.nanmean(X, axis=0)
        dist = np.sqrt(np.nansum((X - centroid) ** 2, axis=1))
        dists.append(pd.Series(dist, index=g.index))

    out["reduced_centroid_distance"] = pd.concat(dists).sort_index()
    out["reduced_state_magnitude"] = np.sqrt(np.nansum(out[zcols].to_numpy(dtype=float) ** 2, axis=1))
    return out


def make_lagged(
    df_region: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    seq_col: str | None,
    time_col: str | None,
    max_lag: int,
) -> pd.DataFrame:
    sort_cols = []
    if seq_col:
        sort_cols.append(seq_col)
    if time_col:
        sort_cols.append(time_col)
    if sort_cols:
        df_region = df_region.sort_values(sort_cols).copy()

    groups = df_region.groupby(seq_col, sort=False) if seq_col else [("__all__", df_region)]
    blocks = []

    for seq, g in groups:
        if time_col:
            g = g.sort_values(time_col)

        b = pd.DataFrame(index=g.index)
        b["y"] = pd.to_numeric(g[target_col], errors="coerce")
        b["sequence_id"] = str(seq)

        for lag in range(1, max_lag + 1):
            b[f"y_lag{lag}"] = b["y"].shift(lag)
            for c in feature_cols:
                safe = c.replace(" ", "_")
                b[f"{safe}_lag{lag}"] = pd.to_numeric(g[c], errors="coerce").shift(lag)

        b = b.dropna()
        if not b.empty:
            blocks.append(b)

    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def split_by_sequence_or_time(lagged: pd.DataFrame, test_frac: float, rng: np.random.Generator):
    groups = lagged["sequence_id"].astype(str).to_numpy()
    unique = np.unique(groups)

    if len(unique) >= 4:
        n_test = max(1, int(round(test_frac * len(unique))))
        test_groups = set(rng.choice(unique, size=n_test, replace=False))
        test = np.array([g in test_groups for g in groups])
        train = ~test
        if train.sum() >= 50 and test.sum() >= 50:
            return train, test

    n = len(lagged)
    cut = int(round((1 - test_frac) * n))
    cut = min(max(cut, 50), n - 50)
    train = np.zeros(n, dtype=bool)
    train[:cut] = True
    return train, ~train


def fit_r2(X_train, y_train, X_test, y_test, alpha: float) -> float:
    if len(y_train) < 10 or len(y_test) < 10 or np.std(y_test) <= 1e-12:
        return np.nan
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(r2_score(y_test, pred))


def run_one_group(lagged: pd.DataFrame, feature_cols: list[str], max_lag: int, alpha: float, rng, test_frac: float):
    train, test = split_by_sequence_or_time(lagged, test_frac, rng)
    y = lagged["y"].to_numpy(float)
    y_train, y_test = y[train], y[test]

    ar_cols = [f"y_lag{i}" for i in range(1, max_lag + 1)]

    current_cols = []
    history_cols = []

    for c in feature_cols:
        safe = c.replace(" ", "_")
        current_cols.append(f"{safe}_lag1")
        for lag in range(2, max_lag + 1):
            history_cols.append(f"{safe}_lag{lag}")

    X_ar = lagged[ar_cols].to_numpy(float)
    X_current = lagged[ar_cols + current_cols].to_numpy(float)
    X_history = lagged[ar_cols + current_cols + history_cols].to_numpy(float)

    r2_ar = fit_r2(X_ar[train], y_train, X_ar[test], y_test, alpha)
    r2_current = fit_r2(X_current[train], y_train, X_current[test], y_test, alpha)
    r2_history = fit_r2(X_history[train], y_train, X_history[test], y_test, alpha)

    return {
        "n_rows": int(len(lagged)),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "ar_r2": r2_ar,
        "ar_plus_current_geometry_r2": r2_current,
        "ar_plus_current_geometry_plus_history_r2": r2_history,
        "delta_current_geometry_over_ar": r2_current - r2_ar,
        "delta_history_over_current_geometry": r2_history - r2_current,
        "delta_history_over_ar": r2_history - r2_ar,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-root", default="results/real_allen")
    ap.add_argument("--out-dir", default="results/reviewer_tests/allen_spontaneous_strict_v2")
    ap.add_argument("--fig-dir", default="results/figures/reviewer_tests/allen_spontaneous_strict_v2")
    ap.add_argument("--lags", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20260620)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    inventory = []

    for sid in STRICT_SESSIONS:
        p = Path(args.session_root) / f"session_{sid}_spontaneous_v1" / "real_neural_hrsm_bin_level_metrics.csv"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue

        df = pd.read_csv(p)
        target_col = find_col(df, TARGET_CANDIDATES)
        region_col = find_col(df, REGION_CANDIDATES)
        seq_col = find_col(df, SEQ_CANDIDATES)
        time_col = find_col(df, TIME_CANDIDATES)
        geom_cols = find_cols(df, GEOM_CANDIDATES)

        if target_col is None or region_col is None or len(geom_cols) < 3:
            print(f"[warn] insufficient columns in {sid}")
            continue

        df = add_centroid_distance(df, region_col, geom_cols)
        feature_cols = geom_cols + ["reduced_centroid_distance", "reduced_state_magnitude"]

        inventory.append({
            "session_id": sid,
            "target_col": target_col,
            "region_col": region_col,
            "seq_col": seq_col or "",
            "time_col": time_col or "",
            "geometry_cols": ";".join(feature_cols),
            "n_rows": len(df),
        })

        for region, g in df.groupby(region_col):
            lagged = make_lagged(
                df_region=g,
                target_col=target_col,
                feature_cols=feature_cols,
                seq_col=seq_col,
                time_col=time_col,
                max_lag=args.lags,
            )

            if len(lagged) < 120:
                continue

            metrics = run_one_group(
                lagged=lagged,
                feature_cols=feature_cols,
                max_lag=args.lags,
                alpha=args.alpha,
                rng=rng,
                test_frac=args.test_frac,
            )

            rows.append({
                "session_id": sid,
                "region": region,
                "target": "population_state_speed",
                "lags": args.lags,
                "alpha": args.alpha,
                **metrics,
            })

    group = pd.DataFrame(rows)
    inv = pd.DataFrame(inventory)

    inv.to_csv(out_dir / "mean_reversion_control_column_inventory.csv", index=False)
    group.to_csv(out_dir / "mean_reversion_control_group_results.csv", index=False)

    if group.empty:
        raise SystemExit("[error] no results produced")

    session_summary = (
        group.groupby("session_id", as_index=False)
        .median(numeric_only=True)
        .sort_values("session_id")
    )
    session_summary.to_csv(out_dir / "mean_reversion_control_session_summary.csv", index=False)

    final = pd.DataFrame([{
        "n_sessions": session_summary["session_id"].nunique(),
        "median_ar_r2": session_summary["ar_r2"].median(),
        "median_ar_plus_current_geometry_r2": session_summary["ar_plus_current_geometry_r2"].median(),
        "median_ar_plus_current_geometry_plus_history_r2": session_summary["ar_plus_current_geometry_plus_history_r2"].median(),
        "median_delta_current_geometry_over_ar": session_summary["delta_current_geometry_over_ar"].median(),
        "median_delta_history_over_current_geometry": session_summary["delta_history_over_current_geometry"].median(),
        "min_delta_history_over_current_geometry": session_summary["delta_history_over_current_geometry"].min(),
        "n_sessions_history_positive": int((session_summary["delta_history_over_current_geometry"] > 0).sum()),
    }])
    final["state_speed_history_survives_mean_reversion_control"] = (
        (final["median_delta_history_over_current_geometry"] > 0)
        & (final["n_sessions_history_positive"] >= 5)
    )
    final.to_csv(out_dir / "mean_reversion_control_final_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(session_summary))
    ax.bar(x, session_summary["delta_history_over_current_geometry"].astype(float))
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(session_summary["session_id"], rotation=35, ha="right")
    ax.set_ylabel("ΔR²: history beyond AR + current geometry")
    ax.set_title("State-speed history after mean-reversion/current-state control")
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_reversion_control_state_speed_by_session.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[ok] wrote mean-reversion control outputs to", out_dir)
    print("\nFINAL SUMMARY")
    print(final.to_string(index=False))
    print("\nSESSION SUMMARY")
    print(session_summary.to_string(index=False))


if __name__ == "__main__":
    main()
