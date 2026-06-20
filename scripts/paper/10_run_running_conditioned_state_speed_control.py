#!/usr/bin/env python3
"""
Reviewer control: state-speed history after target AR, current-state geometry,
and running-speed covariates.

The baseline includes:
- target AR(p)
- lag-1 reduced-state geometry / centroid-distance controls
- current-bin running speed summaries
- lagged running speed summaries

The full model adds neural/reduced-state history lags 2..p.
If the full model does not improve over this baseline, the remaining state-speed
effect is not evidence of history beyond current-state geometry and running state.
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
SEQ_CANDIDATES = ["stimulus_presentation_id", "presentation_id", "trial_id", "trajectory_id"]
TIME_CANDIDATES = ["bin_start_time", "bin_index", "time_s", "t"]
TARGET_CANDIDATES = ["population_state_speed", "state_speed", "speed"]

GEOM_CANDIDATES = [
    "population_mean_rate_hz",
    "population_std_rate_hz",
    "active_unit_fraction",
    "population_l2_rate_norm",
    "population_rate_entropy",
    "H",
    "R",
    "S",
    "M",
    "Phi_neural",
]

RUNNING_CANDIDATES = [
    "running_speed_mean",
    "running_speed_abs_mean",
    "running_speed_max_abs",
    "running_speed_sd",
    "running_speed_center_interp",
    "running_speed_delta",
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

    pieces = []
    for _, g in out.groupby(region_col):
        X = g[zcols].to_numpy(dtype=float)
        centroid = np.nanmean(X, axis=0)
        dist = np.sqrt(np.nansum((X - centroid) ** 2, axis=1))
        pieces.append(pd.Series(dist, index=g.index))

    out["reduced_centroid_distance"] = pd.concat(pieces).sort_index()
    out["reduced_state_magnitude"] = np.sqrt(np.nansum(out[zcols].to_numpy(dtype=float) ** 2, axis=1))
    return out


def make_lagged(df_region, target_col, geom_cols, running_cols, seq_col, time_col, lags):
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

        for c in geom_cols:
            safe = c.replace(" ", "_")
            for lag in range(1, lags + 1):
                b[f"{safe}_lag{lag}"] = pd.to_numeric(g[c], errors="coerce").shift(lag)

        for c in running_cols:
            safe = c.replace(" ", "_")
            b[f"{safe}_t"] = pd.to_numeric(g[c], errors="coerce")
            for lag in range(1, lags + 1):
                b[f"{safe}_lag{lag}"] = pd.to_numeric(g[c], errors="coerce").shift(lag)

        for lag in range(1, lags + 1):
            b[f"y_lag{lag}"] = b["y"].shift(lag)

        b = b.dropna()
        if not b.empty:
            blocks.append(b)

    if not blocks:
        return pd.DataFrame()

    return pd.concat(blocks, ignore_index=True)


def split(lagged, test_frac, rng):
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


def fit_r2(X_train, y_train, X_test, y_test, alpha):
    if len(y_test) < 10 or np.std(y_test) <= 1e-12:
        return np.nan
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(r2_score(y_test, pred))


def run_group(lagged, geom_cols, running_cols, lags, alpha, rng, test_frac):
    train, test = split(lagged, test_frac, rng)
    y = lagged["y"].to_numpy(float)
    y_train, y_test = y[train], y[test]

    ar_cols = [f"y_lag{i}" for i in range(1, lags + 1)]

    geom_current = [f"{c.replace(' ', '_')}_lag1" for c in geom_cols]
    geom_history = [
        f"{c.replace(' ', '_')}_lag{lag}"
        for c in geom_cols
        for lag in range(2, lags + 1)
    ]

    running_current_and_lags = []
    for c in running_cols:
        safe = c.replace(" ", "_")
        running_current_and_lags.append(f"{safe}_t")
        running_current_and_lags.extend([f"{safe}_lag{lag}" for lag in range(1, lags + 1)])

    X_ar_cols = ar_cols
    X_geom_cols = ar_cols + geom_current
    X_behavior_cols = ar_cols + geom_current + running_current_and_lags
    X_full_cols = ar_cols + geom_current + running_current_and_lags + geom_history

    X_ar = lagged[X_ar_cols].to_numpy(float)
    X_geom = lagged[X_geom_cols].to_numpy(float)
    X_behavior = lagged[X_behavior_cols].to_numpy(float)
    X_full = lagged[X_full_cols].to_numpy(float)

    r2_ar = fit_r2(X_ar[train], y_train, X_ar[test], y_test, alpha)
    r2_geom = fit_r2(X_geom[train], y_train, X_geom[test], y_test, alpha)
    r2_behavior = fit_r2(X_behavior[train], y_train, X_behavior[test], y_test, alpha)
    r2_full = fit_r2(X_full[train], y_train, X_full[test], y_test, alpha)

    return {
        "n_rows": int(len(lagged)),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "ar_r2": r2_ar,
        "ar_plus_geometry_r2": r2_geom,
        "ar_plus_geometry_running_r2": r2_behavior,
        "ar_plus_geometry_running_history_r2": r2_full,
        "delta_geometry_over_ar": r2_geom - r2_ar,
        "delta_running_over_geometry": r2_behavior - r2_geom,
        "delta_history_over_geometry_running": r2_full - r2_behavior,
        "delta_history_over_ar": r2_full - r2_ar,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-dir", default="results/reviewer_tests/allen_spontaneous_strict_v2/behavior_augmented")
    ap.add_argument("--out-dir", default="results/reviewer_tests/allen_spontaneous_strict_v2")
    ap.add_argument("--fig-dir", default="results/figures/reviewer_tests/allen_spontaneous_strict_v2")
    ap.add_argument("--lags", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20260620)
    args = ap.parse_args()

    aug_dir = Path(args.aug_dir)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    inventory = []

    for sid in STRICT_SESSIONS:
        p = aug_dir / f"session_{sid}_bin_level_metrics_with_running.csv"
        if not p.exists():
            print("[warn] missing", p)
            continue

        df = pd.read_csv(p)

        target_col = find_col(df, TARGET_CANDIDATES)
        region_col = find_col(df, REGION_CANDIDATES)
        seq_col = find_col(df, SEQ_CANDIDATES)
        time_col = find_col(df, TIME_CANDIDATES)
        geom_cols = find_cols(df, GEOM_CANDIDATES)
        running_cols = find_cols(df, RUNNING_CANDIDATES)

        if target_col is None or region_col is None or len(geom_cols) < 3 or not running_cols:
            print(f"[warn] insufficient columns for {sid}")
            continue

        df = add_centroid_distance(df, region_col, geom_cols)
        geom_cols = geom_cols + ["reduced_centroid_distance", "reduced_state_magnitude"]

        inventory.append({
            "session_id": sid,
            "target_col": target_col,
            "region_col": region_col,
            "seq_col": seq_col or "",
            "time_col": time_col or "",
            "geom_cols": ";".join(geom_cols),
            "running_cols": ";".join(running_cols),
            "n_rows": len(df),
        })

        for region, g in df.groupby(region_col):
            lagged = make_lagged(g, target_col, geom_cols, running_cols, seq_col, time_col, args.lags)
            if len(lagged) < 120:
                continue

            metrics = run_group(lagged, geom_cols, running_cols, args.lags, args.alpha, rng, args.test_frac)
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

    inv.to_csv(out_dir / "running_conditioned_column_inventory.csv", index=False)
    group.to_csv(out_dir / "running_conditioned_state_speed_group_results.csv", index=False)

    if group.empty:
        raise SystemExit("[error] no running-conditioned results produced")

    session = (
        group.groupby("session_id", as_index=False)
        .median(numeric_only=True)
        .sort_values("session_id")
    )
    session.to_csv(out_dir / "running_conditioned_state_speed_session_summary.csv", index=False)

    final = pd.DataFrame([{
        "n_sessions": session["session_id"].nunique(),
        "median_ar_r2": session["ar_r2"].median(),
        "median_ar_plus_geometry_r2": session["ar_plus_geometry_r2"].median(),
        "median_ar_plus_geometry_running_r2": session["ar_plus_geometry_running_r2"].median(),
        "median_ar_plus_geometry_running_history_r2": session["ar_plus_geometry_running_history_r2"].median(),
        "median_delta_geometry_over_ar": session["delta_geometry_over_ar"].median(),
        "median_delta_running_over_geometry": session["delta_running_over_geometry"].median(),
        "median_delta_history_over_geometry_running": session["delta_history_over_geometry_running"].median(),
        "min_delta_history_over_geometry_running": session["delta_history_over_geometry_running"].min(),
        "n_sessions_history_positive_after_running": int((session["delta_history_over_geometry_running"] > 0).sum()),
    }])

    final["state_speed_history_survives_running_control"] = (
        (final["median_delta_history_over_geometry_running"] > 0)
        & (final["n_sessions_history_positive_after_running"] >= 5)
    )
    final.to_csv(out_dir / "running_conditioned_state_speed_final_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(session))
    ax.bar(x, session["delta_history_over_geometry_running"].astype(float))
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(session["session_id"], rotation=35, ha="right")
    ax.set_ylabel("ΔR²: history beyond AR + geometry + running")
    ax.set_title("State-speed history after running-speed control")
    fig.tight_layout()
    fig.savefig(fig_dir / "running_conditioned_state_speed_by_session.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[ok] wrote running-conditioned outputs to", out_dir)
    print("\nFINAL SUMMARY")
    print(final.to_string(index=False))
    print("\nSESSION SUMMARY")
    print(session.to_string(index=False))


if __name__ == "__main__":
    main()
