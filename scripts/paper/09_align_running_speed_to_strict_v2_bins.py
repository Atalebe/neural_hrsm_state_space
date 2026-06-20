#!/usr/bin/env python3
"""
Align Allen NWB running speed to strict-v2 HRSM bin-level rows.

Inputs:
- results/real_allen/session_<sid>_spontaneous_v1/real_neural_hrsm_bin_level_metrics.csv
- data/raw/allen_neuropixels_cache/session_<sid>/session_<sid>.nwb

Output:
- results/reviewer_tests/allen_spontaneous_strict_v2/behavior_augmented/
  session_<sid>_bin_level_metrics_with_running.csv

The output keeps the original HRSM bin table and adds running-speed summaries
for each [bin_start_time, bin_end_time] interval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


STRICT_SESSIONS = [
    "715093703",
    "719161530",
    "750749662",
    "751348571",
    "755434585",
    "756029989",
]


RUNNING_DATA_PATH = "processing/running/running_speed/data"
RUNNING_TS_PATH = "processing/running/running_speed/timestamps"


def find_nwb(raw_root: Path, sid: str) -> Path:
    candidates = [
        raw_root / "allen_neuropixels_cache" / f"session_{sid}" / f"session_{sid}.nwb",
        raw_root / f"session_{sid}" / f"session_{sid}.nwb",
    ]

    for p in candidates:
        if p.exists():
            return p

    found = sorted(raw_root.glob(f"**/*{sid}*.nwb"), key=lambda x: x.stat().st_size, reverse=True)
    if found:
        return found[0]

    raise FileNotFoundError(f"No NWB found for session {sid} under {raw_root}")


def load_running(nwb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(nwb_path, "r") as f:
        if RUNNING_DATA_PATH not in f or RUNNING_TS_PATH not in f:
            raise KeyError(f"Missing running-speed paths in {nwb_path}")
        data = np.asarray(f[RUNNING_DATA_PATH], dtype=float)
        ts = np.asarray(f[RUNNING_TS_PATH], dtype=float)

    good = np.isfinite(data) & np.isfinite(ts)
    data = data[good]
    ts = ts[good]

    order = np.argsort(ts)
    return ts[order], data[order]


def summarize_interval(ts: np.ndarray, data: np.ndarray, start: float, end: float) -> dict[str, float]:
    start = float(start)
    end = float(end)
    mid = 0.5 * (start + end)

    i0 = np.searchsorted(ts, start, side="left")
    i1 = np.searchsorted(ts, end, side="right")

    if i1 > i0:
        vals = data[i0:i1]
        vals = vals[np.isfinite(vals)]
    else:
        vals = np.array([], dtype=float)

    center = float(np.interp(mid, ts, data)) if len(ts) else np.nan
    start_interp = float(np.interp(start, ts, data)) if len(ts) else np.nan
    end_interp = float(np.interp(end, ts, data)) if len(ts) else np.nan

    if len(vals):
        mean = float(np.mean(vals))
        abs_mean = float(np.mean(np.abs(vals)))
        max_abs = float(np.max(np.abs(vals)))
        sd = float(np.std(vals))
        n = int(len(vals))
    else:
        mean = center
        abs_mean = abs(center) if np.isfinite(center) else np.nan
        max_abs = abs(center) if np.isfinite(center) else np.nan
        sd = 0.0 if np.isfinite(center) else np.nan
        n = 0

    return {
        "running_speed_mean": mean,
        "running_speed_abs_mean": abs_mean,
        "running_speed_max_abs": max_abs,
        "running_speed_sd": sd,
        "running_speed_center_interp": center,
        "running_speed_start_interp": start_interp,
        "running_speed_end_interp": end_interp,
        "running_speed_delta": end_interp - start_interp if np.isfinite(end_interp) and np.isfinite(start_interp) else np.nan,
        "running_speed_n_samples": n,
        "running_speed_available": bool(np.isfinite(center)),
    }


def align_session(sid: str, raw_root: Path, session_root: Path, out_dir: Path) -> dict:
    nwb = find_nwb(raw_root, sid)
    ts, running = load_running(nwb)

    in_csv = session_root / f"session_{sid}_spontaneous_v1" / "real_neural_hrsm_bin_level_metrics.csv"
    if not in_csv.exists():
        raise FileNotFoundError(in_csv)

    df = pd.read_csv(in_csv)

    if "bin_start_time" not in df.columns or "bin_end_time" not in df.columns:
        raise KeyError(f"{in_csv} lacks bin_start_time/bin_end_time")

    summaries = [
        summarize_interval(ts, running, s, e)
        for s, e in zip(df["bin_start_time"].values, df["bin_end_time"].values)
    ]
    run_df = pd.DataFrame(summaries)

    out = pd.concat([df.reset_index(drop=True), run_df.reset_index(drop=True)], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"session_{sid}_bin_level_metrics_with_running.csv"
    out.to_csv(out_csv, index=False)

    return {
        "session_id": sid,
        "nwb_file": str(nwb),
        "input_rows": len(df),
        "output_rows": len(out),
        "running_n_samples_total": len(ts),
        "running_time_min": float(np.min(ts)) if len(ts) else np.nan,
        "running_time_max": float(np.max(ts)) if len(ts) else np.nan,
        "bin_time_min": float(df["bin_start_time"].min()),
        "bin_time_max": float(df["bin_end_time"].max()),
        "fraction_bins_with_running_samples": float((run_df["running_speed_n_samples"] > 0).mean()),
        "fraction_bins_with_running_available": float(run_df["running_speed_available"].mean()),
        "running_abs_mean_over_bins": float(run_df["running_speed_abs_mean"].mean()),
        "out_csv": str(out_csv),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--session-root", default="results/real_allen")
    ap.add_argument("--out-dir", default="results/reviewer_tests/allen_spontaneous_strict_v2/behavior_augmented")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    session_root = Path(args.session_root)
    out_dir = Path(args.out_dir)

    rows = []
    for sid in STRICT_SESSIONS:
        print(f"[run] aligning running speed for session {sid}")
        rows.append(align_session(sid, raw_root, session_root, out_dir))

    summary = pd.DataFrame(rows)
    summary_path = out_dir.parent / "running_alignment_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("[ok] wrote", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
