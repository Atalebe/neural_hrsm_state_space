#!/usr/bin/env python3
"""
Inspect real Allen Visual Coding Neuropixels sessions.

This bridge does not build HRSM metrics yet. It checks whether AllenSDK can be
used, lists available ecephys sessions, filters the global unit table by session,
and writes a compact real-session manifest.

Raw Allen cache files must remain outside Git.
"""

from pathlib import Path
import argparse
import pandas as pd


TARGET_REGIONS = ["VISp", "VISl", "LGd", "CA1"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="data/raw/allen_neuropixels_cache",
        help="Local AllenSDK cache directory. This should remain outside Git.",
    )
    parser.add_argument(
        "--out",
        default="metadata/allen_real_session_manifest.csv",
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=20,
        help="Maximum number of sessions to inspect.",
    )
    return parser.parse_args()


def get_session_id(row):
    if "ecephys_session_id" in row.index:
        return int(row["ecephys_session_id"])
    if "id" in row.index:
        return int(row["id"])
    return int(row.name)


def main():
    args = parse_args()

    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
            EcephysProjectCache,
        )
    except ImportError as exc:
        raise SystemExit(
            "AllenSDK is not installed in the active environment. "
            "Activate .venv_allen or neural_hrsm_allen first."
        ) from exc

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.json"
    cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))

    sessions = cache.get_session_table().copy()
    sessions_reset = sessions.reset_index()

    # Project-level tables. These calls may download metadata, not full NWB sessions.
    units = cache.get_units().reset_index()

    records = []
    for _, row in sessions_reset.head(args.max_sessions).iterrows():
        session_id = get_session_id(row)

        session_units = units[units["ecephys_session_id"] == session_id].copy()

        if "ecephys_structure_acronym" in session_units.columns:
            region_col = "ecephys_structure_acronym"
        elif "structure_acronym" in session_units.columns:
            region_col = "structure_acronym"
        else:
            region_col = None

        if region_col is None:
            acronyms = []
            region_counts = {}
        else:
            acronyms = sorted(
                set(str(x) for x in session_units[region_col].dropna().unique())
            )
            region_counts = {
                region: int((session_units[region_col] == region).sum())
                for region in TARGET_REGIONS
            }

        record = {
            "ecephys_session_id": session_id,
            "status": "ok",
            "n_units": int(len(session_units)),
            "regions": ";".join(acronyms),
        }

        for region in TARGET_REGIONS:
            record[f"has_{region}"] = region in acronyms
            record[f"n_units_{region}"] = region_counts.get(region, 0)

        record["target_region_count"] = int(
            sum(1 for region in TARGET_REGIONS if region in acronyms)
        )

        records.append(record)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.DataFrame(records)
    manifest.to_csv(out, index=False)

    print(f"[ok] wrote {out} rows={len(manifest)}")
    print(manifest.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
