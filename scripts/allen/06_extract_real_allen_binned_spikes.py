#!/usr/bin/env python3
"""
Extract real Allen Visual Coding Neuropixels binned spike table.

This script is the first real-data bridge after the session-inspection step.
It loads one Allen ecephys session, selects units from target regions, bins
spike times, assigns coarse stimulus-family labels, and exports a long-form
binned spike table compatible with the existing Neural HRSM population-state
pipeline.

Scientific role:
- Replace the synthetic binned spike table with real Allen Neuropixels activity.
- Keep the output schema close to the synthetic scaffold.
- Start conservatively with one session and four target regions:
  VISp, VISl, LGd, CA1.

Raw Allen cache files must remain outside Git.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


TARGET_REGIONS_DEFAULT = ["VISp", "VISl", "LGd", "CA1"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-id",
        type=int,
        default=715093703,
        help="Allen ecephys session ID to extract.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/raw/allen_neuropixels_cache",
        help="AllenSDK cache directory. Must remain outside Git.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/interim/allen_neuropixels_real/session_715093703",
        help="Output directory for extracted real binned spike table.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=TARGET_REGIONS_DEFAULT,
        help="Target structure acronyms to include.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.25,
        help="Spike bin size in seconds.",
    )
    parser.add_argument(
        "--max-units-per-region",
        type=int,
        default=80,
        help="Maximum number of units retained per region.",
    )
    parser.add_argument(
        "--max-presentations-per-family",
        type=int,
        default=80,
        help="Maximum stimulus presentations retained per coarse family.",
    )
    parser.add_argument(
        "--stimulus-families",
        nargs="+",
        default=["drifting_gratings", "natural_scenes", "spontaneous"],
        help="Coarse stimulus families to retain.",
    )
    parser.add_argument(
        "--quality-min-isi-violations",
        type=float,
        default=0.5,
        help="Maximum ISI violations allowed when available.",
    )
    parser.add_argument(
        "--quality-max-amplitude-cutoff",
        type=float,
        default=0.1,
        help="Maximum amplitude cutoff allowed when available.",
    )
    parser.add_argument(
        "--quality-min-presence-ratio",
        type=float,
        default=0.9,
        help="Minimum presence ratio allowed when available.",
    )
    return parser.parse_args()


def coarse_stimulus_family(stimulus_name):
    name = str(stimulus_name).lower()

    if "drifting" in name and "grating" in name:
        return "drifting_gratings"
    if "natural" in name and "scene" in name:
        return "natural_scenes"
    if "spontaneous" in name:
        return "spontaneous"

    return "other"


def get_region_col(units):
    for col in ["ecephys_structure_acronym", "structure_acronym"]:
        if col in units.columns:
            return col
    raise ValueError(
        "Could not find a structure acronym column in the session unit table."
    )


def apply_quality_filters(units, args):
    out = units.copy()

    if "isi_violations" in out.columns:
        out = out[out["isi_violations"] <= args.quality_min_isi_violations]

    if "amplitude_cutoff" in out.columns:
        out = out[out["amplitude_cutoff"] <= args.quality_max_amplitude_cutoff]

    if "presence_ratio" in out.columns:
        out = out[out["presence_ratio"] >= args.quality_min_presence_ratio]

    return out


def select_units(units, region_col, regions, max_units_per_region):
    selected = []

    for region in regions:
        region_units = units[units[region_col] == region].copy()

        if region_units.empty:
            continue

        sort_cols = []
        ascending = []

        if "snr" in region_units.columns:
            sort_cols.append("snr")
            ascending.append(False)

        if "firing_rate" in region_units.columns:
            sort_cols.append("firing_rate")
            ascending.append(False)

        if sort_cols:
            region_units = region_units.sort_values(sort_cols, ascending=ascending)

        selected.append(region_units.head(max_units_per_region))

    if not selected:
        raise ValueError("No units remained after region and quality filtering.")

    return pd.concat(selected, axis=0)


def build_presentation_table(stimulus_presentations, allowed_families, max_per_family):
    stim = stimulus_presentations.copy().reset_index()

    if "stimulus_presentation_id" not in stim.columns:
        if "id" in stim.columns:
            stim = stim.rename(columns={"id": "stimulus_presentation_id"})
        else:
            stim["stimulus_presentation_id"] = stim.index.astype(int)

    if "stimulus_name" not in stim.columns:
        raise ValueError("stimulus_presentations has no stimulus_name column.")

    stim["stimulus_family"] = stim["stimulus_name"].map(coarse_stimulus_family)
    stim = stim[stim["stimulus_family"].isin(allowed_families)].copy()

    if stim.empty:
        raise ValueError(
            "No stimulus presentations matched the requested families. "
            "Inspect session.stimulus_presentations['stimulus_name'].unique()."
        )

    kept = []
    for family, sub in stim.groupby("stimulus_family", sort=False):
        kept.append(sub.head(max_per_family))

    stim = pd.concat(kept, axis=0).sort_values("start_time").reset_index(drop=True)

    required = ["stimulus_presentation_id", "stimulus_name", "stimulus_family", "start_time", "stop_time"]
    missing = [x for x in required if x not in stim.columns]
    if missing:
        raise ValueError(f"stimulus_presentations missing required columns: {missing}")

    return stim[required].copy()


def bin_spikes_for_unit(spike_times, start, stop, bin_size):
    edges = np.arange(start, stop + bin_size, bin_size)
    if len(edges) < 2:
        return None

    counts, _ = np.histogram(spike_times, bins=edges)
    bin_starts = edges[:-1]
    bin_ends = edges[1:]
    return bin_starts, bin_ends, counts


def main():
    args = parse_args()

    from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
        EcephysProjectCache,
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.json"
    cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))

    print(f"[info] loading Allen session {args.session_id}")
    session = cache.get_session_data(args.session_id)

    units = session.units.copy().reset_index()
    region_col = get_region_col(units)

    units_qc = apply_quality_filters(units, args)
    selected_units = select_units(
        units=units_qc,
        region_col=region_col,
        regions=args.regions,
        max_units_per_region=args.max_units_per_region,
    )

    stimulus_presentations = build_presentation_table(
        stimulus_presentations=session.stimulus_presentations,
        allowed_families=args.stimulus_families,
        max_per_family=args.max_presentations_per_family,
    )

    selected_unit_ids = selected_units["unit_id"].astype(int).tolist()
    selected_units = selected_units[
        ["unit_id", region_col]
        + [c for c in ["firing_rate", "snr", "isi_violations", "amplitude_cutoff", "presence_ratio"] if c in selected_units.columns]
    ].copy()
    selected_units = selected_units.rename(columns={region_col: "region"})

    records = []

    for _, pres in stimulus_presentations.iterrows():
        pres_id = int(pres["stimulus_presentation_id"])
        stim_name = str(pres["stimulus_name"])
        family = str(pres["stimulus_family"])
        start = float(pres["start_time"])
        stop = float(pres["stop_time"])

        for _, unit in selected_units.iterrows():
            unit_id = int(unit["unit_id"])
            region = str(unit["region"])

            spike_times = session.spike_times.get(unit_id, np.array([]))
            binned = bin_spikes_for_unit(spike_times, start, stop, args.bin_size)

            if binned is None:
                continue

            bin_starts, bin_ends, counts = binned

            for local_bin, (b0, b1, count) in enumerate(zip(bin_starts, bin_ends, counts)):
                records.append(
                    {
                        "session_id": str(args.session_id),
                        "ecephys_session_id": int(args.session_id),
                        "stimulus_presentation_id": pres_id,
                        "trial_id": pres_id,
                        "stimulus_name": stim_name,
                        "stimulus_family": family,
                        "region": region,
                        "unit_id": unit_id,
                        "bin_index": int(local_bin),
                        "bin_start_time": float(b0),
                        "bin_end_time": float(b1),
                        "bin_size_sec": float(args.bin_size),
                        "spike_count": int(count),
                        "firing_rate_hz": float(count) / float(args.bin_size),
                    }
                )

    if not records:
        raise ValueError("No binned spike records were produced.")

    binned = pd.DataFrame(records)

    binned_path = out_dir / "real_binned_spike_table.csv"
    units_path = out_dir / "selected_units.csv"
    stim_path = out_dir / "selected_stimulus_presentations.csv"
    summary_path = out_dir / "extraction_summary.csv"

    binned.to_csv(binned_path, index=False)
    selected_units.to_csv(units_path, index=False)
    stimulus_presentations.to_csv(stim_path, index=False)

    summary = (
        binned.groupby(["session_id", "region", "stimulus_family"])
        .agg(
            n_units=("unit_id", "nunique"),
            n_presentations=("stimulus_presentation_id", "nunique"),
            n_bins=("spike_count", "size"),
            mean_spike_count=("spike_count", "mean"),
            mean_firing_rate_hz=("firing_rate_hz", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print(f"[ok] wrote {binned_path} rows={len(binned)}")
    print(f"[ok] wrote {units_path} rows={len(selected_units)}")
    print(f"[ok] wrote {stim_path} rows={len(stimulus_presentations)}")
    print(f"[ok] wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
