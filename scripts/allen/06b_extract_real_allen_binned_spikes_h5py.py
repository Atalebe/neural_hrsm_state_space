#!/usr/bin/env python3
"""
Low-memory HDF5 extractor for Allen Visual Coding Neuropixels NWB files.

This avoids AllenSDK/PyNWB full session loading. It directly reads only:
- /units/id
- /units/peak_channel_id
- /units/spike_times and /units/spike_times_index
- unit QC fields
- /general/extracellular_ephys/electrodes/id
- /general/extracellular_ephys/electrodes/location
- selected /intervals/*_presentations tables

The output is a compact binned spike table compatible with the Neural HRSM
population-state pipeline.

This script is designed for memory-constrained machines.
"""

from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd


INTERVAL_GROUPS = {
    "drifting_gratings": "drifting_gratings_presentations",
    "natural_scenes": "natural_scenes_presentations",
    "spontaneous": "spontaneous_presentations",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb", required=True, help="Path to cached Allen NWB file.")
    parser.add_argument("--session-id", type=int, default=715093703)
    parser.add_argument(
        "--out-dir",
        default="data/interim/allen_neuropixels_real/session_715093703_h5py",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["VISp", "VISl", "LGd", "CA1"],
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["drifting_gratings", "natural_scenes", "spontaneous"],
    )
    parser.add_argument("--bin-size", type=float, default=0.5)
    parser.add_argument("--max-units-per-region", type=int, default=5)
    parser.add_argument("--max-presentations-per-family", type=int, default=3)
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=5.0,
        help="Clip each presentation to this many seconds to avoid huge spontaneous blocks.",
    )
    parser.add_argument("--max-isi-violations", type=float, default=0.5)
    parser.add_argument("--max-amplitude-cutoff", type=float, default=0.1)
    parser.add_argument("--min-presence-ratio", type=float, default=0.9)
    parser.add_argument(
        "--quality-label",
        default="good",
        help="Require this /units/quality label. Use 'any' to disable.",
    )
    return parser.parse_args()


def decode_one(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def read_array(group, name):
    ds = group[name]
    arr = ds[()]
    if arr.dtype.kind in {"S", "O", "U"}:
        return np.array([decode_one(x) for x in arr])
    return arr


def read_scalar_or_array(group, name, n_expected=None, default=np.nan):
    if name in group:
        return read_array(group, name)
    if n_expected is None:
        return default
    return np.full(n_expected, default)


def read_intervals(f, family, group_name, max_presentations, max_duration_sec):
    path = f"intervals/{group_name}"
    if path not in f:
        return pd.DataFrame()

    g = f[path]
    n = len(g["start_time"])

    ids = read_array(g, "id") if "id" in g else np.arange(n)
    start = read_array(g, "start_time")
    stop = read_array(g, "stop_time")

    if "stimulus_name" in g:
        stim_name = read_array(g, "stimulus_name")
    else:
        stim_name = np.array([family] * n)

    out = pd.DataFrame(
        {
            "stimulus_presentation_id": ids.astype(int),
            "stimulus_name": stim_name.astype(str),
            "stimulus_family": family,
            "start_time": start.astype(float),
            "stop_time_original": stop.astype(float),
        }
    )

    out = out.sort_values("start_time").head(max_presentations).copy()
    out["stop_time"] = np.minimum(
        out["stop_time_original"].to_numpy(dtype=float),
        out["start_time"].to_numpy(dtype=float) + float(max_duration_sec),
    )

    out = out[out["stop_time"] > out["start_time"]].copy()
    return out


def build_unit_table(f, target_regions, args):
    ug = f["units"]
    n_units = len(ug["id"])

    unit_ids = read_array(ug, "id").astype(int)
    peak_channel_ids = read_array(ug, "peak_channel_id").astype(int)

    eg = f["general/extracellular_ephys/electrodes"]
    electrode_ids = read_array(eg, "id").astype(int)
    electrode_locations = read_array(eg, "location").astype(str)

    channel_to_region = {
        int(ch): str(loc)
        for ch, loc in zip(electrode_ids, electrode_locations)
    }

    regions = np.array(
        [channel_to_region.get(int(ch), "unknown") for ch in peak_channel_ids]
    )

    quality = (
        read_array(ug, "quality").astype(str)
        if "quality" in ug
        else np.array(["unknown"] * n_units)
    )

    firing_rate = read_scalar_or_array(ug, "firing_rate", n_units).astype(float)
    snr = read_scalar_or_array(ug, "snr", n_units).astype(float)
    isi = read_scalar_or_array(ug, "isi_violations", n_units).astype(float)
    amp_cutoff = read_scalar_or_array(ug, "amplitude_cutoff", n_units).astype(float)
    presence = read_scalar_or_array(ug, "presence_ratio", n_units).astype(float)

    table = pd.DataFrame(
        {
            "unit_table_index": np.arange(n_units, dtype=int),
            "unit_id": unit_ids,
            "peak_channel_id": peak_channel_ids,
            "region": regions,
            "quality": quality,
            "firing_rate": firing_rate,
            "snr": snr,
            "isi_violations": isi,
            "amplitude_cutoff": amp_cutoff,
            "presence_ratio": presence,
        }
    )

    mask = table["region"].isin(target_regions)

    if args.quality_label.lower() != "any":
        mask &= table["quality"].astype(str).str.lower().eq(args.quality_label.lower())

    mask &= table["isi_violations"].le(args.max_isi_violations)
    mask &= table["amplitude_cutoff"].le(args.max_amplitude_cutoff)
    mask &= table["presence_ratio"].ge(args.min_presence_ratio)

    table = table[mask].copy()

    selected = []
    for region in target_regions:
        sub = table[table["region"] == region].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            ["snr", "firing_rate"],
            ascending=[False, False],
        ).head(args.max_units_per_region)
        selected.append(sub)

    if not selected:
        raise RuntimeError(
            "No units survived region and QC filtering. "
            "Try --quality-label any or loosen QC thresholds."
        )

    selected = pd.concat(selected, axis=0).reset_index(drop=True)
    return selected


def unit_spike_slice(spike_times_ds, spike_index_ds, unit_table_index):
    i = int(unit_table_index)
    start = 0 if i == 0 else int(spike_index_ds[i - 1])
    stop = int(spike_index_ds[i])
    return spike_times_ds[start:stop]


def bin_unit_spikes(spikes, start, stop, bin_size):
    if stop <= start:
        return []

    edges = np.arange(start, stop + bin_size, bin_size)
    if len(edges) < 2:
        return []

    left = np.searchsorted(spikes, start, side="left")
    right = np.searchsorted(spikes, stop, side="right")
    local_spikes = spikes[left:right]

    counts, edges = np.histogram(local_spikes, bins=edges)
    rows = []
    for j, count in enumerate(counts):
        rows.append((j, float(edges[j]), float(edges[j + 1]), int(count)))
    return rows


def main():
    args = parse_args()

    nwb_path = Path(args.nwb)
    if not nwb_path.exists():
        raise SystemExit(f"Missing NWB file: {nwb_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] opening {nwb_path}")
    print(f"[info] size_gb={nwb_path.stat().st_size / 1e9:.3f}")
    print("[info] using direct h5py extraction, no PyNWB session construction")

    records = []

    with h5py.File(nwb_path, "r") as f:
        selected_units = build_unit_table(f, args.regions, args)

        presentations = []
        for family in args.families:
            group_name = INTERVAL_GROUPS.get(family)
            if group_name is None:
                print(f"[warn] unknown family ignored: {family}")
                continue
            pres = read_intervals(
                f=f,
                family=family,
                group_name=group_name,
                max_presentations=args.max_presentations_per_family,
                max_duration_sec=args.max_duration_sec,
            )
            if not pres.empty:
                presentations.append(pres)

        if not presentations:
            raise RuntimeError("No selected stimulus presentations were found.")

        presentations = (
            pd.concat(presentations, axis=0)
            .sort_values("start_time")
            .reset_index(drop=True)
        )

        spike_times_ds = f["units/spike_times"]
        spike_index_ds = f["units/spike_times_index"]

        for _, unit in selected_units.iterrows():
            unit_idx = int(unit["unit_table_index"])
            unit_id = int(unit["unit_id"])
            region = str(unit["region"])

            spikes = unit_spike_slice(spike_times_ds, spike_index_ds, unit_idx)
            spikes = np.asarray(spikes, dtype=float)

            for _, pres in presentations.iterrows():
                start = float(pres["start_time"])
                stop = float(pres["stop_time"])

                binned = bin_unit_spikes(
                    spikes=spikes,
                    start=start,
                    stop=stop,
                    bin_size=float(args.bin_size),
                )

                for local_bin, b0, b1, count in binned:
                    records.append(
                        {
                            "session_id": str(args.session_id),
                            "ecephys_session_id": int(args.session_id),
                            "stimulus_presentation_id": int(
                                pres["stimulus_presentation_id"]
                            ),
                            "trial_id": int(pres["stimulus_presentation_id"]),
                            "stimulus_name": str(pres["stimulus_name"]),
                            "stimulus_family": str(pres["stimulus_family"]),
                            "region": region,
                            "unit_id": unit_id,
                            "bin_index": int(local_bin),
                            "bin_start_time": b0,
                            "bin_end_time": b1,
                            "bin_size_sec": float(args.bin_size),
                            "spike_count": int(count),
                            "firing_rate_hz": float(count) / float(args.bin_size),
                        }
                    )

    if not records:
        raise RuntimeError("No binned spike records produced.")

    binned = pd.DataFrame(records)

    binned_path = out_dir / "real_binned_spike_table.csv"
    units_path = out_dir / "selected_units.csv"
    presentations_path = out_dir / "selected_stimulus_presentations.csv"
    summary_path = out_dir / "extraction_summary.csv"

    binned.to_csv(binned_path, index=False)
    selected_units.to_csv(units_path, index=False)
    presentations.to_csv(presentations_path, index=False)

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
        .sort_values(["region", "stimulus_family"])
    )

    summary.to_csv(summary_path, index=False)

    print(f"[ok] wrote {binned_path} rows={len(binned)}")
    print(f"[ok] wrote {units_path} rows={len(selected_units)}")
    print(f"[ok] wrote {presentations_path} rows={len(presentations)}")
    print(f"[ok] wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
