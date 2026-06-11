from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_spike_table(
    seed: int = 42,
    n_sessions: int = 2,
    n_regions: int = 4,
    units_per_region: int = 40,
    n_trials: int = 80,
    n_bins: int = 30,
) -> pd.DataFrame:
    """Create a toy Allen-style binned spike table.

    Rows are unit x trial x time-bin observations. The synthetic generator includes
    region effects, stimulus effects, slow drift, and lag dependence so that the
    smoke test has a non-zero memory signal.
    """
    rng = np.random.default_rng(seed)
    regions = ["VISp", "VISl", "CA1", "LGd"][:n_regions]
    stimuli = ["natural_scenes", "drifting_gratings", "spontaneous"]
    rows = []
    for s in range(n_sessions):
        for region_i, region in enumerate(regions):
            for unit in range(units_per_region):
                unit_gain = rng.lognormal(mean=0.0, sigma=0.25)
                prev = 0.0
                for trial in range(n_trials):
                    stim = stimuli[trial % len(stimuli)]
                    stim_drive = {"natural_scenes": 1.2, "drifting_gratings": 0.9, "spontaneous": 0.35}[stim]
                    for b in range(n_bins):
                        transient = np.exp(-((b - 8) ** 2) / 25) if stim != "spontaneous" else 0.1
                        region_drive = 0.15 * region_i
                        memory = 0.35 * prev
                        rate = unit_gain * (0.2 + stim_drive * transient + region_drive + memory)
                        count = rng.poisson(max(rate, 0.01))
                        prev = 0.85 * prev + 0.15 * count
                        rows.append({
                            "session_id": f"session_{s+1}",
                            "region": region,
                            "unit_id": f"s{s+1}_{region}_u{unit:03d}",
                            "trial_id": trial,
                            "stimulus_family": stim,
                            "time_bin": b,
                            "spike_count": count,
                        })
    return pd.DataFrame(rows)
