from __future__ import annotations

import pandas as pd


def build_population_state(spike_table: pd.DataFrame) -> pd.DataFrame:
    """Convert long spike table into trial-time population summaries.

    The starter state matrix intentionally keeps interpretable summary variables.
    More detailed PCA, factor-model, and manifold embeddings can be added later.
    """
    required = {"session_id", "region", "trial_id", "stimulus_family", "time_bin", "spike_count"}
    missing = required - set(spike_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    grouped = spike_table.groupby(
        ["session_id", "region", "trial_id", "stimulus_family", "time_bin"],
        as_index=False,
    )["spike_count"].agg(["mean", "std", "sum", "count"]).reset_index()
    grouped = grouped.rename(columns={
        "mean": "mean_rate_proxy",
        "std": "dispersion_proxy",
        "sum": "population_spike_count",
        "count": "n_units",
    })
    grouped["dispersion_proxy"] = grouped["dispersion_proxy"].fillna(0.0)
    return grouped
