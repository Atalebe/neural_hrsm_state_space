from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import zscore


def _safe_z(x: pd.Series) -> pd.Series:
    values = x.astype(float).to_numpy()
    if np.nanstd(values) == 0:
        return pd.Series(np.zeros_like(values), index=x.index)
    return pd.Series(zscore(values, nan_policy="omit"), index=x.index).fillna(0.0)


def compute_hrsm_proxies(state: pd.DataFrame) -> pd.DataFrame:
    """Compute first-pass H, R, S, M proxies from population summaries.

    These are deliberately transparent starter proxies. They should be replaced
    or refined as soon as real Allen/NWB structure is attached.
    """
    df = state.copy()
    df = df.sort_values(["session_id", "region", "trial_id", "time_bin"])

    grp = df.groupby(["session_id", "region"], group_keys=False)
    df["rate_z"] = grp["mean_rate_proxy"].apply(_safe_z)
    df["dispersion_z"] = grp["dispersion_proxy"].apply(_safe_z)

    # H: response capacity, high when signal is strong relative to dispersion but not saturated.
    df["H_reserve_raw"] = df["rate_z"] - 0.5 * df["dispersion_z"].abs()

    # R: recoverability, negative absolute distance from regional baseline after stimulus bins.
    baseline = df[df["stimulus_family"].eq("spontaneous")].groupby(["session_id", "region"])["mean_rate_proxy"].median()
    df = df.join(baseline.rename("regional_baseline"), on=["session_id", "region"])
    df["regional_baseline"] = df["regional_baseline"].fillna(df.groupby(["session_id", "region"])["mean_rate_proxy"].transform("median"))
    df["R_recoverability_raw"] = -1.0 * (df["mean_rate_proxy"] - df["regional_baseline"]).abs()

    # S: stability, high when dispersion and abrupt changes are low.
    df["rate_delta"] = grp["mean_rate_proxy"].diff().fillna(0.0)
    df["S_stability_raw"] = -1.0 * (df["dispersion_proxy"].fillna(0.0) + df["rate_delta"].abs())

    # M: starter path-dependence, correlation-like lag signal at local row level.
    df["lag_rate"] = grp["mean_rate_proxy"].shift(1).fillna(df["mean_rate_proxy"].median())
    df["M_memory_raw"] = df["mean_rate_proxy"] * df["lag_rate"]

    for raw, out in [
        ("H_reserve_raw", "H"),
        ("R_recoverability_raw", "R"),
        ("S_stability_raw", "S"),
        ("M_memory_raw", "M"),
    ]:
        df[out] = df.groupby(["session_id"], group_keys=False)[raw].apply(_safe_z)

    df["Phi_neural"] = df[["H", "R", "S", "M"]].mean(axis=1)
    return df


def aggregate_hrsm_domains(hrsm: pd.DataFrame) -> pd.DataFrame:
    agg = hrsm.groupby(["session_id", "region", "stimulus_family"], as_index=False).agg(
        H=("H", "median"), R=("R", "median"), S=("S", "median"), M=("M", "median"),
        Phi_neural=("Phi_neural", "median"), n_bins=("time_bin", "count")
    )
    for axis in ["H", "R", "S", "M"]:
        agg[f"{axis}_domain"] = pd.cut(agg[axis], bins=[-np.inf, -0.5, 0.5, np.inf], labels=["low", "mid", "high"])
    agg["hrsm_domain"] = agg[["H_domain", "R_domain", "S_domain", "M_domain"]].astype(str).agg("_".join, axis=1)
    return agg
