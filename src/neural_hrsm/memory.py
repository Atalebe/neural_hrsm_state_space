from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


def fit_lagged_memory_gain(hrsm: pd.DataFrame, lags=(1, 2, 4, 8), alphas=(0.1, 1.0, 10.0, 100.0)) -> pd.DataFrame:
    """Estimate whether lagged state improves prediction of current neural state.

    The target is current mean activity. Baseline predictors are present stimulus
    and region. Memory predictors add lagged activity terms. The output is a
    conservative per-session estimate of prediction gain.
    """
    records = []
    for session_id, sub in hrsm.sort_values(["region", "trial_id", "time_bin"]).groupby("session_id"):
        work = sub.copy()
        for lag in lags:
            work[f"lag_rate_{lag}"] = work.groupby("region")["mean_rate_proxy"].shift(lag)
        work = work.dropna(subset=[f"lag_rate_{lag}" for lag in lags]).copy()
        if len(work) < 50 or work["trial_id"].nunique() < 3:
            continue
        y = work["mean_rate_proxy"].to_numpy()
        base = pd.get_dummies(work[["region", "stimulus_family"]], drop_first=True).astype(float)
        mem = pd.concat([base, work[[f"lag_rate_{lag}" for lag in lags]].astype(float)], axis=1)
        groups = work["trial_id"].to_numpy()
        n_splits = min(5, len(np.unique(groups)))
        cv = GroupKFold(n_splits=n_splits)
        base_scores, mem_scores = [], []
        for train, test in cv.split(base, y, groups):
            base_model = RidgeCV(alphas=alphas).fit(base.iloc[train], y[train])
            mem_model = RidgeCV(alphas=alphas).fit(mem.iloc[train], y[train])
            base_scores.append(r2_score(y[test], base_model.predict(base.iloc[test])))
            mem_scores.append(r2_score(y[test], mem_model.predict(mem.iloc[test])))
        records.append({
            "session_id": session_id,
            "baseline_r2": float(np.mean(base_scores)),
            "memory_r2": float(np.mean(mem_scores)),
            "memory_gain_r2": float(np.mean(mem_scores) - np.mean(base_scores)),
            "n_rows": int(len(work)),
            "n_trials": int(work["trial_id"].nunique()),
        })
    return pd.DataFrame(records)
