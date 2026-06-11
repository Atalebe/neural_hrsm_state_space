from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def residualize_axes(df, axes):
    out = df.copy()
    used = []

    for axis in axes:
        y = out[axis].to_numpy(dtype=float)

        if used:
            X = out[used].to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(X)), X])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
        else:
            resid = y.copy()

        out[f"{axis}_ortho"] = resid
        used.append(axis)

    ortho_cols = [f"{axis}_ortho" for axis in axes]
    corr = out[ortho_cols].corr()

    corr_arr = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_arr, 0.0)
    max_abs_offdiag = float(np.nanmax(np.abs(corr_arr)))

    out.attrs["max_abs_offdiag_corr_orthogonalized"] = max_abs_offdiag
    return out
