from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from neural_hrsm.io import load_config, ensure_dirs, write_csv
from neural_hrsm.memory import fit_lagged_memory_gain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    processed = Path(config["paths"]["processed_dir"])
    results = Path(config["paths"]["results_dir"])
    hrsm = pd.read_csv(processed / "neural_hrsm_bin_level_metrics.csv")
    lags = tuple(config.get("memory_kernel", {}).get("lags_bins", [1, 2, 4, 8]))
    alphas = tuple(config.get("memory_kernel", {}).get("alpha_grid", [0.1, 1.0, 10.0]))
    memory = fit_lagged_memory_gain(hrsm, lags=lags, alphas=alphas)
    write_csv(memory, results / "tables" / "memory_kernel_gain_summary.csv")
    print(f"[ok] wrote memory-kernel gain summary rows={len(memory)}")


if __name__ == "__main__":
    main()
