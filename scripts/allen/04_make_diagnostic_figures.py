from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from neural_hrsm.io import load_config, ensure_dirs
from neural_hrsm.plots import plot_hrsm_region_summary, plot_memory_gain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    results = Path(config["paths"]["results_dir"])
    domains = pd.read_csv(results / "tables" / "neural_hrsm_domain_summary.csv")
    mem = pd.read_csv(results / "tables" / "memory_kernel_gain_summary.csv")
    plot_hrsm_region_summary(domains, results / "figures" / "neural_hrsm_region_summary.png")
    plot_memory_gain(mem, results / "figures" / "memory_kernel_gain_by_session.png")
    print(f"[ok] wrote diagnostic figures to {results / 'figures'}")


if __name__ == "__main__":
    main()
