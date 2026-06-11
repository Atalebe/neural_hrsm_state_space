from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from neural_hrsm.io import load_config, ensure_dirs, write_csv
from neural_hrsm.population import build_population_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    interim = Path(config["paths"]["interim_dir"])
    processed = Path(config["paths"]["processed_dir"])
    input_path = interim / "synthetic_binned_spike_table.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected input table not found: {input_path}")
    spike_table = pd.read_csv(input_path)
    state = build_population_state(spike_table)
    write_csv(state, processed / "population_state_matrix.csv")
    print(f"[ok] wrote {processed / 'population_state_matrix.csv'} rows={len(state)}")


if __name__ == "__main__":
    main()
