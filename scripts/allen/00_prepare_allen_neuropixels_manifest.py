from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from neural_hrsm.io import load_config, ensure_dirs, write_csv
from neural_hrsm.synthetic import make_synthetic_spike_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    seed = config.get("runtime", {}).get("random_seed", 42)
    interim = Path(config["paths"]["interim_dir"])
    if config.get("runtime", {}).get("synthetic_smoke_test", True):
        spike_table = make_synthetic_spike_table(seed=seed)
        write_csv(spike_table, interim / "synthetic_binned_spike_table.csv")
        manifest = pd.DataFrame([{
            "branch": config["project"]["branch"],
            "mode": "synthetic_smoke_test",
            "n_rows": len(spike_table),
            "n_sessions": spike_table["session_id"].nunique(),
            "n_regions": spike_table["region"].nunique(),
            "n_units": spike_table["unit_id"].nunique(),
            "output": str(interim / "synthetic_binned_spike_table.csv"),
        }])
    else:
        manifest = pd.DataFrame([{
            "branch": config["project"]["branch"],
            "mode": "real_data_placeholder",
            "note": "Attach AllenSDK session download or pre-exported spike table here.",
        }])
    write_csv(manifest, interim / config["allen"].get("manifest_name", "manifest.csv"))
    print(f"[ok] wrote manifest and input table to {interim}")


if __name__ == "__main__":
    main()
