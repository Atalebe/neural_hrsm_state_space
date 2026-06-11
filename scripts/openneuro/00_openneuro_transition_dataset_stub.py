from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from neural_hrsm.io import load_config, ensure_dirs, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    out = Path(config["paths"]["interim_dir"]) / "branch_stub_manifest.csv"
    table = pd.DataFrame([{
        "branch": config["project"]["branch"],
        "status": "stub",
        "purpose": config.get("notes", {}).get("purpose", "later branch"),
        "caution": config.get("notes", {}).get("caution", "define dataset-specific proxies before interpretation"),
    }])
    write_csv(table, out)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
