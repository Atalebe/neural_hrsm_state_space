from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from neural_hrsm.io import load_config, ensure_dirs, write_csv
from neural_hrsm.proxies import compute_hrsm_proxies, aggregate_hrsm_domains
from neural_hrsm.orthogonalize import residualize_axes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    processed = Path(config["paths"]["processed_dir"])
    results = Path(config["paths"]["results_dir"])
    state = pd.read_csv(processed / "population_state_matrix.csv")
    hrsm = compute_hrsm_proxies(state)
    if config.get("orthogonalization", {}).get("enabled", True):
        order = tuple(config.get("orthogonalization", {}).get("axis_order", ["H", "R", "S", "M"]))
        hrsm = residualize_axes(hrsm, axes=order)
        max_corr = hrsm.attrs.get("max_abs_offdiag_corr", None)
    else:
        max_corr = None
    domains = aggregate_hrsm_domains(hrsm)
    write_csv(hrsm, processed / "neural_hrsm_bin_level_metrics.csv")
    write_csv(domains, results / "tables" / "neural_hrsm_domain_summary.csv")
    if max_corr is not None:
        audit = pd.DataFrame([{"max_abs_offdiag_corr_orthogonalized": max_corr}])
        write_csv(audit, results / "tables" / "orthogonality_audit.csv")
    print("[ok] wrote HRSM metrics and domain summary")


if __name__ == "__main__":
    main()
