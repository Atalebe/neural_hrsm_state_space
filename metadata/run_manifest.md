# Run Manifest

## Branch 001: Allen Neuropixels synthetic smoke test

Purpose: Verify repository structure, proxy computation, orthogonalization audit, memory-gain estimation, and diagnostic plotting before attaching real Allen Neuropixels data.

Commands:

```bash
make smoke
```

Expected outputs:

- `data/interim/allen_neuropixels_v1/synthetic_binned_spike_table.csv`
- `data/processed/allen_neuropixels_v1/population_state_matrix.csv`
- `data/processed/allen_neuropixels_v1/neural_hrsm_bin_level_metrics.csv`
- `results/tables/neural_hrsm_domain_summary.csv`
- `results/tables/orthogonality_audit.csv`
- `results/tables/memory_kernel_gain_summary.csv`
- `results/figures/neural_hrsm_region_summary.png`
- `results/figures/memory_kernel_gain_by_session.png`
