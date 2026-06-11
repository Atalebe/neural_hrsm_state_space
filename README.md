# Neural HRSM State Space

Starter repository for operationalising a Homeostatic Reserve, Recoverability, Stability, and Memory state space for neural population dynamics.

This repository is intentionally conservative. It does not claim to measure consciousness directly. It measures neural population homeostasis, perturbation recovery, state stability, and non-Markovian historical dependence in neural systems that may support conscious processing.

## First empirical branch

The initial target is Allen Visual Coding Neuropixels-style data because it provides neuron-level spiking activity, repeated stimulus structure, region labels, and behavioural covariates. The first test asks whether neural population activity occupies structured HRSM domains and whether a historical memory term improves prediction beyond present stimulus and present state.

## Repository layout

- `configs/`: YAML configuration files controlling datasets, proxy definitions, and run settings.
- `src/neural_hrsm/`: reusable Python package for loading data, computing proxies, fitting memory kernels, and plotting.
- `scripts/allen/`: Allen Neuropixels branch scripts.
- `scripts/dandi/`: DANDI replication branch stubs.
- `scripts/hcp/`: human connectome translation branch stubs.
- `scripts/openneuro/`: anesthesia and sleep transition branch stubs.
- `data/`: raw, interim, and processed data. Raw data are not committed.
- `results/`: generated tables and figures.
- `logs/`: run logs and milestone reports.
- `docs/`: logbook copy and notes.

## Quick start

```bash
conda create -n neural_hrsm python=3.11 -y
conda activate neural_hrsm
pip install -r requirements.txt
python scripts/allen/00_prepare_allen_neuropixels_manifest.py --config configs/allen_neuropixels_v1.yaml
python scripts/allen/01_build_population_state_matrix.py --config configs/allen_neuropixels_v1.yaml
python scripts/allen/02_compute_hrsm_proxies.py --config configs/allen_neuropixels_v1.yaml
python scripts/allen/03_fit_memory_kernel.py --config configs/allen_neuropixels_v1.yaml
python scripts/allen/04_make_diagnostic_figures.py --config configs/allen_neuropixels_v1.yaml
```

The starter scripts are designed to run in two modes:

1. `synthetic_smoke_test: true`, which creates a toy neural population dataset and verifies the full pipeline.
2. `synthetic_smoke_test: false`, which expects a real Allen-style spike table or NWB-derived event table.

## Scientific contract

The pipeline tests neural homeostatic state dynamics, not metaphysical consciousness. Consciousness-adjacent interpretations are deferred until neural HRSM axes are stable, orthogonalised, and shown to survive negative controls.
