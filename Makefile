PYTHONPATH := src
CONFIG := configs/allen_neuropixels_v1.yaml

.PHONY: smoke clean

smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/allen/00_prepare_allen_neuropixels_manifest.py --config $(CONFIG)
	PYTHONPATH=$(PYTHONPATH) python scripts/allen/01_build_population_state_matrix.py --config $(CONFIG)
	PYTHONPATH=$(PYTHONPATH) python scripts/allen/02_compute_hrsm_proxies.py --config $(CONFIG)
	PYTHONPATH=$(PYTHONPATH) python scripts/allen/03_fit_memory_kernel.py --config $(CONFIG)
	PYTHONPATH=$(PYTHONPATH) python scripts/allen/04_make_diagnostic_figures.py --config $(CONFIG)

clean:
	rm -rf data/interim/allen_neuropixels_v1 data/processed/allen_neuropixels_v1 results/tables/*.csv results/figures/*.png
