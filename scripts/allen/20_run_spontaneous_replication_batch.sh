#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

if [ "$#" -eq 0 ]; then
  SESSIONS=(751348571 754312389 756029989)
else
  SESSIONS=("$@")
fi

for SESSION_ID in "${SESSIONS[@]}"; do
  RUN_ID="session_${SESSION_ID}_spontaneous_v1"
  NWB="data/raw/allen_neuropixels_cache/session_${SESSION_ID}/session_${SESSION_ID}.nwb"

  echo "=============================="
  echo "[session] ${SESSION_ID}"
  echo "=============================="

  if [ ! -s "$NWB" ]; then
    echo "[download] $SESSION_ID"
    bash -lc "nice -n 15 conda run -n neural_hrsm_allen python scripts/allen/15_download_allen_session_nwb_only.py \
      --session-id ${SESSION_ID} \
      --cache-dir data/raw/allen_neuropixels_cache"
  else
    echo "[ok] NWB exists: $NWB"
  fi

  echo "[probe] $SESSION_ID"
  conda run -n neural_hrsm_allen python scripts/allen/06a_probe_cached_nwb_h5py.py \
    --nwb "$NWB" > "logs/probe_${SESSION_ID}_spontaneous_v1.txt" 2>&1 || {
      echo "[error] probe failed for $SESSION_ID"
      tail -80 "logs/probe_${SESSION_ID}_spontaneous_v1.txt"
      exit 1
    }

  if [ ! -s "data/interim/allen_neuropixels_real/${RUN_ID}/real_binned_spike_table.csv" ]; then
    echo "[extract] $SESSION_ID"
    bash -lc "ulimit -v 3500000; nice -n 15 conda run -n neural_hrsm_allen env PYTHONPATH=src python scripts/allen/06b_extract_real_allen_binned_spikes_h5py.py \
      --nwb ${NWB} \
      --session-id ${SESSION_ID} \
      --out-dir data/interim/allen_neuropixels_real/${RUN_ID} \
      --regions VISp VISl LGd CA1 \
      --families spontaneous \
      --bin-size 0.25 \
      --max-units-per-region 20 \
      --max-presentations-per-family 15 \
      --max-duration-sec 60"
  else
    echo "[skip] extraction exists"
  fi

  echo "[population matrix] $SESSION_ID"
  python scripts/allen/07_build_real_population_state_matrix_h5py.py \
    --binned "data/interim/allen_neuropixels_real/${RUN_ID}/real_binned_spike_table.csv" \
    --out-dir "data/processed/allen_neuropixels_real/${RUN_ID}"

  echo "[HRSM proxies] $SESSION_ID"
  python scripts/allen/08_compute_real_hrsm_proxies_h5py.py \
    --population "data/processed/allen_neuropixels_real/${RUN_ID}/population_state_matrix.csv" \
    --out-dir "results/real_allen/${RUN_ID}"

  echo "[mean-rate memory] $SESSION_ID"
  python scripts/allen/09_fit_real_memory_kernel_h5py.py \
    --population "data/processed/allen_neuropixels_real/${RUN_ID}/population_state_matrix.csv" \
    --out-dir "results/real_allen/${RUN_ID}" \
    --target population_mean_rate_hz \
    --lags 1 2 \
    --alpha 1.0 \
    --min-supervised-rows 200

  echo "[shuffled-lag control] $SESSION_ID"
  python scripts/allen/11_real_memory_negative_controls_h5py.py \
    --population "data/processed/allen_neuropixels_real/${RUN_ID}/population_state_matrix.csv" \
    --out-dir "results/real_allen/${RUN_ID}" \
    --target population_mean_rate_hz \
    --lags 1 2 \
    --alpha 1.0 \
    --min-supervised-rows 200 \
    --n-shuffles 100 \
    --seed "$((714159 + SESSION_ID % 1000))"

  echo "[target sweep] $SESSION_ID"
  python scripts/allen/13_real_memory_target_sweep_h5py.py \
    --population "data/processed/allen_neuropixels_real/${RUN_ID}/population_state_matrix.csv" \
    --out-dir "results/real_allen/${RUN_ID}" \
    --targets population_mean_rate_hz population_std_rate_hz active_unit_fraction population_l2_rate_norm population_rate_entropy population_state_speed \
    --lags 1 2 \
    --alpha 1.0 \
    --min-supervised-rows 200 \
    --n-shuffles 100 \
    --seed "$((714160 + SESSION_ID % 1000))"

  echo "[figures] $SESSION_ID"
  python scripts/allen/10_make_real_hrsm_visual_story_h5py.py \
    --results-dir "results/real_allen/${RUN_ID}" \
    --population-summary "data/processed/allen_neuropixels_real/${RUN_ID}/population_state_summary.csv" \
    --out-dir "results/figures/real_allen/${RUN_ID}"

  python scripts/allen/12_make_real_memory_control_figures_h5py.py \
    --results-dir "results/real_allen/${RUN_ID}" \
    --out-dir "results/figures/real_allen/${RUN_ID}"

  python scripts/allen/14_make_real_memory_target_sweep_figures_h5py.py \
    --results-dir "results/real_allen/${RUN_ID}" \
    --out-dir "results/figures/real_allen/${RUN_ID}"

  echo "[summary] $SESSION_ID"
  cat "data/interim/allen_neuropixels_real/${RUN_ID}/extraction_summary.csv"
  cat "results/real_allen/${RUN_ID}/real_memory_negative_control_pooled_summary.csv"
  cat "results/real_allen/${RUN_ID}/real_memory_target_sweep_target_summary.csv"
done

echo "[ok] batch complete"
