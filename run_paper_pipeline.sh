#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/6] Cross-session synthesis"
python scripts/allen/16_cross_session_spontaneous_synthesis.py \
  --session-dirs \
    results/real_allen/session_715093703_spontaneous_v1 \
    results/real_allen/session_719161530_spontaneous_v1 \
    results/real_allen/session_750749662_spontaneous_v1 \
  --out-dir results/cross_session/allen_spontaneous_v1 \
  --fig-dir results/figures/cross_session/allen_spontaneous_v1

echo "[2/6] Variance-scaling audit"
python scripts/allen/17_cross_session_variance_scaling_audit.py \
  --sessions 715093703 719161530 750749662 \
  --out-dir results/cross_session/allen_spontaneous_v1 \
  --fig-dir results/figures/cross_session/allen_spontaneous_v1

echo "[3/6] Lag-ablation grid"
python scripts/allen/18_spontaneous_memory_lag_ablation_grid.py \
  --sessions 715093703 719161530 750749662 \
  --n-shuffles 50 \
  --skip-existing

echo "[4/6] Descriptive ripeness index"
python scripts/allen/19_compute_spontaneous_ripeness_index.py \
  --synthesis-dir results/cross_session/allen_spontaneous_v1 \
  --fig-dir results/figures/cross_session/allen_spontaneous_v1

echo "[5/6] Verify required manuscript figures"
required=(
  "results/figures/cross_session/allen_spontaneous_v1/cross_session_spontaneous_memory_control.png"
  "results/figures/cross_session/allen_spontaneous_v1/cross_session_region_memory_control.png"
  "results/figures/cross_session/allen_spontaneous_v1/cross_session_target_memory_heatmap.png"
  "results/figures/cross_session/allen_spontaneous_v1/cross_session_variance_residual_target_rank.png"
  "results/figures/ablation/allen_spontaneous_v1/lag_ablation_controlled_memory_curves.png"
  "results/figures/cross_session/allen_spontaneous_v1/spontaneous_neural_ripeness_phi_vs_memory.png"
)

for f in "${required[@]}"; do
  test -s "$f" || { echo "[error] missing required figure: $f" >&2; exit 1; }
  echo "[ok] $f"
done

echo "[6/6] Compile manuscript"
rm -f neural_hrsm_spontaneous_memory.aux \
      neural_hrsm_spontaneous_memory.bbl \
      neural_hrsm_spontaneous_memory.blg \
      neural_hrsm_spontaneous_memory.log \
      neural_hrsm_spontaneous_memory.out \
      neural_hrsm_spontaneous_memory.fls \
      neural_hrsm_spontaneous_memory.fdb_latexmk

latexmk -pdf -interaction=nonstopmode -halt-on-error paper/neural_hrsm_spontaneous_memory.tex

echo "[ok] manuscript built: neural_hrsm_spontaneous_memory.pdf"
