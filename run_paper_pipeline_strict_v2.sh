#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

STRICT_SESSIONS=(715093703 719161530 750749662 751348571 755434585 756029989)

echo "[1/7] Strict six-session synthesis"
python scripts/allen/16_cross_session_spontaneous_synthesis.py \
  --session-dirs \
    results/real_allen/session_715093703_spontaneous_v1 \
    results/real_allen/session_719161530_spontaneous_v1 \
    results/real_allen/session_750749662_spontaneous_v1 \
    results/real_allen/session_751348571_spontaneous_v1 \
    results/real_allen/session_755434585_spontaneous_v1 \
    results/real_allen/session_756029989_spontaneous_v1 \
  --out-dir results/cross_session/allen_spontaneous_strict_v2 \
  --fig-dir results/figures/cross_session/allen_spontaneous_strict_v2

echo "[2/7] Variance/autocorrelation residual audit"
python scripts/allen/17_cross_session_variance_scaling_audit.py \
  --sessions "${STRICT_SESSIONS[@]}" \
  --out-dir results/cross_session/allen_spontaneous_strict_v2 \
  --fig-dir results/figures/cross_session/allen_spontaneous_strict_v2

echo "[3/7] Ripeness index"
python scripts/allen/19_compute_spontaneous_ripeness_index.py \
  --synthesis-dir results/cross_session/allen_spontaneous_strict_v2 \
  --fig-dir results/figures/cross_session/allen_spontaneous_strict_v2

echo "[4/7] Raw and orthogonalized HRSM axis audit"
python scripts/allen/21_hrsm_raw_axis_audit.py \
  --session-dirs \
    results/real_allen/session_715093703_spontaneous_v1 \
    results/real_allen/session_719161530_spontaneous_v1 \
    results/real_allen/session_750749662_spontaneous_v1 \
    results/real_allen/session_751348571_spontaneous_v1 \
    results/real_allen/session_755434585_spontaneous_v1 \
    results/real_allen/session_756029989_spontaneous_v1 \
  --out-dir results/cross_session/allen_spontaneous_strict_v2 \
  --fig-dir results/figures/cross_session/allen_spontaneous_strict_v2

echo "[5/8] Verify strict-v2 lag-ablation outputs"
test -s results/ablation/allen_spontaneous_strict_v2/lag_ablation_cross_session_target_summary.csv
test -s results/ablation/allen_spontaneous_strict_v2/lag_ablation_flags.csv
test -s results/figures/ablation/allen_spontaneous_strict_v2/lag_ablation_controlled_memory_curves.png

echo "[6/8] Generate combined manuscript figures"
python scripts/paper/03_make_strict_v2_combined_figures.py

echo "[7/8] Generate strict-v2 manuscript tables and source"
python scripts/paper/01_make_strict_v2_tables.py
python scripts/paper/02_write_strict_v2_manuscript.py

echo "[8/9] Compile strict-v2 manuscript"
rm -f neural_hrsm_spontaneous_memory_strict_v2.aux \
      neural_hrsm_spontaneous_memory_strict_v2.bbl \
      neural_hrsm_spontaneous_memory_strict_v2.blg \
      neural_hrsm_spontaneous_memory_strict_v2.log \
      neural_hrsm_spontaneous_memory_strict_v2.out \
      neural_hrsm_spontaneous_memory_strict_v2.fls \
      neural_hrsm_spontaneous_memory_strict_v2.fdb_latexmk \
      neural_hrsm_spontaneous_memory_strict_v2.pdf

latexmk -pdf -interaction=nonstopmode -halt-on-error paper/neural_hrsm_spontaneous_memory_strict_v2.tex

echo "[9/9] Build strict-v2 submission bundle"
python scripts/paper/04_make_strict_v2_submission_bundle.py

echo "[ok] built neural_hrsm_spontaneous_memory_strict_v2.pdf"
echo "[ok] built submission_bundle/neural_hrsm_spontaneous_memory_strict_v2_submission_bundle.zip"
