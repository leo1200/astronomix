#!/bin/bash
# Validation grid for the deep-void vacuum_rest fix.
# Runs each config (autocvd picks a free GPU), greps the result, appends a
# one-line PASS/FAIL summary to deepvoid_grid_summary.txt.
# Success bar (per HANDOFF_deepvoid_instability.md): first_bad_snap=-1 with
# M_turb preserved, across cfl {1.5,1.0,0.7,0.4} and nsnap {8,60,200}.
set -u
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
PY=/export/home/lstorcks/.local/share/mamba/envs/astx/bin/python
export PYTHONPATH=/export/home/lstorcks/agent-home/astronomix-refactor-port
SUMMARY=deepvoid_grid_summary.txt
: > "$SUMMARY"

run() {
  tag="$1"; shift
  log="grid_${tag}.log"
  $PY paper_turbulence.py --outdir data_repro --eos iso --N 128 \
    --beta 0.1 --F0 3.5 --tcross 4 --protect 1 --rhomin 0.005 \
    --vmaxcap 1000000000 --tag "$tag" "$@" > "$log" 2>&1
  line=$(grep -E "first_bad_snap" "$log" | head -1)
  mline=$(grep -E "M_turb\(t/tc\)" "$log" | head -1)
  if echo "$line" | grep -q "first_bad_snap=-1"; then verdict=PASS; else verdict="FAIL/ERR"; fi
  echo "[$verdict] $tag :: $line" | tee -a "$SUMMARY"
}

# --- redistribute + vacuum_rest (the production fix) across the CFL bar ---
run rg_redist_cfl15 --mhd 0 --mturb 40 --cfl 1.5 --nsnap 8  --stage_mode redist --vacuum_rest 1
run rg_redist_cfl07 --mhd 0 --mturb 40 --cfl 0.7 --nsnap 8  --stage_mode redist --vacuum_rest 1
run rg_redist_cfl04 --mhd 0 --mturb 40 --cfl 0.4 --nsnap 8  --stage_mode redist --vacuum_rest 1
# --- nsnap bar (dt-sequence sensitivity) at cfl 1.0 ---
run rg_redist_ns60  --mhd 0 --mturb 40 --cfl 1.0 --nsnap 60 --stage_mode redist --vacuum_rest 1
run rg_redist_ns200 --mhd 0 --mturb 40 --cfl 1.0 --nsnap 200 --stage_mode redist --vacuum_rest 1
# --- MHD M20 deep-void repro (low-beta adds stiffness) ---
run rg_mhd_redist   --mhd 1 --mturb 20 --cfl 1.0 --nsnap 8  --stage_mode redist --vacuum_rest 1
# --- floor + vacuum_rest fallback at the hardest CFLs (for comparison) ---
run rg_floor_cfl15  --mhd 0 --mturb 40 --cfl 1.5 --nsnap 8  --stage_mode floor  --vacuum_rest 1
run rg_floor_cfl04  --mhd 0 --mturb 40 --cfl 0.4 --nsnap 8  --stage_mode floor  --vacuum_rest 1

echo "=== GRID COMPLETE ===" | tee -a "$SUMMARY"
