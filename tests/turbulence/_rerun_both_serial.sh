#!/bin/bash
# Serial re-run of BOTH 512^3 cases, ONE AT A TIME (single resident 512^3 job).
# Two concurrent 512^3 jobs exhausted host RAM and were OOM-killed (exit 137) at
# 18:10; this script keeps them strictly sequential.
#   ICM: cfl 1.5 (subsonic, was OOM-killed before it could run)
#   ISM: cfl 1.0 (cfl 1.5 went unstable: first_bad_snap=5)
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence

run_case () {
  local label="$1"; shift
  echo "=== ${label} START $(date) ==="
  eval $(autocvd -q)
  export CUDA_VISIBLE_DEVICES
  echo "${label} using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  python paper_turbulence.py "$@" 2>&1 | tee "data_fig14/${label}.log"
  echo "=== ${label} EXIT ${PIPESTATUS[0]} $(date) ==="
}

# --- ICM first (subsonic, robust) ---
run_case ICM_N512 \
  --tag ICM_N512 --outdir data_fig14 \
  --eos iso --N 512 --mturb 0.5 --beta 1e6 \
  --F0 3.5 --cfl 1.5 --tcross 5 --nsnap 6 \
  --stage_mode redist --rhomin 0.02 --vmaxcap 50 --protect 0

# --- ISM second (hypersonic, cfl 1.0 for stability) ---
run_case ISM_N512_cfl1 \
  --tag ISM_N512 --outdir data_fig14 \
  --eos iso --N 512 --mturb 10 --beta 0.1 \
  --F0 3.5 --cfl 1.0 --tcross 5 --nsnap 6 \
  --stage_mode redist --rhomin 0.02 --vmaxcap 50 --protect 1

echo "=== ALL DONE $(date) ==="
