#!/bin/bash
# Validate fixes on the cheap reproduction (128^3, M20, rho005, cfl1.5, vmaxcap OFF -> NaN).
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence

run () {
  local tag="$1"; shift
  echo "=== ${tag} START $(date +%H:%M:%S) ==="
  eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
  echo "${tag} GPU=$CUDA_VISIBLE_DEVICES args: $*"
  python paper_turbulence.py --tag "$tag" --outdir data_repro \
    --eos iso --N 128 --beta 0.1 --F0 3.5 --tcross 4 --nsnap 8 \
    --stage_mode redist --protect 1 "$@" 2>&1 | tee "data_repro/${tag}.log" | grep -E "first_bad_snap|wall"
  echo "=== ${tag} DONE $(date +%H:%M:%S) ==="
}

mkdir -p data_repro
# reproduction config = --mturb 20 --rhomin 0.005 --cfl 1.5
run val_cap10    --mturb 20 --rhomin 0.005 --cfl 1.5 --vmaxcap 10
run val_vacrest  --mturb 20 --rhomin 0.005 --cfl 1.5 --vmaxcap 1000000000 --vacuum_rest 1
run val_codefix  --mturb 20 --rhomin 0.005 --cfl 1.5 --vmaxcap 1000000000 --cfl_vcap 1
# regression: normal stable ISM config + code fix -> must stay stable, M~10
run val_regr     --mturb 10 --rhomin 0.02  --cfl 1.5 --vmaxcap 50 --cfl_vcap 1
echo "=== VALIDATE DONE $(date +%H:%M:%S) ==="
