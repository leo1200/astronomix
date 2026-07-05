#!/bin/bash
# Escalated provoke: the instability is the near-vacuum runaway-velocity
# mechanism, strongest with DEEP voids. Provoke at low res via higher Mach
# (more compressible -> deeper voids) and lower floor, vmaxcap OFF. Then
# isolate: re-run the FIRST config that breaks with vmaxcap 10 / rhomin 0.05.
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence

run () {
  local tag="$1" N="$2" M="$3"; shift 3
  echo "=== ${tag} START $(date +%H:%M:%S) ==="
  eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
  echo "${tag} GPU=$CUDA_VISIBLE_DEVICES N=$N mturb=$M args: $*"
  python paper_turbulence.py --tag "$tag" --outdir data_repro \
    --eos iso --N "$N" --mturb "$M" --beta 0.1 --F0 3.5 --tcross 4 --nsnap 8 \
    --stage_mode redist --protect 1 "$@" 2>&1 | tee "data_repro/${tag}.log" | grep -E "first_bad_snap|wall|eos="
  echo "=== ${tag} DONE $(date +%H:%M:%S) ==="
}

mkdir -p data_repro
# higher Mach -> deeper voids (128^3, vmaxcap OFF, low floor)
run prov_M20_rho02_capOFF 128 20 --cfl 1.5 --rhomin 0.02  --vmaxcap 1000000000
run prov_M20_rho005_capOFF 128 20 --cfl 1.5 --rhomin 0.005 --vmaxcap 1000000000
run prov_M40_rho005_capOFF 128 40 --cfl 1.5 --rhomin 0.005 --vmaxcap 1000000000
# resolution fallback (closer to where it actually broke)
run prov_256_M10_capOFF   256 10 --cfl 1.8 --rhomin 0.02  --vmaxcap 1000000000
echo "=== SWEEP2 DONE $(date +%H:%M:%S) ==="
