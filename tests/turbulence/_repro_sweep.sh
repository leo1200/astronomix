#!/bin/bash
# Cheap low-res reproduction of the ISM late-NaN + cheap-fix isolation.
# Provoke: vmaxcap off (1e9) + low rhomin + higher cfl at 128^3 -> expect NaN.
# Fixes: vmaxcap 10 or rhomin 0.05 -> expect first_bad_snap=-1.
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence

run () {
  local tag="$1"; shift
  echo "=== ${tag} START $(date +%H:%M:%S) ==="
  eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
  echo "${tag} GPU=$CUDA_VISIBLE_DEVICES  args: $*"
  python paper_turbulence.py --tag "$tag" --outdir data_repro \
    --eos iso --N 128 --mturb 10 --beta 0.1 --F0 3.5 --tcross 4 --nsnap 8 \
    --stage_mode redist --protect 1 "$@" 2>&1 | tee "data_repro/${tag}.log" | grep -E "first_bad_snap|M_turb\(|wall|eos="
  echo "=== ${tag} DONE $(date +%H:%M:%S) ==="
}

mkdir -p data_repro
# provoke (vmaxcap OFF)
run repro_cfl15_capOFF --cfl 1.5 --rhomin 0.02 --vmaxcap 1000000000
run repro_cfl20_capOFF --cfl 2.0 --rhomin 0.02 --vmaxcap 1000000000
# cheap fixes on the harder-provoke config
run fix_cap10        --cfl 2.0 --rhomin 0.02 --vmaxcap 10
run fix_rho05        --cfl 2.0 --rhomin 0.05 --vmaxcap 1000000000
echo "=== SWEEP DONE $(date +%H:%M:%S) ==="
