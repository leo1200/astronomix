#!/bin/bash
# Push pure hydro toward a crash: deeper voids (higher Mach, lower floor) + higher cfl.
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
run () { local tag="$1" M="$2" rho="$3" cfl="$4" ns="$5"
  eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
  python paper_turbulence.py --tag "$tag" --outdir data_repro --eos iso --N 128 --mhd 0 \
    --mturb "$M" --beta 0.1 --F0 3.5 --cfl "$cfl" --tcross 4 --nsnap "$ns" \
    --stage_mode redist --protect 1 --rhomin "$rho" --vmaxcap 1000000000 \
    --vacuum_rest 0 --nan_safe 0 \
    2>&1 | grep -E "first_bad_snap" | sed "s/^/[$tag hydro M=$M rho=$rho cfl=$cfl] /"
}
run hyd_M40_r005       40 0.005 1.5 8
run hyd_M40_r001       40 0.001 1.5 8
run hyd_M60_r002       60 0.002 1.5 8
run hyd_M40_r005_cfl2  40 0.005 2.0 8
run hyd_M40_r0005_cfl2 40 0.0005 2.0 8
echo "=== HYDRO EXTREME DONE ==="
