#!/bin/bash
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
run () { local tag="$1" cfl="$2" pp="$3" ns="$4"
  eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
  python paper_turbulence.py --tag "$tag" --outdir data_repro --eos iso --N 128 --mhd 0 \
    --mturb 40 --beta 0.1 --F0 3.5 --cfl "$cfl" --tcross 4 --nsnap "$ns" \
    --stage_mode redist --protect 1 --rhomin 0.005 --vmaxcap 1000000000 \
    --vacuum_rest 0 --nan_safe 0 --pp_flux "$pp" \
    2>&1 | grep -E "first_bad_snap" | sed "s/^/[$tag cfl=$cfl pp=$pp ns=$ns] /"
}
run base_cfl10   1.0 0 8
run pp_cfl10_n8  1.0 1 8
run pp_cfl10_n60 1.0 1 60
run pp_cfl05_n8  0.5 1 8
echo "=== PP TEST DONE ==="
