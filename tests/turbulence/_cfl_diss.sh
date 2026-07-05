#!/bin/bash
# Does more dissipation (lower cfl) robustly PREVENT the crash (no masking)?
# Marginal crasher (M20 rho005 nsnap8 vmaxcap-off), NO firewall/vacuum_rest, vary cfl.
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
run () { local cfl="$1"
  eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
  python paper_turbulence.py --tag cfl_${cfl/./} --outdir data_repro --eos iso --N 128 \
    --mturb 20 --beta 0.1 --F0 3.5 --cfl "$cfl" --tcross 4 --nsnap 8 \
    --stage_mode redist --protect 1 --rhomin 0.005 --vmaxcap 1000000000 \
    --vacuum_rest 0 --nan_safe 0 \
    2>&1 | grep -E "first_bad_snap" | sed "s/^/[cfl=$cfl] /"
}
run 1.0; run 0.7; run 0.4
echo "=== CFL DISS DONE ==="
