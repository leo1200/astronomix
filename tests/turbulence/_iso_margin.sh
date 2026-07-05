#!/bin/bash
# Is the crash marginal (sensitive to nsnap/segmentation) or robust?
# Same config that crashed in sweep2 (M20 rho005 vmaxcap-off vacuum_rest0), vary nsnap.
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
run () { local tag="$1" ns="$2"
  eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
  python paper_turbulence.py --tag "$tag" --outdir data_repro --eos iso --N 128 \
    --mturb 20 --beta 0.1 --F0 3.5 --cfl 1.5 --tcross 4 --nsnap "$ns" \
    --stage_mode redist --protect 1 --rhomin 0.005 --vmaxcap 1000000000 --vacuum_rest 0 \
    2>&1 | grep -E "first_bad_snap" | sed "s/^/[$tag ns=$ns] /"
}
run iso_ns8  8
run iso_ns60 60
run iso_ns200 200
echo "=== ISO DONE ==="
