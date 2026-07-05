#!/bin/bash
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
run () { local tag="$1" ns="$2"
  eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
  python paper_turbulence.py --tag "$tag" --outdir data_repro --eos iso --N 128 \
    --mturb 20 --beta 0.1 --F0 3.5 --cfl 1.5 --tcross 4 --nsnap "$ns" \
    --stage_mode redist --protect 1 --rhomin 0.005 --vmaxcap 1000000000 \
    --vacuum_rest 0 --nan_safe 1 \
    2>&1 | grep -E "first_bad_snap" | sed "s/^/[$tag ns=$ns] /"
}
run fw2_ns8   8
run fw2_ns60  60
run fw2_ns200 200
echo "=== FW2 DONE ==="
